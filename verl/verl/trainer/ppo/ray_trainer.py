# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from tqdm import tqdm

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import random
from collections import defaultdict

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, mask_truncated_samples=False, corr_list=None, repetition_penalty_list=None):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch['token_level_rewards'],
            values=data.batch['values'],
            eos_mask=data.batch['response_mask'],
            gamma=gamma,
            lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            eos_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'],
            mask_truncated_samples=mask_truncated_samples,
            corr_list=corr_list,
            repetition_penalty_list=repetition_penalty_list)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], eos_mask=data.batch['response_mask'], gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            reward_baselines=data.batch['reward_baselines'],
            eos_mask=data.batch['response_mask'])

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            eos_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         max_response_length=self.config.data.max_response_length,
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.get('truncation', 'error'),
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch_padded.meta_info['val_temperature'] = self.config.actor_rollout_ref.rollout.val_temperature
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')

        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _rm_neg_samples(self, batch: DataProto, corr_list, reserve_prob=0.8, threshold=0):
        world_size = self.actor_rollout_wg.world_size
        reserve_list = []
        corr_count = 0
        for if_corr in corr_list:
            if if_corr:
                reserve_list.append(True)
                corr_count += 1
                continue
            num = random.uniform(0, 1)
            if num < reserve_prob:
                reserve_list.append(True)
            else:
                reserve_list.append(False)
        # print(f"Correctness count: {corr_count}")

        def _ensure_true_threshold(mask, threshold):
            """
            Count the number of True values in mask. If it’s less than the threshold,
            randomly select positions of False and set them to True until the threshold is met.

            :param mask: List[bool] - A list containing only True and False
            :param threshold: int - The minimum number of True values needed
            :return: List[bool] - The modified list
            """
            num_to_flip = threshold - true_count
            if num_to_flip == 0:
                return mask
            if num_to_flip > 0:
                false_indices = [i for i, val in enumerate(mask) if not val]
                indices_to_flip = random.sample(false_indices, num_to_flip)
                for idx in indices_to_flip:
                    mask[idx] = True
                return mask
            else:
                num_to_flip = -num_to_flip
                true_indices = [i for i, val in enumerate(mask) if val]
                indices_to_flip = random.sample(true_indices, num_to_flip)
                for idx in indices_to_flip:
                    mask[idx] = False
                return mask
        
        true_count = sum(reserve_list)
        if true_count < threshold:
            reserve_list = _ensure_true_threshold(reserve_list, threshold)
        true_count = sum(reserve_list)
        if true_count % world_size != 0: # Ensure that at this point, the batch can be evenly divided by the number of GPUs.
            reserve_list = _ensure_true_threshold(reserve_list, world_size*(true_count//world_size))
        
        if sum(reserve_list) == 0: # When positive and negative examples are both at 100% and nothing remains after cutoff, directly return an empty result.
            return None

        dict_batch = {}
        meta_info = {}
        mask = torch.tensor(reserve_list, dtype=torch.bool)
        for key, tensor in batch.batch.items():
            dict_batch[key] = tensor[mask]
        
        mask = np.array(reserve_list, dtype=bool)
        for key, numpy_array in batch.non_tensor_batch.items():
            dict_batch[key] = numpy_array[mask]
        
        for key, value in batch.meta_info.items():
            if isinstance(value, list):
                meta_info[key] = [val for i, val in enumerate(value) if reserve_list[i]]
            else:
                meta_info[key] = value

        processed_batch: DataProto = DataProto.from_single_dict(dict_batch, meta_info)
        return processed_batch
    
    def _inject_offline_samples(self, batch: DataProto, corr_list, offline_samples_alpha=0.0, max_inject_num=-1):
        bsz, _ = batch.batch["prompts"].shape
        index = batch.non_tensor_batch['uid']
        uid2replace_idx = defaultdict(list)
        uid2all_idx = defaultdict(list)
        uid2score = defaultdict(list)
        for i in range(bsz):
            uid2score[index[i]].append(corr_list[i])
            uid2all_idx[index[i]].append(i)
            if not corr_list[i]:
                uid2replace_idx[index[i]].append(i)

        dict_batch = {}
        dict_batch["responses"] = batch.batch["responses"].clone()
        dict_batch["input_ids"] = batch.batch["input_ids"].clone()
        dict_batch["attention_mask"] = batch.batch["attention_mask"].clone()
        dict_batch["position_ids"] = batch.batch["position_ids"].clone()
        dict_batch["prompts"] = batch.batch["prompts"]

        valid_replace_idx = []
        for uid in uid2score:
            corr_ratio = sum(uid2score[uid]) / len(uid2score[uid])
            if corr_ratio <= offline_samples_alpha:
                replace_idx_list = uid2replace_idx[uid]
                offline_samples_list = batch.non_tensor_batch['offline_samples_list'][uid2all_idx[uid][0]]
                curr_max_inject_num = min(max_inject_num if max_inject_num > 0 else 9999, len(replace_idx_list), len(offline_samples_list))
                print(f"Inject {curr_max_inject_num} offline samples for uid {uid} with original correct-ratio {corr_ratio}")
                for i in range(curr_max_inject_num):
                    replace_idx = replace_idx_list[i]
                    offline_sample = offline_samples_list[i]
                    dict_batch["responses"][replace_idx] = offline_sample["responses"]
                    dict_batch["input_ids"][replace_idx] = offline_sample["input_ids"]
                    dict_batch["attention_mask"][replace_idx] = offline_sample["attention_mask"]
                    dict_batch["position_ids"][replace_idx] = offline_sample["position_ids"]
                    valid_replace_idx.append(replace_idx)

        if self.config.actor_rollout_ref.actor.algorithm.enable_luffy_loss:
            dict_batch["offline_sample_mask"] = torch.tensor([1 if i in valid_replace_idx else 0 for i in range(bsz)], dtype=torch.bool)

        # For non-tensor batch, directly use the original batch.
        for key, numpy_array in batch.non_tensor_batch.items():
            dict_batch[key] = numpy_array

        processed_batch: DataProto = DataProto.from_single_dict(dict_batch, batch.meta_info)
        return processed_batch
    
    def _count_low_accuracy_questions(self, uid_list, correctness_list, threshold):
        stats = {}
        for uid, is_correct in zip(uid_list, correctness_list):
            if uid not in stats:
                stats[uid] = {'total': 0, 'correct': 0}
            stats[uid]['total'] += 1
            if is_correct:
                stats[uid]['correct'] += 1

        count = 0
        for uid, data in stats.items():
            accuracy = data['correct'] / data['total']
            if accuracy < threshold:
                count += 1
        return count

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # Our offline samples need to be injected here, before starting the ref, adv, etc. steps.
                    if self.config.algorithm.enable_offline_samples:
                        _, corr_list, _ = self.reward_fn(batch, return_corr_list=True)
                        need_inject_samples_num = self._count_low_accuracy_questions(batch.non_tensor_batch['uid'], corr_list, self.config.algorithm.offline_samples_alpha)
                        print(f"#### {sum(corr_list)} correct samples ([BEFORE] inject offline samples), {need_inject_samples_num} problems need offline samples")
                        batch = self._inject_offline_samples(batch, corr_list, self.config.algorithm.offline_samples_alpha, self.config.algorithm.max_inject_num)

                    batch.batch['response_mask'] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, corr_list, repetition_penalty_list = self.reward_fn(batch, return_corr_list=True, enble_relative_length_reward=self.config.algorithm.enble_relative_length_reward, rlr_args={"corr_ratio": self.config.algorithm.rlr_args_corr_ratio, "upper_bound": self.config.algorithm.rlr_args_upper_bound, "lower_bound": self.config.algorithm.rlr_args_lower_bound, "length_base": self.config.algorithm.rlr_args_length_base}, enble_repetition_reward=self.config.algorithm.enble_repetition_reward, rr_args={"ngram_size": self.config.algorithm.rr_args_ngram_size, "max_penalty": self.config.algorithm.rr_args_max_penalty})
                        if self.config.algorithm.enable_offline_samples:
                            need_inject_samples_num = self._count_low_accuracy_questions(batch.non_tensor_batch['uid'], corr_list, self.config.algorithm.offline_samples_alpha)
                            print(f"#### {sum(corr_list)} correct samples ([AFTER] inject offline samples), {need_inject_samples_num} problems need offline samples")
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  mask_truncated_samples=self.config.algorithm.mask_truncated_samples,
                                                  corr_list=corr_list,
                                                  repetition_penalty_list=repetition_penalty_list)
                        
                        # Here, we prune negative or positive samples here to ensure that the GRPO adv calculation is correct.
                        original_bsz = len(corr_list)
                        original_corr_samples = sum(corr_list)
                        if self.config.algorithm.cut_down_negative_samples:
                            batch = self._rm_neg_samples(batch, corr_list, self.config.algorithm.cut_down_samples_reserve_ratio, self.config.algorithm.cut_down_samples_reserve_min_num)
                            if batch is None:
                                print("##### WARNING: no samples left after cutting down")
                                progress_bar.update(1)
                                self.global_steps += 1
                                continue
                            # print("pruned tensor", batch.batch['token_level_scores'].shape)
                            _, corr_list, _ = self.reward_fn(batch, return_corr_list=True)
                            print(f"#### Total {len(corr_list)} samples, {sum(corr_list)} correct samples (after cutting down {self.config.algorithm.cut_down_samples_reserve_ratio} [negative] samples)")
                            metrics.update({"original_bsz": original_bsz, "original_corr_samples": original_corr_samples, "cut_down_negative_reserve_samples": len(corr_list), "cut_down_negative_corr_samples": sum(corr_list)})
                        elif self.config.algorithm.cut_down_positive_samples:
                            corr_list = [not x for x in corr_list]
                            batch = self._rm_neg_samples(batch, corr_list, self.config.algorithm.cut_down_samples_reserve_ratio, self.config.algorithm.cut_down_samples_reserve_min_num)
                            if batch is None:
                                print("##### WARNING: no samples left after cutting down")
                                progress_bar.update(1)
                                self.global_steps += 1
                                continue
                            # print("pruned tensor", batch.batch['token_level_scores'].shape)
                            _, corr_list, _ = self.reward_fn(batch, return_corr_list=True)
                            print(f"#### Total {len(corr_list)} samples, {sum(corr_list)} correct samples (after cutting down {self.config.algorithm.cut_down_samples_reserve_ratio} [positive] samples)")
                            metrics.update({"original_bsz": original_bsz, "original_corr_samples": original_corr_samples, "cut_down_positive_reserve_samples": len(corr_list), "cut_down_positive_corr_samples": sum(corr_list)})

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            pprint(f'Current validation metrics: {val_metrics}')
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

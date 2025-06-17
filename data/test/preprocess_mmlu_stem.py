import json
import string

input_path = './Orig.mmlu_stem.jsonl'
output_path = './mmlu_stem.json'

letters = list(string.ascii_uppercase)

dataset = []
with open(input_path, 'r', encoding='utf-8') as fin, \
     open(output_path, 'w', encoding='utf-8') as fout:

    for line in fin:
        data = json.loads(line)
        question = data.get('question', '').strip()
        choices = data.get('choices', [])
        answer_idx = data.get('answer', 0)

        parts = [question, 'Answer Choices:']
        for idx, choice in enumerate(choices):
            label = letters[idx]
            parts.append(f"({label}) {choice.strip()}")
        problem = " \n".join(parts)

        answer = letters[answer_idx]

        dataset.append({
            'problem': problem,
            'answer': answer
        })

    fout.write(json.dumps(dataset, ensure_ascii=False, indent=2) + '\n')

print(f'Saved to {output_path}')

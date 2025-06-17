import json

input_path = './Orig.gaokao_math_qa.jsonl'
output_path = './gaokao_math_qa.json'

dataset = []
with open(input_path, 'r', encoding='utf-8') as fin, \
     open(output_path, 'w', encoding='utf-8') as fout:

    for line in fin:
        data = json.loads(line)
        question = data.get('question', '').strip()
        options = data.get('options', {})

        parts = [question, '从以下选项中选择:'] # construct Chinese prompt
        for key in sorted(options.keys()):
            parts.append(f"({key}) {options[key].strip()}")
        problem = "\n".join(parts)

        answer = data.get('label', '').strip()
        dataset.append({
            'problem': problem,
            'answer': answer
        })

    fout.write(json.dumps(dataset, ensure_ascii=False, indent=2) + '\n')

print(f'Saved to {output_path}')

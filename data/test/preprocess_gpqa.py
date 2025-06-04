import csv
import json
import random
from pathlib import Path

input_path = Path('./gpqa_diamond.csv')
output_path = Path('./gpqa_diamond.json')
random.seed(42)

records = []
letters = ['A', 'B', 'C', 'D']

with input_path.open('r', encoding='utf-8', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        q = row['Question'].strip()
        correct = row['Correct Answer'].strip()
        wrongs = [
            row['Incorrect Answer 1'].strip(),
            row['Incorrect Answer 2'].strip(),
            row['Incorrect Answer 3'].strip()
        ]

        choices = [correct] + wrongs
        random.shuffle(choices)

        lines = [q, 'Answer Choices:']
        for label, text in zip(letters, choices):
            lines.append(f"({label}) {text}")
        problem = "\n".join(lines)

        answer_label = letters[choices.index(correct)]

        records.append({
            'problem': problem,
            'answer': answer_label
        })

with output_path.open('w', encoding='utf-8') as fout:
    json.dump(records, fout, ensure_ascii=False, indent=2)

print(f"Saved to {output_path}")

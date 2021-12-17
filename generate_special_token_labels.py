from pathlib import Path
"""
This script extracts the Constituency labels from augmented data and generates special tokens for training using BERT Based Pre-trained  LM 
"""


data_dir = 'data'
token_labels = set(['-LBRAC-', '-RBRAC-'])

for data_path in Path('data').glob('**/seq.in'):
    with open(data_path) as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                if token.startswith('-LBRAC') or token.startswith('-RBRAC'):
                    token_labels.add(token)

token_labels = sorted(list(token_labels))
print("token_labels:", token_labels)
for ddir in Path('data').glob('*/'):
    out_path = ddir/'special_token_label.txt'
    with open(out_path, 'w') as wf:
        for token in token_labels:
            wf.write(token + '\n')
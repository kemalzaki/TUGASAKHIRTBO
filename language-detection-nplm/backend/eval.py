import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
from model import load_or_create_model, predict_text
from sklearn.metrics import classification_report, confusion_matrix


def load_dataset_labels(dataset_dir):
    files = {'eng.txt':'eng','ind.txt':'ind','sun.txt':'sun'}
    texts = []
    labels = []
    for fname, lbl in files.items():
        path = os.path.join(dataset_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                t = line.strip()
                if not t:
                    continue
                texts.append(t)
                labels.append(lbl)
    return texts, labels


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    dataset_dir = os.path.join(root, '..', 'dataset')
    model, vectorizer = load_or_create_model()

    texts, y_true = load_dataset_labels(dataset_dir)
    y_pred = []
    for t in texts:
        p, conf = predict_text(t, model, vectorizer)
        y_pred.append(p)

    labels = ['eng','ind','sun']
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = len(y_true)
    acc = sum(1 for a,b in zip(y_true,y_pred) if a==b) / total if total else 0

    out_lines = []
    out_lines.append('# Evaluation Report')
    out_lines.append('')
    out_lines.append(f'- Total samples: {total}')
    out_lines.append(f'- Accuracy: {acc:.4f}')
    out_lines.append('')
    out_lines.append('## Classification Report')
    out_lines.append('')
    out_lines.append('```')
    out_lines.append(report)
    out_lines.append('```')
    out_lines.append('')
    out_lines.append('## Confusion Matrix (rows=true, cols=pred)')
    out_lines.append('')
    out_lines.append('```')
    out_lines.extend(['\t'.join(map(str,row)) for row in cm])
    out_lines.append('```')

    report_path = os.path.join(root, 'eval_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))

    print('Wrote evaluation report to', report_path)
    print('\nSummary:\n')
    print('\n'.join(out_lines[:8]))

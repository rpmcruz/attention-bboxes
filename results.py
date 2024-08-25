import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results')
args = parser.parse_args()

import pandas as pd
import numpy as np

df = pd.read_csv(args.results)
df['l1'] = [model.split('-')[4] if model.count('-') >= 5 else '' for model in df['model']]
df['heatmap'] = [model.split('-')[6] if model.count('-') >= 7 else '' for model in df['model']]
df['model'] = [model.split('-')[2] for model in df['model']]

# choose highest pg score
score = 'degscore'
def get_max_pg_row(group):
    if group[score].notna().any():
        return group.loc[group[score].idxmax()]
    else:
        return group.iloc[0]  # if NaN, return the first row
df = df.groupby(['model', 'dataset', 'heatmap'], as_index=False).apply(get_max_pg_row, include_groups=False)

# re-order rows
dataset_order = ['Birds', 'StanfordCars', 'StanfordDogs']
model_order = ['OnlyClass', 'ProtoPNet', 'Heatmap', 'SimpleDet', 'FasterRCNN', 'FCOS', 'DETR']
df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)
df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
df = df.sort_values(by=['dataset', 'model']).reset_index(drop=True)

# concat model and heatmap
df['model'] = np.where(df['heatmap'] != '', df['model'].astype(str) + ' w/ ' + df['heatmap'], df['model'])

# repeated dataset make as idem
df['dataset'] = df['dataset'].astype(str)
df.loc[df['dataset'].duplicated(), 'dataset'] = '``'

# select columns
df = df[['dataset', 'model', 'acc', 'pg', 'degscore', 'sparsity']]
df[['acc', 'pg', 'degscore', 'sparsity']] = df[['acc', 'pg', 'degscore', 'sparsity']]*100
df = df.rename(columns={'dataset': r'\bf Dataset', 'model': r'\bf Model', 'acc': r'\bf Accuracy', 'pg': r'\bf Pointing Game', 'degscore': r'\bf Degradation Score', 'sparsity': r'\bf Sparsity'})

latex = df.style \
      .hide(axis=0) \
      .format(precision=0) \
      .background_gradient('RdYlGn') \
      .to_latex(convert_css=True, column_format='|llrrrr|')
latex = latex.split('\n')
old = ''
for i, line in enumerate(latex):
    if line == '':
        continue
    if i > 0 and '``' not in line.split('&')[0]:
        old = line.split('&')[0]
        print(r'\hline')
    print(line)
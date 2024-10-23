import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results')
parser.add_argument('--adversarial', action='store_true')
args = parser.parse_args()

import pandas as pd
import re

pd.set_option('display.max_columns', None)

df = pd.read_csv(args.results)

df['l1'] = [re.search(r'-l1-([\d.]+)-', model).group(1) if '-l1-' in model else '' for model in df['model']]
df['heatmap'] = [re.search(r'-heatmap-(\w+)-', model).group(1)[:-7] if '-heatmap-' in model else '' for model in df['model']]
df['occlusion'] = [re.search(r'-occlusion-(\w+)-', model).group(1) if '-occlusion-' in model else '' for model in df['model']]
df['adversarial'] = ['1' if model.endswith('-adversarial') else '0' if model.endswith('-sigmoid') else '' for model in df['model']]
df['objdet'] = [model.split('-')[2] for model in df['model']]
df['model2'] = [row['xai'] if row['model'].endswith('-OnlyClass') else f'ProtoPNet (train=full/test={'crop' if row["crop"] else 'full'})' if row['model'].endswith('-ProtoPNet') else f'ProtoPNet (train=crop/test={'crop' if row["crop"] else 'full'})' if row['model'].endswith('-ProtoPNet-crop') else f"Proposal ({row['objdet']})" for _, row in df.iterrows()]

# filter options
df = df[df['adversarial'] == args.adversarial]

# sort
model_order = ['OnlyClass', 'ProtoPNet', 'Heatmap', 'SimpleDet', 'FasterRCNN', 'FCOS', 'DETR']
df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
df = df.sort_values(by=['model', 'heatmap', 'l1']).reset_index(drop=True)

# choose highest pg score
'''
score = 'degscore'
def get_max_pg_row(group):
    if group[score].notna().any():
        return group.loc[group[score].idxmax()]
    else:
        return group.iloc[0]  # if NaN, return the first row
df = df.groupby(['model', 'dataset', 'heatmap'], as_index=False).apply(get_max_pg_row, include_groups=False)
'''

print(r'\documentclass{article}')
print(r'\usepackage[a4paper, margin=2cm]{geometry}')
print(r'\usepackage[table]{xcolor}')
print(r'\begin{document}')
print(r'\hspace{-5em}')
latex = df.style \
      .hide(axis=0) \
      .background_gradient('RdYlGn') \
      .to_latex(convert_css=True)#, column_format='|llrrrr|')
#.format(precision=0) \
last_model = None
for i, line in enumerate(latex.splitlines()):
    if i > 0 and line.split()[0] != last_model:
        print(r'\hline')
        last_model = line.split()[0]
    print(line)
print(r'\end{document}')

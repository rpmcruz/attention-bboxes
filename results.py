import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results')
parser.add_argument('--gauss', action='store_true')
args = parser.parse_args()

import pandas as pd
import re
from itertools import groupby
df = pd.read_csv(args.results)

# only bigger version of datasets
datasets = list(df['dataset'].unique()[[i for i, (d1, d2) in enumerate(zip(df['dataset'].unique(), df['dataset'].unique()[1:])) if re.match(r'([A-Za-z]+)', d1).group(1) != re.match(r'([A-Za-z]+)', d2).group(1)]]) + [df['dataset'].unique()[-1]]
datasets += [d for d in df['dataset'].unique() if re.match(r'^[A-Za-z]+$', d)]
df = df.loc[df['dataset'].isin(datasets)]
# drop ProtoPNet-crop and crop=True
df = df.loc[df['model'] != 'model-Birds-ProtoPNet-crop']
df = df.loc[df['crop'] == False]
# drop OnlyClass except the following xAI methods
xai_methods = ['CAM', 'GradCAM', 'DeepLIFT', 'Occlusion', 'IBA']
df = df.loc[(df['model'] != 'OnlyClass') | (df['xai'].isin(xai_methods))]
# heatmap only LogisticHeatmap (ignore Gauss)
if args.gauss:
    df = df.loc[~df['model'].str.contains('-LogisticHeatmap-')]
else:
    df = df.loc[~df['model'].str.contains('-GaussHeatmap-')]

# rename model column
df['model'] = [m.split('-')[2] for m in df['model']]
df['dataset'] = [re.match(r'([A-Za-z]+)', d).group(1) for d in df['dataset']]
df.loc[df['model'] == 'OnlyClass', 'model'] = df.loc[df['model'] == 'OnlyClass', 'xai']

# choose highest score
def get_max_score(group):
    score = 'degscore'
    if group[score].notna().any():
        return group.loc[group[score].idxmax()]
    else:
        return group.iloc[0]  # if NaN, return the first row
df = df.groupby(['dataset', 'model'], as_index=False, observed=True).apply(get_max_score, include_groups=False)

# sort rows
model_order = xai_methods + ['ProtoPNet', 'ViTb', 'ViTl', 'ViTr', 'Heatmap', 'SimpleDet', 'FasterRCNN', 'FCOS', 'DETR']
df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
df = df.sort_values(by=['dataset', 'model']).reset_index(drop=True)

# columns order and filter
df = df[['dataset', 'model', 'acc', 'degscore', 'pg', 'density', 'totalvariance', 'entropy']]

print(r'\documentclass{standalone}')
print(r'\usepackage[table]{xcolor}')
print(r'\begin{document}')
print(r'\footnotesize')
print(df.style
    .hide(axis=0)
    .background_gradient('RdYlGn')
    .format({col: lambda x: f'{x*100:.1f}' if type(x) == float else x for col in df.columns})
    .to_latex(convert_css=True))
print(r'\end{document}')

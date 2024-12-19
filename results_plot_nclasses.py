import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results')
args = parser.parse_args()

import pandas as pd
import re
df = pd.read_csv(args.results)

# filter heatmap (Logistic)
df = df.loc[~df['model'].str.contains('-heatmap-GaussHeatmap-')]
# filter models
df = df.loc[
    (df['model'].str.endswith('-OnlyClass') & df['xai'] == 'GradCAM') |
    df['model'].str.endswith('-ViTb') |
    df['model'].str.endswith('-ViTr') |
    df['model'].str.contains('-FasterRCNN-') |
    df['model'].str.contains('-FCOS-') |
    df['model'].str.contains('-DETR-')]

df = pd.DataFrame({
    'model': ['Grad-CAM' if s.split('-')[2] == 'OnlyClass' else s.split('-')[2] for s in df['model']],
    'nclasses': [int(re.search(r'\d+$', s).group()) for s in df['dataset']],
    'acc': df['acc'],
    'degscore': df['degscore'],
})

# choose highest score
def get_max_score(group):
    score = 'degscore'
    if group[score].notna().any():
        return group.loc[group[score].idxmax()]
    else:
        return group.iloc[0]  # if NaN, return the first row
df = df.groupby(['model', 'nclasses'], as_index=False, observed=True).apply(get_max_score, include_groups=False)

# long to wide (easier for pgfplots)
df = df.pivot(columns='model', index='nclasses', values=['acc', 'degscore'])
df.columns = [f'{x}_{y}' for x, y in df.columns]

df.sort_index(inplace=True)
df.to_csv(args.results[:-4] + '-nclasses.csv')
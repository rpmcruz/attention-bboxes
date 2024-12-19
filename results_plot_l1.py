import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results')
args = parser.parse_args()

import pandas as pd
import re
df = pd.read_csv(args.results)

# filter heatmap (Logistic)
df = df.loc[~df['model'].str.contains('-heatmap-GaussHeatmap-')]
# filter dataset
df = df.loc[df['dataset'].str.endswith('200')]
# filter models
df = df.loc[
    df['model'].str.contains('-FasterRCNN-') |
    df['model'].str.contains('-FCOS-') |
    df['model'].str.contains('-DETR-')]

df = pd.DataFrame({
    'model': [s.split('-')[2] for s in df['model']],
    'l1': [s.split('-')[4] for s in df['model']],
    'acc': df['acc'],
    'degscore': df['degscore'],
    'density': df['density'],
    'totalvariance': df['totalvariance'],
    'entropy': df['entropy'],
})

# long to wide (easier for pgfplots)
df = df.pivot(columns='model', index='l1', values=['acc', 'degscore', 'density', 'totalvariance', 'entropy'])
df.columns = [f'{x}_{y}' for x, y in df.columns]

df.sort_index(inplace=True)
df.to_csv(args.results[:-4] + '-l1.csv')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results')
args = parser.parse_args()

import pandas as pd
df = pd.read_csv(args.results)

# filter models
df = df.loc[
    df['model'].str.endswith('-OnlyClass') &
    df['model'].str.endswith('-ViTb') &
    df['model'].str.endswith('-ViTr') &
    df['model'].str.endswith('-FasterRCNN') &
    df['model'].str.endswith('-FCOS') &
    df['model'].str.endswith('-DETR')]


df.drop(columns=['captum', 'crop'], inplace=True)

# modify columns
df['l1'] = [model.split('-')[4] if model.count('-') >= 5 else '' for model in df['model']]
df['heatmap'] = [model.split('-')[6] if model.count('-') >= 7 else '' for model in df['model']]
df['model'] = [model.split('-')[2] for model in df['model']]

# drop some columns and rows
df = df.loc[df['dataset'] == 'Birds']
df = df.loc[df['heatmap'] == 'LogisticHeatmap']
df = df.drop(columns=['dataset', 'heatmap'])
df = df[df['model'].isin(['FasterRCNN', 'FCOS', 'DETR'])]

df.to_csv(args.output, index=False)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results')
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

for dataset in ['Birds', 'StanfordCars', 'StanfordDogs']:
    for occlusion in ['encoder', 'image']:
        for adversarial in ['0', '1']:
            for heatmap in ['Logistic', 'Gauss']:
                df2 = df.loc[df['dataset'].isin([dataset]) & df['occlusion'].isin([occlusion, '']) & df['adversarial'].isin([adversarial, '']) & df['heatmap'].isin([heatmap, ''])]
                df2 = df2[['dataset', 'model2', 'l1', 'acc', 'pg', 'degscore', 'sparsity']]
                df2 = df2.sort_values(['dataset', 'model2', 'l1'],
                    key=lambda col: col.map({
                        **dict(zip(df2['dataset'].unique(), range(len(df2['dataset'].unique())))),
                        **dict(zip(df2['model2'].unique(), range(len(df2['model2'].unique()))))}).fillna(col))
                df2.loc[:, 'dataset'] = [df2.loc[ix, 'dataset'] if prev == None or df2.loc[prev, 'dataset'] != df2.loc[ix, 'dataset'] else '``' for prev, ix in zip([None] + list(df2.index), df2.index)]
                out = open(f'results-dataset-{dataset}-occlusion-{occlusion}-adversarial-{adversarial}-heatmap-{heatmap}.tex', 'w')
                print(r'\documentclass{standalone}', file=out)
                print(r'\usepackage[table]{xcolor}', file=out)
                print(r'\begin{document}', file=out)
                latex = df2.style \
                    .hide(axis=0) \
                    .background_gradient('RdYlGn') \
                    .to_latex(convert_css=True)
                print(latex, file=out)
                print(r'\end{document}', file=out)
                out.close()
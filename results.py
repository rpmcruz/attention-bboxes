import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results')
args = parser.parse_args()

import pandas as pd
import re

df = pd.read_csv(args.results)

df['l1'] = [re.search(r'-l1-([\d.]+)-', model).group(1) if '-l1-' in model else '' for model in df['model']]
df['heatmap'] = [re.search(r'-heatmap-(\w+)-', model).group(1)[:-7] if '-heatmap-' in model else '' for model in df['model']]
df['adversarial'] = [model.endswith('-adversarial') for model in df['model']]
df['model'] = [model.split('-')[2] for model in df['model']]

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

print(r'\documentclass{standalone}')
print(r'\usepackage[table]{xcolor}')
print(r'\begin{document}')
latex = df.style \
      .hide(axis=0) \
      .background_gradient('RdYlGn') \
      .to_latex(convert_css=True)#, column_format='|llrrrr|')
#.format(precision=0) \
print(latex)
print(r'\end{document}')
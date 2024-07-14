import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results')
args = parser.parse_args()

for result in open(args.results):
    result = result.split()
    if len(result) == 5 and result[0].endswith('.pth'):
        params = result[0][:-4].split('-')
        l1 = {p1: p2 for p1, p2 in zip(params[3::2], params[4::2])}.get('l1', '-')
        heatmap = {p1: p2 for p1, p2 in zip(params[3::2], params[4::2])}.get('heatmap', '-')
        act = params[-1] if params[-1] in ('sigmoid', 'softmax') else '-'
        print(params[1], params[2], l1, heatmap, act, result[1], result[2], result[4])
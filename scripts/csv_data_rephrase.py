import pandas as pd
from trustyai.detoxify import TMaRCo

import numpy as np
from tqdm import tqdm
import argparse
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default='jigsaw-toxic-comment-classification-challenge/test.csv')
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--diff", type=str, default='')
    parser.add_argument("--models", type=str, nargs='+', default=['trustyai/gminus', 'trustyai/gplus'])
    parser.add_argument("--weights", type=float, nargs='+', default=[-0.9, 2.5])
    parser.add_argument("--kind", type=str, default='rephrase', choices=['reflect', 'rephrase'])
    parser.add_argument("--chat_model", type=str)
    args = parser.parse_args()

    tqdm.pandas()
    tmarco = TMaRCo(expert_weights=args.weights)
    tmarco.load_models(args.models)
    data = pd.read_csv(args.csv)[:args.samples]
    kind = args.kind
    chat_model = args.chat_model
    print(data.shape)
    new_rows = []
    changes = []


    def process_data(row):
        nr = row.copy()
        for c in nr.index:
            if data[c].dtype != np.number:
                value = nr[c]
                if kind == 'rephrase':
                    detoxified = tmarco.rephrase(value, tmarco.mask(value), combine_original=True)
                elif kind == 'reflect':
                    if chat_model is not None:
                        detoxified = tmarco.reflect(value, chat_model=chat_model)[0]
                    else:
                        detoxified = tmarco.reflect(value)[0]
                else:
                    raise Exception(f'unexpected kind "{kind}"')
                nr[c] = detoxified
                changes.append({'original': value, 'rephrased': detoxified})
        new_rows.append(nr)


    data.progress_apply(process_data, axis=1)
    output_file = args.output
    diff_file = args.diff if args.diff != '' else output_file + "-diff.json"
    pd.concat(new_rows).to_csv(output_file)
    if len(changes) > 0:
        changes_json = json.dumps(changes)
        with open(diff_file, "w") as outfile:
            json.dump(changes, outfile)

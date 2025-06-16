import pandas as pd
from os.path import exists
from config import cfg
from sklearn import metrics
import itertools

from src.utils import create_file
from src.segment_matching import SegmentMatcher


def evaluate_matching_approaches_separately(overwrite=False):
    '''
    Main method for evaluating different matching approaches separately
    for different thresholds
    - compute similarity between manual segments and PaLM-generated segments
    '''

    outpath = cfg["split_data"] + "matching_evaluation/split_matching_results.csv"
    if exists(outpath) and not overwrite:
        print("load matching results:")
        df = pd.read_csv(outpath)
        print(df)
        exit()
    else:
        print("compute matching results")
        ngram = 3
        gpt_prompt = "Give me the percentage of the content of text A contained in text B. Text A: AAA, Text B: BBB"
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        approaches = ["ags21"] # "ngram", "sequencematcher", "transformer", "ags21", "gpt4"
        seg_matcher = SegmentMatcher(approaches, thresholds, ngram, gpt_prompt)
        dfout = pd.DataFrame([""] + thresholds, columns=["threshold"])
        for approach in approaches:
            sub_df = seg_matcher.evaluate_approach(approach)
            dfout = pd.concat([dfout, sub_df], axis=1)
        dfout.to_csv(outpath, index=False)
        return dfout


def evaluate_matching_approaches_combinations(get_best=False):
    '''
    Main method for evaluating the combination of different similarity approaches
    for different thresholds
    - compute similarity between manual segments and PaLM-generated segments
    '''
    if get_best:
        get_best_results()
    outfile = cfg["split_data"] + "matching_evaluation/split_matching_results_combination.csv"
    if not create_file(outfile):
        exit()
    matching_approaches = ["ags21", "ngram", "sequencematcher"]
    seg_matcher = SegmentMatcher(matching_approaches, [], 3, "")
    gt = seg_matcher.get_segment_pairs_groundtruth()
    gts = [e for sub in gt for e in sub]
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresh_list = [thresholds for i in range(len(matching_approaches))]
    for combi in itertools.product(*thresh_list):
        thresh_dict = {}
        for i, elem in enumerate(combi):
            thresh_dict[matching_approaches[i]] = elem
        combine_match_approaches(thresh_dict, gts, seg_matcher, outfile)


def get_best_results():
    infile = cfg["split_data"] + "matching_evaluation/split_matching_results_combination.csv"
    df = pd.read_csv(infile)
    df["f1"] = df["f1"].replace(',','.', regex=True).astype(float)
    best = []
    for i, row in df.iterrows():
        if row["f1"] > 0.75: # 0.80:
            best.append(list(row))

    outfile = infile.replace(".csv", "_best.csv")
    if not create_file(outfile):
        print(best)
        exit()        
    dfout = pd.DataFrame(best, columns=list(df))
    dfout.to_csv(outfile, index=False)


def combine_match_approaches(thresh_dict, gts, seg_matcher, outfile):
    # thresh_dict = {"ngram": 0.4, "sequencematcher": 0.7, "transformer": 0.8}
    overall_matches = []
    preds = []
    for i, approach in enumerate(thresh_dict.keys()):
        current_thresh = thresh_dict[approach]
        seg_matcher.thresholds = [current_thresh]
        data = []
        matches = seg_matcher.get_segment_pairs_predicted(approach)[current_thresh]
        if i < len(thresh_dict.keys()) - 1:
            overall_matches.append(matches)
        else:
            for appr1, appr2 in zip(overall_matches[0], matches):
                if len(appr1) != len(appr2):
                    print("lists have different lengths - exit")
                    exit()
                # match if at least one measure assigns a match
                final_matches = [1 if 1 in [appr1[i], appr2[i]] else 0 for i in range(len(appr1))]
                preds += final_matches
            prec = metrics.precision_score(gts, preds)
            recall = metrics.recall_score(gts, preds)
            f1 = metrics.f1_score(gts, preds)
            data.append([thresh_dict, prec, recall, f1])

    dfout = pd.DataFrame(data, columns=["thresh", "precision", "recall", "f1"])
    dfout.to_csv(outfile, mode="a", index=False, header=not exists(outfile))
            

if __name__ == "__main__":
    evaluate_matching_approaches_separately(overwrite=False)
    evaluate_matching_approaches_combinations(get_best=True)

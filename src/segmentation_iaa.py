from os import path, makedirs
import numpy as np
import pandas as pd
import json
from pathlib import Path
from nltk.tokenize import word_tokenize
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import jaccard_score
from scipy.stats import spearmanr, pearsonr

from src.utils import get_file_suffix, create_file

from src.segment_matching import match_segments
from src.segmentation_evaluation import evaluate_segments
from src.segment_kp_matching import match_segments_with_kps, get_kp_coverage
from src.doccano_data import extract_doccano_segments
from src.segment_boundary_evaluation import get_category_labels, evaluate_segment_boundaries
from config import cfg

annotators = ["ann1", "ann2", "ann3"]

'''
Compute the inter-annotator-agreement of the segmentation task
'''

def extract_annotations_parts():
    '''
    Get segments only of annotated text parts.
    If only first 3 words and last three words of a sentence are annotated, the segment covers only these
    '''

    for annot in annotators:
        outpath = cfg["annotator_data"] + f"segments_parts_{annot}.csv"
        
        data = []   # index, arg_id, argument, topic, stance, topic_in_args_me, argsme_id
        with open(cfg["annotator_data"] + f"annotations_{annot}.jsonl", 'r') as json_file:
            json_list = list(json_file)

        for arg in json_list:
            segments = {}
            result = json.loads(arg)
            text = result["text"]
            labels = result["label"]
            if len(labels) == 0:
                data.append([result["index"], result["arg_id"], "", result["topic"], result["stance"], result["topic_in_argsme"], result["argsme_id"]])
            for l in labels:
                seg = l[2]
                if seg not in segments.keys():
                    segments[seg] = text[l[0]:l[1]].strip()
                else:
                    segments[seg] += " " + text[l[0]:l[1]].strip()
                
            for _, seg in segments.items():
                data.append([result["index"], result["arg_id"], seg, result["topic"], result["stance"], result["topic_in_argsme"], result["argsme_id"]])
            data.append(["", "", "", "", "", "", ""])
        
        df = pd.DataFrame(data, columns=["index", "arg_id", "argument", "topic", "stance", "topic_in_args_me", "argsme_id"])
        df_sorted = sort_df(df)
        df_sorted.to_csv(outpath, index=False)


def extract_annotations_continguous():
    '''
    Get contigous segments of annotations with segment boundaries.
    If only first 3 words and last three words of a sentence are annotated,
    the segment here still covers the whole sentence
    '''
    for annot in annotators:
        inpath = cfg["annotator_data"] + f"annotations_{annot}.jsonl"
        outpath = cfg["annotator_data"] + f"segments_contiguous_{annot}.csv"
        df_sorted = sort_df(pd.read_csv(outpath))
        df_sorted.to_csv(outpath, index=False)


def sort_df(df):
    '''Save manual segments in same order as predicted segments'''

    ref = ["arg_3_623", "arg_0_329", "arg_17_390", "arg_0_284", "arg_9_2546",
           "arg_20_166", "arg_9_2576", "arg_3_349", "arg_23_194", "arg_9_147"]
    df_sorted = pd.DataFrame(columns=list(df))
    for id in ref:
        sub = df[df["arg_id"] == id]
        df_sorted = pd.concat([df_sorted, sub, pd.DataFrame([[""]*7])])
    return df_sorted


def get_cat_labels_for_human_annotations():
    '''
    Each annotators annotations serve as grountruth once
    get matching category labels (matched, missed, spurious, ...) with remaining annotations
    '''

    for annot in annotators:
        gtfolder = Path(cfg["split_data"], f"segmentation_inter_ann/gt_{annot}")
        if not path.exists(gtfolder):
            makedirs(gtfolder)
        gt_infile = Path(cfg["annotator_data"], f"segments_contiguous_{annot}.csv")
        for pred in annotators:
            if pred == annot:
                continue
            gt_outfile = Path(gtfolder, f"/segments_{annot}_compare_with_{pred}.csv")
            pred_infile = Path(cfg["annotator_data"], f"segments_contiguous_{pred}.csv")
            pred_outfile = Path(gtfolder, f"/segments_{pred}_labels.csv")
            get_category_labels(annot, gt_outfile, pred_outfile, gt_infile, pred_infile)


def evaluate_human_annotations():
    '''Segment boundary evaluation for different groundtruths (per annotator)'''
    for annot in annotators:
        gtfolder = Path(cfg["split_data"], f"segmentation_inter_ann/gt_{annot}")
        for pred in annotators:
            if pred == annot:
                continue
            gt_file = Path(gtfolder, f"/segments_{annot}_compare_with_{pred}.csv")
            pred_file = Path(gtfolder, f"/segments_{pred}_labels.csv")
            print("--------------------")
            print(f"gt: {annot}, compare with {pred}")
            print("--------------------")
            evaluate_segment_boundaries(pred, get_labels=False, gt_file=gt_file, pred_file=pred_file)


def get_seg(labels, char):
    for l in labels:
        if char >= l[0] and char <= l[1]:
            return l[2][1:]
    return "none"

def get_seg_bin(labels, char):
    for l in labels:
        if char >= l[0] and char <= l[1]:
            return 1
    return 0


def extract_annotations_per_word():
    '''Get (binary) labels word-wise depending on whether they part are part of a segment'''
    
    annotations_bin = {}
    for annot in annotators:
        annotations_bin[annot] = {}
        inpath = cfg["annotator_data"] + f"annotations_{annot}.jsonl"
        with open(inpath, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            all_segs_bin = []   # list of 1 if word is part of seg, 0 otherwise
            result = json.loads(json_str)
            text = result["text"]
            tokens = word_tokenize(text)
            labels = result["label"]
            if len(labels) == 0:
                all_segs_bin.append([0] * len(tokens))
            else:
                for label in labels:
                    tokens_before = len(word_tokenize(text[:label[0]]))
                    tokens_after = len(word_tokenize(text[label[1]:]))
                    tokens_seg = len(tokens) - tokens_before - tokens_after
                    all_segs_bin.append([0]*tokens_before + [1]*tokens_seg + [0]*tokens_after)
                            
            combi = all_segs_bin[0]
            for segs in all_segs_bin[1:]:
                for i, elem in enumerate(segs):
                    combi[i] = combi[i] + elem
            annotations_bin[annot][text[:20]] = combi
    
    all12, all13, all23 = 0, 0, 0
    all1, all2, all3 = [], [], []
    for text in annotations_bin["ann2"].keys():
        a1 = annotations_bin["ann1"][text]
        a2 = annotations_bin["ann2"][text]
        a3 = annotations_bin["ann3"][text]
        all1 += a1
        all2 += a2
        all3 += a3
        kappa12 = cohen_kappa_score(a1, a2)
        kappa13 = cohen_kappa_score(a1, a3)
        kappa23 = cohen_kappa_score(a2, a3)
        print(f"text {text}:", kappa12, kappa13, kappa23)
        
        all12 += kappa12
        all13 += kappa13
        all23 += kappa23
    
    print(f"ma avg: a12 = {round(all12/5, 4)}, a13 = {round(all13/5, 4)}, a23 = {round(all23/5, 4)}")    
    print("micro kappa: a12 =", round(cohen_kappa_score(all1, all2), 2),
          "a13 =", round(cohen_kappa_score(all1, all3), 2),
          "a23 =", round(cohen_kappa_score(all2, all3), 2))


def get_text_coverage():
    for annot in annotators:
        inpath = cfg["annotator_data"] + f"annotations_{annot}.jsonl"
        with open(inpath, 'r') as json_file:
            json_list = list(json_file)

        cov_macro = []
        cov_micro, alltext = 0, 0
        for arg in json_list:
            seglen = 0
            result = json.loads(arg)
            for l in result["label"]:
                seglen += l[1] - l[0]
            cov_macro.append(seglen / len(result["text"]))
            cov_micro += seglen
            alltext += len(result["text"])

        print("avg. text coverage", annot)
        print("- macro:", round(sum(cov_macro) / 5, 2))  # avg. per arg, avg. over all args
        print("- micro:", round(cov_micro / alltext, 2)) # avg. over all args added together


def create_segment_subset():
    
    ids = ["arg_3_623", "arg_0_329", "arg_17_390" ,"arg_9_2546", "arg_23_194",
           "arg_0_284", "arg_20_166", "arg_9_2576", "arg_3_349", "arg_9_147"]
    for model in ["palm", "gpt4", "gemini", "paragraph", "sentence", "ajjour", "targer"]:
        suffix = get_file_suffix(model)
        filepath = cfg["split_data"] + model + f"/sample_split_train_segments_{model}{suffix.replace('_sub_iaa', '')}.csv"
        if not create_file(filepath.replace(".csv", f"_sub_iaa.csv")):
            continue
        df = pd.read_csv(filepath, encoding="iso-8859-1")
        data = []
        last_id = list(df["arg_id"])[0]
        if last_id.count("_") == 3:
            df["arg_id"] = [str(id).rsplit("_", 1)[0] for id in list(df["arg_id"])]
            last_id = last_id.rsplit("_", 1)[0]
        for i, row in df.iterrows():
            id = str(row["arg_id"])
            if id in ids:
                if id != last_id:
                    data.append([""] * len(list(row)))
                    last_id = id
                data.append(list(row))
        dfout = pd.DataFrame(data, columns=list(df))
        dfout.to_csv(filepath.replace(".csv", f"_sub_iaa.csv"), index=False)


def get_jaccard():
    
    a1 = [[1,0], [1,1,1,1,1,1,1,0,1,1,1,1], [1], [1,1,1,1,0,0,0], [1,1,1,0,1],  [1,1,1,1,1,0,0,0,0,0,0], [1,1,1,1,0,0,1], [0,1,1,0], [1,1,1,0,0], [1,0,0,1]] # ann1
    a2 = [[1,1], [1,1,1,1,0,0,0,1,0,0,0,0], [1], [1,0,0,0,1,1,1], [1,1,1,1,0],  [1,0,1,1,1,1,1,1,1,1,1], [1,1,0,1,1,1,0], [1,1,1,1], [1,1,1,1,1], [1,1,1,0]] # ann2
    a3 = [[1,0], [1,1,1,1,1,1,1,0,1,0,0,0], [1], [1,1,1,1,0,0,0], [1,1,1,1,1],  [1,1,1,1,1,0,0,0,0,0,0], [1,1,1,1,0,0,0], [1,1,0,0], [1,1,1,0,0], [1,1,0,0]] # ann3
    num = len(a1)

    all12, all13, all23 = 0, 0, 0
    for i in range(len(a1)):
        jacc12 = jaccard_score(np.array(a1[i]).reshape(-1,1), np.array(a2[i]).reshape(-1,1))
        jacc13 = jaccard_score(np.array(a1[i]).reshape(-1,1), np.array(a3[i]).reshape(-1,1))
        jacc23 = jaccard_score(np.array(a2[i]).reshape(-1,1), np.array(a3[i]).reshape(-1,1))
        print(round(jacc12, 4), round(jacc13, 4), round(jacc23, 4))
        all12 += jacc12
        all13 += jacc13
        all23 += jacc23
    
    print(f"macro average: a12 = {round(all12/num, 4)}, a13 = {round(all13/num, 4)}, a23 = {round(all23/num, 4)}")

    jacc12 = jaccard_score([x for xs in a1 for x in xs], [x for xs in a2 for x in xs])
    jacc13 = jaccard_score([x for xs in a1 for x in xs], [x for xs in a3 for x in xs])
    jacc23 = jaccard_score([x for xs in a2 for x in xs], [x for xs in a3 for x in xs])
    print(f"micro avg over all:", round(jacc12, 4), round(jacc13, 4), round(jacc23, 4))
    

def get_model_rank_correlation():
    ann1_p = [0.46, 0.15, 0.28, 0.53, 0.24, 0.22, 0.17]
    ann1_r = [0.58, 0.21, 0.68, 1.00, 1.00, 0.92, 1.00]
    ann2_p = [0.5, 0.21, 0.31, 0.48, 0.23, 0.23, 0.19]
    ann2_r = [0.63, 0.28, 0.67, 1.0, 1.0, 0.89, 1.0]
    ann3_p = [0.4, 0.19, 0.29, 0.48, 0.19, 0.17, 0.16]
    ann3_r = [0.61, 0.28, 0.62, 1.0, 1.0, 0.78, 1.0]
    pears_p = pearsonr(ann1_p, ann2_p)
    spear_p = spearmanr(ann1_p, ann2_p)
    pears_r = pearsonr(ann1_r, ann2_r)
    spear_r = spearmanr(ann1_r, ann2_r)
    print("ann1, ann2")
    print(pears_p, spear_p)
    print(pears_r, spear_r)

    pears_p = pearsonr(ann1_p, ann3_p)
    spear_p = spearmanr(ann1_p, ann3_p)
    pears_r = pearsonr(ann1_r, ann3_r)
    spear_r = spearmanr(ann1_r, ann3_r)
    print("ann1, ann3")
    print(pears_p, spear_p)
    print(pears_r, spear_r)

    pears_p = pearsonr(ann2_p, ann3_p)
    spear_p = spearmanr(ann2_p, ann3_p)
    pears_r = pearsonr(ann2_r, ann3_r)
    spear_r = spearmanr(ann2_r, ann3_r)
    print("ann2, ann3")
    print(pears_p, spear_p)
    print(pears_r, spear_r)


def clean_df(df, remove):
    df = df.drop(remove, axis=1)
    df = df.dropna(how="all")
    for col in list(df):
        df[col] = df[col].str.replace("\g{", "").str.replace("}", "").str.replace(",", ".").astype(float)
    return df


def get_variance_per_value():
    dfz = pd.read_csv("data/latex_tables/results_table_gtann3.csv", encoding="iso-8859-1")
    dfh = pd.read_csv("data/latex_tables/results_table_gtann2.csv", encoding="iso-8859-1")
    dfd = pd.read_csv("data/latex_tables/results_table_gtann1.csv", encoding="iso-8859-1")
    remove = [col for col in list(dfz) if "_num" in col] + ["category", "arrow", "pict"]
    dfz = clean_df(dfz, remove)
    dfh = clean_df(dfh, remove)
    dfd = clean_df(dfd, remove)

    
    variances, sdeviations, averages = [], [], []
    for row in list(dfz.index):
        rowz = dfz.loc[row, :].values.tolist()
        rowh = dfh.loc[row, :].values.tolist()
        rowd = dfd.loc[row, :].values.tolist()
        rowvar, rowstd, rowavg = [], [], []
        for col in range(len(list(dfz))):
            variance = np.var([rowz[col], rowh[col], rowd[col]])
            sdev = np.std([rowz[col], rowh[col], rowd[col]])
            rowvar.append(variance)
            rowstd.append(sdev)
            rowavg.append(round((rowz[col] + rowh[col] + rowd[col]) / 3, 2))
        variances.append(rowvar)
        sdeviations.append(rowstd)
        averages.append(rowavg)
    dfvar = pd.DataFrame(variances, columns=list(dfz))
    dfstd = pd.DataFrame(sdeviations, columns=list(dfz))
    dfavg = pd.DataFrame(averages, columns=list(dfz))
    print("variance:\n", dfvar)
    print("-----")
    print("standard deviation:\n", dfstd)
    print("-----")
    print("average:\n", dfavg)



def calculate_inter_ann_agreement():
    ''' Main method for evaluating human annotations. '''
    
    '''Data preparation'''
    # extract_annotations_continguous()
    # extract_annotations_parts()
    # get_cat_labels_for_human_annotations()

    '''Agreement'''
    # print("eval 1")
    # evaluate_human_annotations()
    # print("char annot")
    # extract_annotations_per_char()
    # print("word annot")
    # extract_annotations_per_word()
    
    '''Text and KP coverage'''
    # print("coverage")
    # get_text_coverage()

    resfile = "covered_kps_maxsim_t0.9_ags21.csv"
    matchfold = cfg["annotator_data"] + "matched_kps/"
    # for annot in annotators:
    #     sf = cfg["annotator_data"] + f"segments_parts_{annot}.csv"
    #     match_segments_with_kps(annot, resfile, matchfold, segfile=sf)
    # get_kp_coverage(matchfold+resfile, gt="ann3")

    '''Eval with segmenter'''
    # create_segment_subset()
    # eval_data_with_changed_groundtruth("ann2")
    # print("var")
    # get_variance_per_value()
    # print("corr")
    # get_model_rank_correlation()

    '''Eval of manual matching'''
    # get_jaccard()

def eval_data_with_changed_groundtruth(annot):
    '''
    Evaluate model performance with groundtruth segments created by ann2/ann1
    '''

    for model in ["palm", "gpt4", "paragraph", "sentence", "ajjour", "targer"]:
        # match_segments(model, outfold=cfg["split_data"]+"segmentation_inter_ann/")
        evaluate_segments(model, manual_matching=False,metafold=cfg["split_data"]+"segmentation_inter_ann/")
        print("\n########################################################################")
        
    exit()


if __name__ == "__main__":
    calculate_inter_ann_agreement()

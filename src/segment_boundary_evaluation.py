import pandas as pd
import numpy as np
from pathlib import Path

from config import cfg
from src.segmentation_evaluation import get_filtered, get_micro_average, get_macro_average, get_granularity_score, get_plagdet
from src.utils import create_file, split_df_at_empty_lines


def count_category_labels(df, mod, results, model):
    '''Count categories for a given dataframe with either manual or predicted segments'''
    df = df.dropna(how="all")
    if cfg["filter"]:
        df, removed = get_filtered(df, model)
        removed_manuals = list(set(list(removed[removed["manual"].notna()]["manual"])))
        results["missed"] += len([man for man in removed_manuals if man not in list(df["manual"])])

    labels = list(df["label"])
    
    if mod == "gt":
        results["correct_m"] += labels.count("matched")
        results["incomplete_m"] += labels.count("incomplete")
        results["impure_m"] += labels.count("impure")
        results["inc_imp_m"] += labels.count("incomplete_impure")
        results["missed"] += labels.count("missed")
    else:
        results["correct_p"] += labels.count("matched")
        results["incomplete_p"] += labels.count("incomplete")
        results["impure_p"] += labels.count("impure")
        results["inc_imp_p"] += labels.count("incomplete_impure")
        results["spurious"] += labels.count("spurious")

    return results


def get_category_counts_per_arg(df_gt, df_pred, pos_rel_dict, model):
    '''Get number of categories for each argument separately'''
    results_per_arg = {}
    
    arg_dfs_gt = split_df_at_empty_lines(df_gt, col="arg_id")
    arg_dfs_pred = split_df_at_empty_lines(df_pred, col="arg_id")

    for i, arg_df in enumerate(arg_dfs_gt):
        empty_df = {"correct_m": 0, "incomplete_m": 0, "impure_m": 0, "inc_imp_m": 0, "missed": 0,
                    "correct_p": 0, "incomplete_p": 0, "impure_p": 0, "inc_imp_p": 0,"spurious": 0,
                    "positives": 0, "relevants": 0, "filtered": 0}
        arg_results = count_category_labels(arg_df, "gt", empty_df, model)
        arg_results = count_category_labels(arg_dfs_pred[i], "pred", arg_results, model)
        arg_results["positives"] = pos_rel_dict[i]["pred"]
        arg_results["relevants"] = pos_rel_dict[i]["man"]
        results_per_arg[f"arg_{i}"] = arg_results

    return results_per_arg


def get_pos_rel(df_gt, df_pred):
    '''Create a dict with number of relevant and positive segments per argument/ for all arguments'''
    prd = {"all": {}}
    arg_dfs = split_df_at_empty_lines(df_gt)
    for i, arg_df in enumerate(arg_dfs):
        arg_df = arg_df.dropna(how="any")
        prd[i] = {}
        prd[i]["man"] = arg_df["segment"].nunique()
    df = df_gt.dropna(how="all")
    prd["all"]["man"] = df["segment"].nunique()
    
    vals = [str(seg) == "nan" for seg in list(df_pred["arg_id"])]
    arg_dfs = np.split(df_pred, *np.where(vals))
    for i, arg_df in enumerate(arg_dfs):
        prd[i]["pred"] = arg_df["segment"].nunique()
    df = df_pred.dropna(how="all")
    prd["all"]["pred"] = df["segment"].nunique()
    return prd


def evaluate_segment_boundaries(model, get_labels=False, gt_file="", pred_file=""):
    '''
    Main method for evaluating segments mapped to original text,
    i.e., compare boundaries of predicted segments with those of the manual (ground truth) segments
    param model: segmentation approach to be evaluated
    param get_labels: False if labels can be loaded from file
    '''

    filt = "_filt" if cfg["filter"] else ""
    if gt_file == "" and pred_file == "":
        gt_file = Path(cfg["split_data"], f"manual/sample_split_train_boundaries_segments_labels_{model}{filt}.csv")
        pred_file = Path(cfg["split_data"], f"matched_to_text/{model}_extractions_labels{filt}.csv")

    if get_labels:
        df_gt, df_pred = get_category_labels(model, gt_file, pred_file)
    else:
        df_gt = pd.read_csv(gt_file)
        df_pred = pd.read_csv(pred_file)
    
    results = {"correct_m": 0, "incomplete_m": 0, "impure_m": 0, "inc_imp_m": 0, "missed": 0,
               "correct_p": 0, "incomplete_p": 0, "impure_p": 0, "inc_imp_p": 0,"spurious": 0, 
               "positives": 0, "relevants": 0, "filtered": 0}
    
    pos_rel_dict = get_pos_rel(df_gt, df_pred)

    results = count_category_labels(df_gt, "gt", results, model)
    results = count_category_labels(df_pred, "pred", results, model)
    results["positives"] = pos_rel_dict["all"]["pred"]
    results["relevants"] = pos_rel_dict["all"]["man"]
    print(results, "\n")

    mi_f1, mi_f1_rel, miavg = get_micro_average(results)    
    print(f"--- micro avg ---\n"\
        f"{miavg}\n" \
        f"micro F1 strict: {mi_f1} \n" \
        f"micro F1 relaxed: {mi_f1_rel} \n")
        
    results_per_arg = get_category_counts_per_arg(df_gt, df_pred, pos_rel_dict, model)
    num_args = df_gt.last_valid_index()+1 - len(df_gt.dropna(how="all").index) + 1

    ma_f1, ma_f1_rel, maavg = get_macro_average(results_per_arg, num_args)
    print(f"\n--- macro avg ---\n" \
        f"{maavg}\n" \
        f"macro F1 strict: {ma_f1} \n" \
        f"macro F1 relaxed: {ma_f1_rel} \n")

    gran_mi, gran_ma = get_granularity_score(results, results_per_arg)
    plagdet = get_plagdet(mi_f1, gran_mi)
    print(f"\n--- plagdet --- {plagdet}")
    

def get_category_labels(model, gt_outfile, pred_outfile, gt_infile="", pred_infile=""):
    filt = "_filt" if cfg["filter"] else ""
    if gt_infile == "" and pred_infile == "":
        gt_infile = gt_outfile.replace(f"_labels_{model}{filt}", "")
        pred_infile = pred_outfile.replace(f"labels{filt}", "manual_revision_corr")
    
    df_gt = pd.read_csv(gt_infile)
    df_preds = pd.read_csv(pred_infile)
    
    if cfg["filter"]:
        df_preds, _ = get_filtered(df_preds, model, "argument")

    vals_gt = [str(val) == "nan" for val in list(df_gt.arg_id)]
    vals_p = [str(val) == "nan" for val in list(df_preds.arg_id)]

    labels_man, labels_pred = [], []
    for gt, preds in zip(np.split(df_gt, *np.where(vals_gt)), np.split(df_preds, *np.where(vals_p))):
        gt_bounds = [(s, e) for s, e in zip(list(gt["start"]), list(gt["end"]))]
        preds_bounds = [(s, e) for s, e in zip(list(preds["start"]), list(preds["end"]))]
        
        # for each pred_seg: collect index of all different gt_seg of which at least one boundary is contained
        gts_in_preds = collect_gts_in_preds(gt_bounds, preds_bounds)
        preds_in_gts = collect_preds_in_gts(gt_bounds, preds_bounds)
        labels_man += get_labels_man(gt_bounds, preds_bounds, gts_in_preds, preds_in_gts)
        labels_pred += get_labels_pred(gt_bounds, preds_bounds, gts_in_preds, preds_in_gts)

    df_gt["label"] = labels_man
    if create_file:
        df_gt.to_csv(gt_outfile, index=False)

    if str(list(df_preds["arg_id"])[-1]) == "nan":          # remove last row if entries are nan
        df_preds = df_preds.iloc[0:len(df_preds.index)-1]
    labels_pred = labels_pred[:-1] if labels_pred[-1] == "" else labels_pred
    df_preds["label"] = labels_pred
    if create_file:
        df_preds.to_csv(pred_outfile, index=False)
    return df_gt, df_preds


def get_labels_man(gtbounds, pbounds, gts_in_preds, preds_in_gts):
    '''
    Gets two lists of tuples (start, end) for gt_segs and pred_segs
    Returns a list of labels for ground truth segments
    '''
    labels = []
    for j, gb in enumerate(gtbounds):
        if str(gb[0]) == "nan" and str(gb[1]) == "nan":
            labels.append("")
            continue
        gtstart_between = [between(gb[0], pb[0], pb[1]) for pb in pbounds]
        gtend_between = [between(gb[1], pb[0], pb[1]) for pb in pbounds]
        gt_contained = [contained(gb, pb) for pb in pbounds]
        label = get_single_category(gt_contained, gtstart_between, gtend_between, 
                                            [l for l in gts_in_preds.values() if j in l], [preds_in_gts[j]])
        labels.append(label)
    return labels


def get_labels_pred(gtbounds, pbounds, gts_in_preds, preds_in_gts):
    '''
    Gets two lists of tuples (start, end) for gt_segs and pred_segs
    Returns a list of labels for predicted segments
    '''
    labels = []
    for j, pb in enumerate(pbounds):
        if str(pb[0]) == "nan" and str(pb[1]) == "nan":
            labels.append("")
            continue
        gtstart_between = [between(gb[0], pb[0], pb[1]) for gb in gtbounds]
        gtend_between = [between(gb[1], pb[0], pb[1]) for gb in gtbounds]
        gt_contained = [contained(gb, pb) for gb in gtbounds]
        label = get_single_category(gt_contained, gtstart_between, gtend_between,
                                      [gts_in_preds[j]], [l for l in preds_in_gts.values() if j in l])
        labels.append("spurious") if label=="missed" else labels.append(label)
    return labels


def get_single_category(gt_contained, gtstart_between, gtend_between, multiple_gts, multiple_preds):
    '''
    Decide whether a segment is matched/incomplete/impure/missed/spurious
    param gt_contained: list of booleans indicating whether a gt seg is completely contained in a pred seg (1 gt for all pred, or all gt for 1 pred)
    param gtstart_between: list of booleans indicating whether gtstart is contained in pred
    param gtend_between: list of booleans indicating whether gt end is contained in pred
    param multiple_gts: list of values-lists from gts_in_preds (for current pred, or all preds containing current gt)
    '''
    
    # pred_seg (partly) covers multiple gt_segs
    if len(multiple_gts)==1 and len(multiple_gts[0])>1: 
        if len(multiple_preds)>0 and any([len(cand) > 1 for cand in multiple_preds]):
            return "incomplete_impure"
        return "impure"
    elif len(multiple_gts) > 1 and any([len(cand) > 1 for cand in multiple_gts]):
        return "incomplete_impure"
    elif len(multiple_gts) > 1:
        return "incomplete"
    
    # gt_seg (partly) covered by multiple pred_segs
    if len(multiple_preds)==1 and len(multiple_preds[0])>1:
        return "incomplete"
    elif len(multiple_preds)>1: # pred contained in multiple gts -> impure + inc.
        print("**")
    
    if any(gt_contained):
        return "matched"
    elif not any(gtstart_between) and not any(gtend_between):
        return "missed"
    # one boundary is not contained in any pred_seg
    elif not all([any(gtstart_between), any(gtend_between)]):
        return "incomplete"
    else: 
        return "impure"
   

def collect_gts_in_preds(gtbounds, pbounds): 
    '''
    For all pred_seg of an argument:
    list index of all gt_segs that contain at least one boundary of the pred_seg
    '''
    gts_in_preds = {}
    for i, pb in enumerate(pbounds):
        contained_in_gt = []
        for j, gb in enumerate(gtbounds):
            # start and/or end of gt_seg is contained in pred_seg
            if between(gb[0], pb[0], pb[1]) or between(gb[1], pb[0], pb[1]):
                contained_in_gt.append(j)
        gts_in_preds[i] = contained_in_gt
    return gts_in_preds


def collect_preds_in_gts(gtbounds, pbounds):
    '''
    For all gt_seg of an argument:
    list index of all pred_segs that contain at least one boundary of the gt_seg
    '''

    preds_in_gts = {}
    for i, gb in enumerate(gtbounds):
        contained_in_pred = []
        for j, pb in enumerate(pbounds):
            # start and/or end of pred_seg is contained in gt_seg
            if between(pb[0], gb[0], gb[1]) or between(pb[1], gb[0], gb[1]):
                contained_in_pred.append(j)
        preds_in_gts[i] = contained_in_pred
    return preds_in_gts


def between(check, lower, upper):
    '''Verify if index 'check' is between segment boundaries 'lower' and 'upper' (ints)'''
    return check >= lower and check <= upper


def contained(seg1, seg2):
    '''Verify if seg1 is completely contained in seg2 (tuples)'''
    return seg1[0] >= seg2[0] and seg1[0] <= seg2[1] and \
           seg1[1] >= seg2[0] and seg1[1] <= seg2[1]


def get_covered_text_percentage(model):
    len_original, len_pred = 0, 0
    ref_df = pd.read_csv(cfg["argument_texts"], encoding="iso-8859-1")
    for arg in list(ref_df["segment"]):
        len_original += len(arg)
    filepath = Path(cfg["split_data"], f"matched_to_text/{model}_extractions_manual_revision_corr.csv")
    df = pd.read_csv(filepath, encoding="iso-8859-1")
    last_end = -1
    for i, row in df.iterrows():
        seg = str(row["segment"])
        if seg == "nan":
            last_end = -1
            continue
        start = row["start"]
        end = row["end"]
        if start == -1:
            len_original += end + 1
        elif str(last_end) == -1 or (start >= last_end and end >= last_end):
            len_pred += len(seg)
        elif end > last_end:
            len_pred += (end - last_end)
            print(f"overlapping segments ({i})", start, end, "-", last_end, "add", end-last_end)
        else:
            print(f"overlapping segments ({i})", start, end, "-", last_end, "add 0")
        last_end = row["end"]
    print("coverage =", round(len_pred / len_original, 4))



if __name__ == "__main__":
    model = cfg["segmentation_approach"]
    evaluate_segment_boundaries(model, get_labels=False)


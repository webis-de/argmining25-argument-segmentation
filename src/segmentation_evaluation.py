import pandas as pd

from config import cfg
from src.segment_matching import match_segments
from src.argument_classification.ukp_code_inference import add_arg_class_to_file
from src.utils import get_file_suffix, parameter_check, split_df_at_empty_lines


def update_counts(df, model, col):
    pred_col = model if col=="key_statement" else col
    updated = []
    manuals = list(df[col])
    preds = list(df[pred_col])
    for i, row in df.iterrows():
        if str(row[col]) != "nan" and str(row[pred_col]) != "nan":
            row["matches_for_manual"] = manuals.count(row[col])
            row["matches_for_pred"] = preds.count(row[pred_col])
        updated.append(row)
    return pd.DataFrame(updated, columns=list(df))


def get_filtered(df, model, col="key_statement"):
    kept = []
    removed = []
    col = cfg["filter_column"]
    if col not in list(df):
        print(f"ATTENTION: column {col} not found - no filtering applied!")
        return df, removed
    
    for i, row in df.iterrows():
        keep = str(row["arg_class"]) != "NoArgument"
        if keep:
            kept.append(list(row))
        else:
            removed.append(list(row))

    kept_df = update_counts(pd.DataFrame(kept, columns=list(df)), model, col)
    return kept_df, pd.DataFrame(removed, columns=list(df))


'''
Based on the similarity scores and matching of the chosen matching approaches
(ngram, sequencematcher, transformer): count occurrence of every category
- correct if manual and predicted segment are matched exactly one time, or if sim.>0.9
- spurious if predicted segment is never matched
- missed if manual segment is never matched
- incomplete if manual seg. is matched multiple times (with different pred. seg.s)
- impure if pred. seg is matched multiple times (with different manual seg.s)
- incomplete & impure if pred. and manual seg. both are matched multiple times
'''
def get_matching_category(df, model):
    results = {"correct_m": 0, "incomplete_m": 0, "impure_m": 0, "inc_imp_m": 0,
               "correct_p": 0, "incomplete_p": 0, "impure_p": 0, "inc_imp_p": 0,
               "spurious": 0, "missed": 0, "positives": 0, "relevants": 0, "filtered": 0,
               "pred_duplicates": 0, "man_duplicates": 0,
               "distorted_matched": 0, "distorted_other": 0, "correct_removed": 0}
    df = df.dropna(how="all")
    results["relevants"] = df["key_statement"].nunique()   # denominator for recall

    if cfg["filter"]:
        df, removed = get_filtered(df, model)
        removed_manuals = list(set(list(removed[removed["key_statement"].notna()]["key_statement"])))
        results["missed"] += len([man for man in removed_manuals if man not in list(df["key_statement"])])
        results["correct_removed"] += len(removed[removed["key_statement"].isna()].index)
        results["filtered"] = len(list(set(list(removed["segment"]))))
        
    results["positives"] = df["segment"].nunique()      # denominator for precision
    labels_m, labels_p, seen_pred, seen_man = [], [], [], []
    last = -1
    for i, row in df.iterrows():
        man = str(row["key_statement"])
        pred = str(row["segment"])
        if i-1 != last:                         # add empty label for empty separator row
            labels_m.append("")
            labels_p.append("")
            last += 1
        num_m_man = row["matches_for_manual"]
        num_m_pred = row["matches_for_pred"]
        if (man != "nan" and pred != "nan" and num_m_man == 1 and num_m_pred == 1) \
            or row["sim_transformer"] > 0.9 or row["sim_seq_matcher"] > 0.9:
            results["correct_m"] += 1 if man not in seen_man else 0
            results["correct_p"] += 1 if pred not in seen_pred else 0
            labels_m.append("correct")
            labels_p.append("correct")
        elif man != "nan" and pred == "nan":
            results["missed"] += 1              # manual seg. has no matched predicted seg.
            labels_m.append("missed")
            labels_p.append("")
        elif man == "nan" and pred != "nan":
            results["spurious"] += 1            # predicted seg. has no matched manual seg.
            labels_m.append("")
            labels_p.append("spurious")
        else:                                   # incorrect matches (i.e., num_m_pred and/or num_m_man > 1)
            if num_m_man > 1 and num_m_pred > 1:
                results["inc_imp_m"] += 1 if man not in seen_man else 0
                results["inc_imp_p"] += 1 if pred not in seen_pred else 0
                labels_m.append("inc_imp")
                labels_p.append("inc_imp")
            elif num_m_man > 1:
                results["incomplete_m"] += 1 if man not in seen_man else 0
                results["incomplete_p"] += 1 if pred not in seen_pred else 0
                labels_m.append("incomplete")
                labels_p.append("incomplete")
            else:
                results["impure_m"] += 1 if man not in seen_man else 0
                results["impure_p"] += 1 if pred not in seen_pred else 0
                labels_m.append("impure")
                labels_p.append("impure")

        last += 1
        seen_pred.append(pred)
        seen_man.append(man)

    return results, labels_m, labels_p


def get_matching_category_by_comments(df, model):
    '''
    Based on the manual (groundtruth) matching of PaLM and manual segments: 
    count occurrence of every category (correct, spurious, missed, incomplete, impure)
    '''

    results = {"correct_m": 0, "incomplete_m": 0, "impure_m": 0, "inc_imp_m": 0,
               "correct_p": 0, "incomplete_p": 0, "impure_p": 0, "inc_imp_p": 0,
               "spurious": 0, "missed": 0, "positives": 0, "relevants": 0, "filtered": 0,
               "pred_duplicates": 0, "man_duplicates": 0,
               "distorted_matched": 0, "distorted_other": 0, "correct_removed": 0}
    labels_m, labels_p = [""]*len(df.index), [""]*len(df.index)
    df = df.dropna(how="all")
    results["relevants"] += len(df[df["key_statement"].notna()])     # denominator for recall

    if cfg["filter"]:
        removed = df[df[cfg["filter_column"]] == "NoArgument"]
        df = df[~df.isin(removed)]
        results["missed"] += len(removed[removed["key_statement"].notna()].index)
        results["correct_removed"] += len(removed[removed["key_statement"].isna()].index)
        results["filtered"] = len(removed.index) # results["correct_removed"] + results["missed"]
    
    results["positives"] += len(df[df["segment"].notna()].index)  # denominator for precision
    labels_m, labels_p, seen_pred, seen_man = [], [], [], []
    for i, row in df.iterrows():
        man = str(row["key_statement"])
        pred = str(row["segment"])
        if man == "nan" and pred == "nan":                    # add empty label for empty separator row
            labels_m.append("")
            labels_p.append("")
            continue
        comment = str(row["comments"])
        if comment == "distorted" and man == "nan":
            results["distorted_other"] += 1
        elif comment == "distorted" and man != "nan":
            results["distorted_matched"] += 1

        if man != "nan" and pred != "nan" and (comment == "nan" or comment == "distorted"):
            results["correct_m"] += 1 if man not in seen_man else 0
            results["correct_p"] += 1 if pred not in seen_pred else 0
            labels_m.append("correct")
            labels_p.append("correct")
        elif man != "nan" and pred == "nan":
            results["missed"] += 1              # manual seg. has no matched predicted seg.
            labels_m.append("missed")
            labels_p.append("")
        elif man == "nan" and pred != "nan" and "part" not in comment:
            results["spurious"] += 1            # predicted seg. has no matched manual seg.
            labels_m.append("")
            labels_p.append("spurious")
        else:                                   # incorrect matches (i.e., num_m_pred and/or num_m_man > 1)
            if "part" in comment and "missing split" in comment:
                results["inc_imp_m"] += 1 if man not in seen_man else 0
                results["inc_imp_p"] += 1 if pred not in seen_pred else 0
                labels_m.append("inc_imp")
                labels_p.append("inc_imp")
            elif "part" in comment:
                results["incomplete_m"] += 1 if man not in seen_man else 0
                results["incomplete_p"] += 1 if pred not in seen_pred else 0
                labels_m.append("incomplete")
                labels_p.append("incomplete")
            elif "missing split" in comment:
                results["impure_m"] += 1 if man not in seen_man else 0
                results["impure_p"] += 1 if pred not in seen_pred else 0
                labels_m.append("impure")
                labels_p.append("impure")
            else:
                print(f"*{man}* \n*{pred}* \n*{comment}*\n")

        seen_pred.append(pred)
        seen_man.append(man)
    return results, labels_m, labels_p


def get_categories(df, model, manual_matching):
    if manual_matching:
        results, labels_m, labels_p = get_matching_category_by_comments(df, model)
    else:
        results, labels_m, labels_p = get_matching_category(df, model)
    return results, labels_m, labels_p


def count_match_categories(df, model, manual_matching):
    '''
    Get results for all arguments together and for each argument separately
    '''

    results_all, labels_m, labels_p = get_categories(df, model, manual_matching)
    results_per_arg = {}
    arg_dfs = split_df_at_empty_lines(df, "topic")
    for i, arg_df in enumerate(arg_dfs):
        arg_results, _, _ = get_categories(arg_df, model, manual_matching)
        results_per_arg[f"arg_{i}"] = arg_results
    return results_all, results_per_arg, labels_m, labels_p
    

def get_f1(prec, rec):
    denominator = prec + rec
    return 0.0 if denominator == 0.0 else round((2 * prec * rec / denominator), 2)


def get_micro_average(results):
    '''Average over all argument texts together'''

    avg_dict = {"correct_m": 0, "incomplete_m": 0, "impure_m": 0, "inc_imp_m": 0, "missed": 0,
                "correct_p": 0, "incomplete_p": 0, "impure_p": 0, "inc_imp_p": 0, "spurious": 0}
    
    positives = results["positives"]
    relevants = results["relevants"]
    for key, val in results.items():
        if relevants > 0 and (key.endswith("_m") or key == "missed"):
            avg_dict[key] = round(val / relevants, 2)
        elif positives > 0 and (key.endswith("_p") or key == "spurious"):
            avg_dict[key] = round(val / positives, 2)

    mi_f1_strict = get_f1(avg_dict["correct_p"], avg_dict["correct_m"])
    mi_f1_relaxed = get_f1(1-avg_dict["spurious"], 1-avg_dict["missed"])
    return mi_f1_strict, mi_f1_relaxed, avg_dict


def get_macro_average(results_per_arg, num_args):
    '''Average over averages per argument text'''

    print("num args:", num_args)
    avg_dict = {"correct_m": 0, "incomplete_m": 0, "impure_m": 0, "inc_imp_m": 0, "missed": 0,
                "correct_p": 0, "incomplete_p": 0, "impure_p": 0, "inc_imp_p": 0, "spurious": 0}

    for arg, adict in results_per_arg.items():
        _, _, sub_avg = get_micro_average(adict)
        avg_dict = {key: avg_dict.get(key, 0) + sub_avg.get(key, 0) 
            for key in avg_dict.keys() | sub_avg.keys()}

    for key, val in avg_dict.items():
        avg_dict[key] = round((val / num_args), 2)

    ma_f1_strict = get_f1(avg_dict["correct_p"], avg_dict["correct_m"])
    ma_f1_relaxed = get_f1(1-avg_dict["spurious"], 1-avg_dict["missed"])
    return ma_f1_strict, ma_f1_relaxed, avg_dict


def add_matching_label(model, manual_matching=False):
    '''
    For a file with matched segments: add a label-columns with matching category
    (correct, missed, spurious, incomplete, impure, inc+imp)
    '''

    df, matchfile = get_matching_dataframe(model, manual_matching)
    if manual_matching:
        _, labels_m, labels_p = get_matching_category(df, model, sub=False)
    else:
        _, labels_m, labels_p = get_matching_category(df, model)
    df["matching_label_manual"] = labels_m + ["nan"] * (len(df.index) - len(labels_m))
    df["matching_label_pred"] = labels_p + ["nan"] * (len(df.index) - len(labels_p))
    df.to_csv(matchfile, index=False)


def get_matching_dataframe(model, manual_matching, metafold=cfg["split_data"]):
    dataset = cfg["dataset"] if cfg["dataset"] == "" else "_"+cfg["dataset"]
    suffix = get_file_suffix(model)
    matchfile = metafold + f"matched_segments{dataset}/matched_segments_{model}{suffix}.csv"
    if manual_matching:
        if model not in ["palm", "gpt4", "sentence"]:
            print(f"No manual matching available for {model}. Exit.")
            exit()
        matchfile = cfg["manual_matching_file"]
    df = pd.read_csv(matchfile, encoding="ISO-8859-1")
    if not manual_matching:
        df["sim_transformer"] = df["sim_transformer"].replace(',','.', regex=True).astype(float)
        df["sim_seq_matcher"] = df["sim_seq_matcher"].replace(',','.', regex=True).astype(float)
    if cfg["filter"] and cfg["filter_column"] != "arg_class":
        df[cfg["filter_column"]] = df[cfg["filter_column"]].replace(',','.', regex=True).astype(float)
    
    return df, matchfile


'''
Main method for evaluating predicted segments with different metrics
'''
def evaluate_segments(model, manual_matching=False, metafold=cfg["split_data"]):

    if cfg["check_params"]:
        parameter_check(["filter", "filter_column", "inter_annot"])
    print(f"evaluate segments with {model} approach, filtering={cfg['filter']}, " \
          f"filter_column={cfg['filter_column']}, inter_annot={['inter_annot']}, " \
          f"manual matching={manual_matching}")
    df, _ = get_matching_dataframe(model, manual_matching, metafold)
    results, results_per_arg, _, _ = count_match_categories(df, model, manual_matching)
    print(results, "\n")
    
    num_args = df.last_valid_index()+1 - len(df.dropna(how="all").index) + 1
    mi_f1, mi_f1_rel, miavg = get_micro_average(results)    
    print(f"--- micro avg ---\n"\
          f"{miavg}\n" \
          f"micro F1 strict: {mi_f1} \n" \
          f"micro F1 relaxed: {mi_f1_rel} \n")

    ma_f1, ma_f1_rel, maavg = get_macro_average(results_per_arg, num_args)
    print(f"\n--- macro avg ---\n" \
          f"{maavg}\n" \
          f"macro F1 strict: {ma_f1} \n" \
          f"macro F1 relaxed: {ma_f1_rel} \n")



if __name__ == "__main__":
    model = cfg["segmentation_approach"]
    # match segments to ground truth key statements
    # match_segments(model)
    # add_matching_label(model, manual_matching=False)

    '''Add classes (arg_pro, arg_con, no_arg) to file of segments and matched segments'''
    # add_arg_class_to_file(cfg["split_data"] + f"{model}/segments_{model}{get_file_suffix(model)}.csv")
    # add_arg_class_to_file(cfg["split_data"] + f"matched_segments/matched_segments_{model}{get_file_suffix(model)}.csv")

    # evaluate_segments(model)
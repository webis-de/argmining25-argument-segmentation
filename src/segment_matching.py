import pandas as pd
import nltk
from os.path import exists
from config import cfg
from difflib import SequenceMatcher
from sklearn import metrics
import os
from collections import Counter
from sentence_transformers import SentenceTransformer

from src.utils import get_embedding_overlap, create_file, parameter_check, get_file_suffix, split_df_at_empty_lines
import src.llms


def match_segments(model, outfold=cfg["split_data"]):
    '''
    Main method for matching some generated with the manual segments of the test set
    - param model: the segmentation model
    - param pred_path: path to the created segments
    '''
    dataset = cfg["dataset"] if cfg["dataset"] == "" else "_"+cfg["dataset"]
    suffix = get_file_suffix(model)
    pred_path = cfg["split_data"] + f"{model}/"
    outfile = outfold + f"matched_segments{dataset}/matched_segments_{model}{suffix}.csv"
    os.makedirs(os.path.dirname(f"{outfold}/matched_segments/"), exist_ok=True)
    
    if cfg["check_params"]:
        parameter_check(["matching_approaches", "thresh_ngram", "thresh_seq", "thresh_transf", 
                        "n_ngram", "key_statements_groundtruth", "inter_annot"],
                        {"pred_path": pred_path, "outfile": outfile})
    if not create_file(outfile, append=True):
        return
    matcher = SegmentMatcher(cfg["matching_approaches"], [cfg["thresh_ngram"],  
                             cfg["thresh_seq"], cfg["thresh_transf"]], cfg["n_ngram"], "")
    cols = ["arg_id", "key_statement", "segment", f"sim_{cfg['n_ngram']}gram", "sim_seq_matcher", 
            "sim_transformer", "matches_for_manual", "matches_for_pred", "topic"]
    
    df_preds = get_predicted_segments(pred_path, model)
    if len(df_preds[-1].index) == 1 and pd.isna(df_preds[-1].iloc[0,0]):
        df_preds = df_preds[:-1]
    if len(df_preds) != len(matcher.arg_dfs):
        print(f"Error different lengths: predictions {len(df_preds)}, manual: {len(matcher.arg_dfs)}")
        exit()

    # iterate arguments
    for i, df in enumerate(matcher.arg_dfs):
        topic = list(df["topic"])[-1]
        arg_id = list(df["arg_id"])[-1].rsplit("_", 1)[0] if list(df["arg_id"])[-1].count("_") > 2 else list(df["arg_id"])[-1]
        manual, preds = prepare_segments(df, df_preds[i])
        matched_pairs = []
        matched_predictions = []
        # iterate manual segments and check overlap with predicted segments
        for m, man in enumerate(manual):
            mantok = nltk.word_tokenize(man)
            num_matches_for_manual = 0      # count how many different predicted segments are matched to a manual segment (impure)
            for p, pred in enumerate(preds):
                ptok = nltk.word_tokenize(pred)
                value_ngram = matcher.get_overlap("ngram", man, pred, mantok, ptok)
                value_seq = matcher.get_overlap("sequencematcher", man, pred, mantok, ptok)
                value_transf = matcher.get_overlap("transformer", man, pred, mantok, ptok)
                # at least one similarity measure exceeds threshold
                if value_ngram > cfg["thresh_ngram"] or value_seq > cfg["thresh_seq"] or \
                   value_transf > cfg["thresh_transf"]:
                    num_matches_for_manual += 1
                    matched_predictions.append(pred)
                    matched_pairs.append([arg_id, man, pred, value_ngram, value_seq, value_transf, -1, 1, topic])          

            if num_matches_for_manual == 0: # category 'missed'
                matched_pairs.append([arg_id, man, "", 0, 0, 0, 0, 0, topic])
            else:
                for i, mp in enumerate(matched_pairs):
                    if mp[6] == -1:
                        matched_pairs[i][6] = num_matches_for_manual

        for not_matched in [pred for pred in preds if pred not in matched_predictions]: # category 'spurious'
            matched_pairs.append([arg_id, "", not_matched, 0, 0, 0, 0, 0, topic])
        matched_pairs.append(["", "", "", "", "", "", "", "", ""])

        # count to how many different manual segments a predicted segment is matched (incomplete)
        matched_predictions_nums = Counter(matched_predictions)
        dfout = pd.DataFrame(matched_pairs, columns=cols)
        dfout["matches_for_pred"] = [matched_predictions_nums[elem] if elem != "" else "" for elem in list(dfout["segment"])]
        dfout.to_csv(outfile, mode="a", index=False, header=not exists(outfile))


def get_predicted_segments(pred_path, model):
    '''Returns a list of dataframes (one dataframe per argument)'''
    suffix = get_file_suffix(model)
    seg_file = f"segments_{model}{suffix}.csv"
    df = pd.read_csv(pred_path + seg_file)
    seg_dfs = split_df_at_empty_lines(df)

    return seg_dfs
    

def prepare_segments(df, df_pred):
    df = df.dropna(how="all")
    df = df.reset_index()
    manual, preds = [], []
    for elem in list(df["key_statement"]):
        if str(elem) != "nan" and str(elem) not in manual:
            manual.append(elem)

    df_pred = df_pred.dropna(how="all")
    df_pred = df_pred.reset_index()

    for i, p in enumerate(list(df_pred["segment"])):
        # filter short segments
        if str(p).count(" ") > 2 and str(p) not in preds:
            preds.append(p)
    return manual, preds



class SegmentMatcher():
    def __init__(self, approaches, thresholds, ngram, gpt_prompt):
        self.approaches = approaches
        self.segmenter = cfg["segmentation_approach"]
        self.thresholds = thresholds
        self.ngram = ngram
        self.model_transformer = cfg["sent_transformer"]
        self.model_ags21 = SentenceTransformer(cfg["ags21_similarity_model"])
        self.prompt = gpt_prompt

        dfall = pd.read_csv(cfg["key_statements_groundtruth"], encoding="iso-8859-1")
        self.arg_dfs = split_df_at_empty_lines(dfall)


    def get_ngrams(self, tokens, n):
        return list(zip(*[tokens[i:] for i in range(n)]))

    def get_ngram_overlap(self, mantok, ptok, n_gram):
        mang = self.get_ngrams(mantok, n_gram)
        modelg = list(set(self.get_ngrams(ptok, n_gram)))
        overlaps = sum(p in mang for p in modelg)
        overlap_ratio = round(overlaps/len(mang), 3)
        return overlap_ratio


    def get_gpt_overlap(self, manual, model, approach):
        prompting = getattr(src.llms, f"prompt_{approach}")
        overlap = prompting(self.prompt.replace("AAA", model).replace("BBB", manual))
        if "%" in overlap:
            overlap_num = round(float(overlap.split("%")[0].split(" ")[-1]) / 100, 2)
        else:
            overlap_num = 0.0
        print(overlap, "->", overlap_num)
        return overlap_num
    

    def get_list_without_duplicates(self, items):
        outlist, seen = [], []
        for elem in items:
            if str(elem) != "nan" and elem not in seen:
                outlist.append(elem)
                seen.append(elem)
        return outlist
        

    def get_segment_pairs_groundtruth(self):
        '''
        Iterate all predicted segments for all manual segments and return a nested list 
        (one sublist for each argumentative text) of matches, e.g. for one argumentative
        text with two manual segments m1 and m2 and three predicted segments c1, c2, c3
        results in sublist [1,0,0,0,0,1] if m1-c2 and m2-c3 belong together 
        - corresponding to m1-c1, m1-c2, m1-c3, m2-c1, m2-c2, m2-c3 
        '''
        gt = []
        for df in self.arg_dfs:
            arggt = []
            df = df.dropna(how="all")
            df = df.reset_index()
            manual = self.get_list_without_duplicates(list(df["key_statement"]))
            predictions = self.get_list_without_duplicates(list(df[self.segmenter]))
            for man in manual:
                matches = list(df[df.manual == man][self.segmenter])
                for pred in predictions:
                    arggt.append(1) if pred in matches else arggt.append(0)
            gt.append(arggt)
        return gt


    def check_precomputation(self, man, model, approach):
        '''save computed overlap for transformer and GPT in file and read afterwards'''
        overlap_file = cfg["split_data"] + f"overlap_{approach}.csv"
        if exists(overlap_file):
            df = pd.read_csv(overlap_file)
            df["overlap"] = df["overlap"].replace(',','.', regex=True).astype(float)
            row = df.loc[(df["key_statement"] == man) & (df["segment"] == model)]
            if not row.empty:
                return list(row["overlap"])[0]
        
        if approach == "transformer":
            overlap = get_embedding_overlap(self.model_transformer, man, model)
        elif approach == "ags21":
            overlap = get_embedding_overlap(self.model_ags21, man, model)
        else:
            overlap = self.get_gpt_overlap(man, model, approach)
        out_df = pd.DataFrame([[man, model, overlap]], columns=["key_statement", "model", "overlap"])
        out_df.to_csv(overlap_file, mode="a", index=False, header=not exists(overlap_file))
        return overlap


    def get_overlap(self, approach, man, pred, mantok, ptok):
        '''Calculate the overlap between a manual and a predicted segment with the given approach'''
        if approach == "ngram":
            return self.get_ngram_overlap(mantok, ptok, self.ngram)
        elif approach == "sequencematcher":
            return round(SequenceMatcher(None, man, pred).ratio(),3)
        elif approach in ["transformer", "ags21", "gpt4"]:
            return self.check_precomputation(man, pred, approach)
        else:
            print("unknown approach", approach)
            exit()


    def get_segment_pairs_predicted(self, approach):
        '''
        Iterate all argumentative texts. For each argumentative text, iterate all 
        manual segments and all predicted segments by the model to find the
        predictions that match the current manual segment. Save in a dict with different
        thresholds whether the computed overlap is below (0) or above (1) the threshold.
        '''
        matches_per_threshold = {}
        for thresh in self.thresholds:
            matches_per_threshold[thresh] = []
        for i, df in enumerate(self.arg_dfs):
            for thresh in self.thresholds:
                matches_per_threshold[thresh].append([])
            df = df.dropna(how="all")
            df = df.reset_index()
            manual = self.get_list_without_duplicates(list(df["key_statement"]))
            predictions = self.get_list_without_duplicates(list(df[self.segmenter]))
            for m, man in enumerate(manual):             # iterate manual segments...
                mantok = nltk.word_tokenize(man)
                for p, model in enumerate(predictions):  # ... and check similarity with predicted segments
                    modeltok = nltk.word_tokenize(model)
                    value = self.get_overlap(approach, man, model, mantok, modeltok)
                    for thresh in self.thresholds:
                        matches_per_threshold[thresh][i].append(1) if value > thresh else matches_per_threshold[thresh][i].append(0)

        return matches_per_threshold

    
    def evaluate_approach(self, approach):
        '''
        Evaluate the given matching approach for different thresholds (i.e., consider two segments as matched if threshold exceeded).
        Collect precision, recall and F1 for each threshold in a dataframe
        param approach: matching approach (e.g. ngram, sequencematcher, transformer, gpt4)
        '''
        data = [["precision", "recall", "f1"]]
        matches_per_threshold = self.get_segment_pairs_predicted(approach)
        gt = self.get_segment_pairs_groundtruth()
        gts = [e for sub in gt for e in sub]
        
        for thresh in self.thresholds:
            matches = matches_per_threshold[thresh] 
            preds = [a for arg in matches for a in arg]
            prec = metrics.precision_score(gts, preds)
            recall = metrics.recall_score(gts, preds)
            f1 = metrics.f1_score(gts, preds)
            data.append([round(prec,4), round(recall,4), round(f1,4)])

        print(data)
        return pd.DataFrame(data, columns=[approach, approach, approach])


if __name__ == "__main__":
    match_segments("sentence_oracle", outfold=cfg["split_data"])
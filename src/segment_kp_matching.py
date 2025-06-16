import pandas as pd
from os.path import exists

from src.utils import create_file, get_embedding_overlap, parameter_check, get_file_suffix
from src.segment_matching import SegmentMatcher, get_predicted_segments
from config import cfg


def match_segments_with_kps(model, resultsfile, matchfolder, segfile=""):
    '''
    Match segments to IBM key points
    - param model: the segmentation model
    - param resultspath: file to collect the coverage of each key point per model
    - param matchfolder: folder where segments and matched key point per model are stored
    - param segfile (optional): define file with segments (other than default file)
    - param filter (optional): remove segments that are not argumentative
    '''

    if segfile == "":
        suffix = get_file_suffix(model)
        segfile = f"{cfg['split_data']}{model}/sample_split_train_segments_{model}{suffix}.csv"

    df_preds = pd.read_csv(segfile)
    df_kps = pd.read_csv(cfg["ibm_data"] + "key_points_all.csv")
    if cfg["filter"]:
        model = model + "_filt"

    resultspath = matchfolder + resultsfile
    matchpath = matchfolder + f"matched_kps_{model}.csv"
    if cfg["check_params"]:
        parameter_check([], {"model": model, "resultspath": resultspath, "matchpath": matchpath, "segfile": segfile})
    if not create_file(matchpath, append=True):
        exit()
    if not exists(resultspath):
        df_kps.drop(["key_point_id", "stance"], axis=1, inplace=True)
        cols = list(df_kps)
        cols[0], cols[1] = cols[1], cols[0]
        df_kps = df_kps[cols]
        df_kps.to_csv(resultspath, index=False)

    results = pd.read_csv(resultspath)
    results[model] = [0] * len(results.index)
    threshold = 0.9
    matcher = SegmentMatcher(cfg["matching_approaches"], [cfg["thresh_ngram"],  
                             cfg["thresh_seq"], cfg["thresh_transf"]], cfg["n_ngram"], "")

    
    topic = ""
    col = "segment" if model != "manual" else "key_statement"
    for i, row in df_preds.iterrows():
        seg = row[col]
        if str(seg) == "nan":
            continue
        if cfg["filter"] and row["arg_class"] == "NoArgument":
                continue
        if row["topic"] != topic:
            topic = row["topic"]
            kps = list(df_kps[df_kps["topic"] == topic]["key_point"])
            
        similarities = {}
        for kp in kps:
            sim = get_embedding_overlap(matcher.model_ags21, seg, kp)
            similarities[kp] = sim
        max_sim = max(list(similarities.values()))
        if max_sim > threshold:
            most_similar_kps = [k for k, s in similarities.items() if s == max_sim]
            for i, kp in enumerate(most_similar_kps):
                results.loc[results["key_point"] == kp, model] += 1
                write_seg = seg if i == 0 else ""
                data = [row["arg_id"], write_seg, kp, max_sim, topic]
                out_df = pd.DataFrame([data], columns=["arg_id", "segment", "keypoint", "similarity", "topic"])
                out_df.to_csv(matchpath, mode="a", index=False, header=not exists(matchpath))

    results.to_csv(resultspath, index=False)


def get_kp_coverage(datapath, gt="key_statement"):
    df = pd.read_csv(datapath)
    models = list(df)[2:]
    manual_covered = len([e for e in list(df[gt]) if e > 0])
    print(f"num KPs covered by manual segments = {manual_covered}")
    models.remove(gt)
    uncovered = {model: 0 for model in models}
    addcovered = {model: 0 for model in models}
    eqcovered = {model: 0 for model in models}
    for i, row in df.iterrows():
        for model in models:
            if row[gt] > 0 and row[model] > 0:
                eqcovered[model] += 1
            elif row[gt] == 0 and row[model] > 0:
                addcovered[model] += 1
            elif row[gt] > 0 and row[model] == 0:
                uncovered[model] += 1
    print("uncovered", uncovered)
    print("addcovered", addcovered)
    print("equally covered", eqcovered)
    for mod, num in eqcovered.items():
        print(f"Coverage for {mod} = {round(num/manual_covered, 4)}")


if __name__ == "__main__":
    matchfolder = cfg["split_data"] + "matched_keypoints/"
    resultsfile = "covered_keypoints_maxsim+thresh0.9_ags21.csv"
    get_kp_coverage(matchfolder + resultsfile, gt="manual")
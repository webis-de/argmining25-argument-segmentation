import os

from src.segment_text_matching import map_segments_to_original_text, adapt_seg_boundaries
from src.segment_kp_matching import match_segments_with_kps, get_kp_coverage
from config import cfg



def segment_key_point_matching():
    '''Match segments and key points and evaluate'''
    model = cfg["segmentation_approach"]
    matchfolder = cfg["split_data"] + "matched_keypoints/"
    os.makedirs(matchfolder, exist_ok=True)
    resultsfile = "covered_keypoints_maxsim+thresh0.9_ags21.csv"
    if not os.path.exists(matchfolder + resultsfile):
        seg_file = cfg["key_statements_groundtruth"]
        match_segments_with_kps("key_statement", resultsfile, matchfolder, segfile=seg_file)
    seg_file = cfg["split_data"] + f"{model}/segments_{model}.csv"
    match_segments_with_kps(model, resultsfile, matchfolder, segfile=seg_file)
    get_kp_coverage(matchfolder + resultsfile, gt="manual")



def segment_boundary_mapping():
    '''Mapping (edited) segments to original text'''
    model = cfg["segmentation_approach"]
    map_segments_to_original_text(model)
    adapt_seg_boundaries(model)



if __name__ == "__main__":
    segment_key_point_matching()
# Segmentation of Argumentative Texts
We address the task of segmenting arbitrary argumentative texts into argumentative segments by separating different key statements from each other.

## Usage
Install the necessary requirements via via `pip install -r requirements.txt`. (Note: we use Python 3.10.13)

Adjust the parameters and approaches for the different steps in `config.py` and choose the functions to be executed in `main.py`. Alternatively, you can run the modules directly, e.g., `python -m src.segmentation_step`.

## External code and data
Download the [UKP argument-classification model](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_classification_ukp_all_data.zip). The necessary code is already included in this repository, taken from the [official repository](https://github.com/UKPLab/acl2019-BERT-argument-classification-and-clustering/tree/master/argument-classification).

To get the file `key_points_all.csv` in `data/ibm_data/`, download all key points from the [IBM Key Point Analysis Shared Task](https://github.com/IBM/KPA_2021_shared_task/) ([train/dev](https://github.com/IBM/KPA_2021_shared_task/tree/main/kpm_data), [test](https://github.com/IBM/KPA_2021_shared_task/tree/main/test_data)) and collect them in a single file.

## Data
The texts are taken from the args.me dataset ([Ajjour et al., 2019](https://dl.acm.org/doi/abs/10.1007/978-3-030-30179-8_4)) where arguments from different debater portals were crawled. 

Important files like the testset and the manually extracted key statements are uploaded on [Zenodo](https://zenodo.org/uploads/14865977) and should be put into `data/split_data`.

The automatically created segments (e.g. by PaLM and GPT-4) are stored in separate folders in `data/split_data/` named similarly to the segmentation approach.

## Notes
We requested the matching model that performed best in the Key Point Analysis shared task from the authors ([Alshomary et al. 2021](https://aclanthology.org/2021.argmining-1.19/)) to use it for the matching with key points.

## Citation
If you use the dataset or the code in your research, please cite the following paper describing the segmentation by key statements:

> Ines Zelch, Matthias Hagen, Benno Stein, and Johannes Kiesel. [Segmentation of Argumentative Texts by Key Statements for Argument Mining from the Web.](https://webis.de/publications.html#zelch_2025a), In Proceedings of the _12th Workshop on Argument Mining_, July 2025.


You can use the following BibTeX entry for citation:

```bibtex
@InProceedings{zelch:2025,
    author = {Ines Zelch and Matthias Hagen and Benno Stein and Johannes Kiesel},
    booktitle = {12th Workshop on Argument Mining (ArgMining 2025) at ACL},
    doi = {tbd},
    editor = {tbd},
    month = jul,
    numpages = 15,
    pages = {tbd},
    title = {{Segmentation of Argumentative Texts by Key Statements for Argument Mining from the Web}},
    url = tbd,
    year = 2025
}
```
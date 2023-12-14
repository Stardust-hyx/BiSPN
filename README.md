# BiSPN

Source code for the paper [**BiSPN: Generating Entity Set and Relation Set Coherently in One Pass**](https://aclanthology.org/2023.findings-emnlp.136/) (Findings of EMNLP 2023).

##  Requirements

```
Python >= 3.8
```
Required packages can be installed with:
```
pip install -r requirements.txt
```

##  Data

For ACE05, we use the preprocessing code from DyGIE repo. The BioRelEx dataset is available at https://github.com/YerevaNN/BioRelEx. The ADE dataset can be downloaded via the script at https://github.com/markus-eberts/spert. To obtain the Text2DT dataset, please follow the request instruction at http://www.cips-chip.org.cn/2022/eval3.

Place the data as follows:
```text
    - data/
        - ACE05
            - train.json
            - test.json
            - dev.json
        - BioRelEx
            - train.json
            - dev.json
        - ADE
            ...
        - Text2DT
            ...
```

##  Training & Evaluation
Run the following command:
```shell
bash {ace2005|biorelex|ade|text2dt}.sh
```

## Cite
If you find our work helpful, please consider citing our paper.
```
@inproceedings{he2023bispn,
  title={BiSPN: Generating Entity Set and Relation Set Coherently in One Pass},
  author={He, Yuxin and Tang, Buzhou},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={2066--2077},
  year={2023}
}
```
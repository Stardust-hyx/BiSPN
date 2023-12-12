# BiSPN

Source code for the paper **BiSPN: Generating Entity Set and Relation Set Coherently in One Pass**.

##  Requirements

```
Python >= 3.7   
PyTorch == 1.6.0 
Transformers == 3.4.0
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
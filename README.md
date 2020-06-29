# Question Answering using character-level RNN over babi (FAIR dataset) and SQUAD (Stanford dataset)

The different models will first be challenged with the bAbI dataset from FAIR and the SQuAD dataset from Stanford.

Name | Description
--- | ---
/babi| babi dataset and utilities
/squad | squad dataset and utilities
/qrn | contains the QRN cell
char2word.py | Char2Word-only module (on bAbI dataset)
qrn.py | implementation of the QRN model (on bAbI dataset)
char2word_qrn.py | implementation of the QRN model w/ Char2Word module (on bAbI dataset)
squad.py | implementation of the QRN model w/ Char2Word module (on SQuAD dataset)


### Files :

* **babi_formatting.py** - processes the dataset for character-level handling.
* **/datasets** - repository for datasets
* **/model/dataset.py** - loads the data
* **/model/model.py** - constructs model_fn that will be fed into the Estimator in main.py
* **main.py** - runs TensorFlow instances to train and evaluate the model
* **run.sh** - runs main.py given a dataset and a seed
* **char2word.py** - contains the Char2Word block

### Usage (on bAbI dataset):

Query-Reduction Network without Char2Word:
```bash
python qrn.py
```

Query-Reduction Network with Char2Word:
```bash
python char2word_qrn.py
```

EntNet:
---WORK IN PROGRESS---

### Usage (on SQuAD dataset):

Query-Reduction Network with Char2Word:
```bash
python squad.py
```

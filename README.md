# Question Answering using character-level RNN over babi (FAIR dataset) and SQUAD (Stanford dataset)

... WORK IN PROGRESS ...

The models will first be challenged with the bAbI dataset from FAIR.
Next, there will be experiments on the SQuAD dataset.


### Files :

* **babi_formatting.py** - processes the dataset for character-level handling.
* **/datasets** - repository for datasets
* **/model/dataset.py** - loads the data
* **/model/model.py** - constructs model_fn that will be fed into the Estimator in main.py
* **main.py** - runs TensorFlow instances to train and evaluate the model
* **run.sh** - runs main.py given a dataset and a seed
* **char2word.py** - contains the Char2Word block

### Usage :

Query-Reduction Network without Char2Word:
```bash
python qrn.py
```

Query-Reduction Network with Char2Word:
```bash
python char2word_qrn.py
```

Entity Neural Network (first, run format_babi.py to preprocess the data):
```bash
bash run.sh
```

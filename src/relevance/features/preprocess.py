import cPickle
import numpy as np
import pandas as pd
from relevance.utils.nlp_utils import clean_text
from relevance.config import config

# load data
print("Load data...")

dfTrain = pd.read_csv(config.original_train_data_path).fillna("")
dfTest = pd.read_csv(config.original_test_data_path).fillna("")
# number of train/test samples
num_train, num_test = dfTrain.shape[0], dfTest.shape[0]

print("Done.")


# Preprocess data
print("Pre-process data...")

# insert fake label for test
dfTest["median_relevance"] = np.ones((num_test))
dfTest["relevance_variance"] = np.zeros((num_test))

# insert sample index
dfTrain["index"] = np.arange(num_train)
dfTest["index"] = np.arange(num_test)

# one-hot encode the median_relevance
for i in range(config.n_classes):
    dfTrain["median_relevance_%d" % (i + 1)] = 0
    dfTrain["median_relevance_%d" % (i + 1)][dfTrain["median_relevance"] == (i + 1)] = 1

# query ids
qid_dict = dict()
for i, q in enumerate(np.unique(dfTrain["query"]), start=1):
    qid_dict[q] = i

# insert query id
dfTrain["qid"] = list(map(lambda q: qid_dict[q], dfTrain["query"]))
dfTest["qid"] = list(map(lambda q: qid_dict[q], dfTest["query"]))

# clean text
clean = lambda line: clean_text(line, drop_html_flag=config.drop_html_flag)
dfTrain = dfTrain.apply(clean, axis=1)
dfTest = dfTest.apply(clean, axis=1)

print("Done.")


# save data
print("Save data...")

with open(config.processed_train_data_path, "wb") as f:
    cPickle.dump(dfTrain, f, -1)
with open(config.processed_test_data_path, "wb") as f:
    cPickle.dump(dfTest, f, -1)

print("Done.")

import cPickle
from sklearn.cross_validation import StratifiedKFold
from relevance.config import config


if __name__ == "__main__":

    # load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = cPickle.load(f)

    skf = [0] * config.n_runs
    for stratified_label, key in zip(["relevance", "query"],
                                     ["median_relevance", "qid"]):
        for run in range(config.n_runs):
            random_seed = 2018 + 1000 * (run + 1)
            skf[run] = StratifiedKFold(dfTrain[key],
                                       n_folds=config.n_folds,
                                       shuffle=True,
                                       random_state=random_seed)
            for fold, (validInd, trainInd) in enumerate(skf[run]):
                print("================================")
                print("Index for run: %s, fold: %s" % (run + 1, fold + 1))
                print("Train (num = %s)" % len(trainInd))
                print(trainInd[:10])
                print("Valid (num = %s)" % len(validInd))
                print(validInd[:10])
        with open("%s/stratifiedKFold.%s.pkl" %
                  (config.data_folder, stratified_label), "wb") as f:
            cPickle.dump(skf, f, -1)

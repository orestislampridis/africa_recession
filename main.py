import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from configuration import ModelConfiguration, IMBALANCE_OPTION_SMOTE, IMBALANCE_OPTION_OTHER, \
    COST_OPTION_REJECTION_SAMPLING, COST_OPTION_MODEL, EXPLAIN_OPTION_WHITE_BOX, EXPLAIN_OPTION_BLACK_BOX
from cost_sensitive import Cost

if __name__ == '__main__':
    cost = Cost(cost_recession_predicted_growth=3, cost_growth_predicted_recession=1)

    # Select one of: IMBALANCE_OPTION_SMOTE, IMBALANCE_OPTION_OTHER
    imbalance_option = IMBALANCE_OPTION_SMOTE

    # Select one of: COST_OPTION_REJECTION_SAMPLING, COST_OPTION_MODEL
    cost_option = COST_OPTION_MODEL

    # Select one of: EXPLAIN_OPTION_WHITE_BOX, EXPLAIN_OPTION_BLACK_BOX
    explain_option = EXPLAIN_OPTION_WHITE_BOX

    config = ModelConfiguration(cost=cost,
                                imbalance_option=imbalance_option,
                                cost_option=cost_option,
                                explain_option=explain_option)

    data = pd.read_csv("./dataset/africa_recession.csv", delimiter=",")

    x = np.asarray(data.iloc[:, :-2])
    y = np.asarray(data.iloc[:, -1])

    x, y = config.transform_dataset(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=8, shuffle=True)

    model = config.create_model()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred, average="macro")
    precision = metrics.precision_score(y_test, y_pred, average="macro")
    f1 = metrics.f1_score(y_test, y_pred, average="macro")
    print("\tAccuracy: %f" % accuracy)
    print("\tRecall: %f" % recall)
    print("\tPrecision: %f" % precision)
    print("\tF1: %f" % f1)

    model.save_model("./models/model.sav")
    print("END")

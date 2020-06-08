import numpy as np
import pandas as pd

from configuration import ModelConfiguration, IMBALANCE_OPTION_SMOTE, IMBALANCE_OPTION_OTHER,\
    COST_OPTION_REJECTION_SAMPLING, COST_OPTION_MODEL, EXPLAIN_OPTION_WHITE_BOX, EXPLAIN_OPTION_BLACK_BOX
from cost_sensitive import Cost

if __name__ == '__main__':
    cost = Cost(cost_recession_predicted_growth=0, cost_growth_predicted_recession=0)

    # Select one of: IMBALANCE_OPTION_SMOTE, IMBALANCE_OPTION_OTHER
    imbalance_option = IMBALANCE_OPTION_SMOTE

    # Select one of: COST_OPTION_REJECTION_SAMPLING, COST_OPTION_MODEL
    cost_option = COST_OPTION_MODEL

    # Select one of: EXPLAIN_OPTION_WHITE_BOX, EXPLAIN_OPTION_BLACK_BOX
    explain_option = EXPLAIN_OPTION_WHITE_BOX

    config = ModelConfiguration(cost=cost, imbalance_option=imbalance_option,
                                cost_option=cost_option,
                                explain_option=explain_option)

    data = pd.read_csv("./dataset/africa_recession.csv", delimiter=",")

    x = np.asarray(data.iloc[:, :-2])
    y = np.asarray(data.iloc[:, -1])

    model = config.create_model()

    model.fit(x, y)
    y = model.predict(x)

    print("END_OF_LINE")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from configuration import ModelConfiguration, IMBALANCE_OPTION_NONE, IMBALANCE_OPTION_SMOTE, \
    IMBALANCE_OPTION_TOMEK_OVERSAMPLE, \
    COST_OPTION_NONE, COST_OPTION_REJECTION_SAMPLING, COST_OPTION_MODEL, EXPLAIN_OPTION_WHITE_BOX, \
    EXPLAIN_OPTION_BLACK_BOX, \
    ad_hoc_try_logistic_reg
from cost_sensitive import Cost

create_ad_hoc_model = True


def try_model_with_options(cost: Cost, imbalance_option: str, cost_option: str, explain_option: str):
    # create configuration from options
    config = ModelConfiguration(cost=cost,
                                imbalance_option=imbalance_option,
                                cost_option=cost_option,
                                explain_option=explain_option)

    # read and transform dataset
    print("Reading data-set...")
    data = pd.read_csv("./dataset/africa_recession.csv", delimiter=",")

    x = np.asarray(data.iloc[:, :-1])
    y = np.asarray(data.iloc[:, -1])

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    # plit train test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=8)

    x_train, y_train = config.transform_dataset(x_train, y_train)

    # create model based on the provided options
    model = config.create_model()
    print("Created ML model", model.model_name(), ".")

    # train model
    print("Training ML model...")
    model.fit(x_train, y_train)

    # model evaluation
    print("Evaluating ML model...")
    y_pred = model.predict(x_test)
    model.print_metrics(y_test, y_pred)

    # save model for explanation
    model.save_model("./models")

    # Ad hoc logistic regression
    if config.cost_option == COST_OPTION_NONE and create_ad_hoc_model:
        print("-------------------------------------------------------------------------------------------")
        print()
        print()
        print("-------------------------------------------------------------------------------------------")
        print("AD-HOC Model with options:", imbalance_option, cost_option, explain_option)
        ad_hoc_try_logistic_reg(imbalance_option, x_train, y_train, x_test, y_test, "./models")


if __name__ == '__main__':
    costs = [
        Cost(cost_recession_predicted_growth=4, cost_growth_predicted_recession=1)
    ]

    # Select one of: IMBALANCE_OPTION_SMOTE, IMBALANCE_OPTION_TOMEK_UNDERSAMPLE
    imbalance_options = [IMBALANCE_OPTION_NONE, IMBALANCE_OPTION_SMOTE, IMBALANCE_OPTION_TOMEK_OVERSAMPLE]

    # Select one of: COST_OPTION_REJECTION_SAMPLING, COST_OPTION_MODEL
    cost_options = [COST_OPTION_NONE, COST_OPTION_REJECTION_SAMPLING, COST_OPTION_MODEL]

    # Select one of: EXPLAIN_OPTION_WHITE_BOX, EXPLAIN_OPTION_BLACK_BOX
    explain_options = [EXPLAIN_OPTION_WHITE_BOX, EXPLAIN_OPTION_BLACK_BOX]

    for imbalance_option in imbalance_options:
        for cost_option in cost_options:
            for explain_option in explain_options:
                print()
                print("-------------------------------------------------------------------------------------------")
                print("Model with options:", imbalance_option, cost_option, explain_option)
                try_model_with_options(costs[0], imbalance_option, cost_option, explain_option)
                print("-------------------------------------------------------------------------------------------")
                print()

    print("END")

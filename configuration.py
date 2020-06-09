import pickle

import numpy as np

from costcla import CostSensitiveDecisionTreeClassifier, CostSensitiveRandomForestClassifier, metrics as cost_metrics
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from cost_sensitive import Cost

IMBALANCE_OPTION_SMOTE = "kImbalanceSMOTE"
IMBALANCE_OPTION_OTHER = "kImbalanceOther"  # TODO: Please replace with final selected method

COST_OPTION_REJECTION_SAMPLING = "kCostRejectionSample"
COST_OPTION_MODEL = "kCostModel"

EXPLAIN_OPTION_WHITE_BOX = "kExplainableWhiteBox"
EXPLAIN_OPTION_BLACK_BOX = "kExplainableBlackBox"


def not_implemented(x, y):
    return x, y

class ModelConfiguration:
    cost = Cost(0, 0)
    imbalance_option = ""
    cost_option = ""
    explain_option = ""

    def __init__(self, cost: Cost, imbalance_option: str = IMBALANCE_OPTION_SMOTE,
                 cost_option: str = COST_OPTION_MODEL, explain_option: str = EXPLAIN_OPTION_WHITE_BOX):
        self.cost = cost

        # Raise error, unrecognized value in imbalance option
        if imbalance_option != IMBALANCE_OPTION_SMOTE and imbalance_option != IMBALANCE_OPTION_OTHER:
            raise ValueError("Unexpected IMBALANCE option specified.")

        self.imbalance_option = imbalance_option

        # Raise error, unrecognized value in cost option
        if cost_option != COST_OPTION_REJECTION_SAMPLING and cost_option != COST_OPTION_MODEL:
            raise ValueError("Unexpected COST option specified.")

        self.cost_option = cost_option

        # Raise error, unrecognized value in explain option
        if explain_option != EXPLAIN_OPTION_WHITE_BOX and explain_option != EXPLAIN_OPTION_BLACK_BOX:
            raise ValueError("Unexpected EXPLAIN option specified.")

        self.explain_option = explain_option

    def transform_dataset(self, x, y):
        # todo add proper methods

        if self.imbalance_option == IMBALANCE_OPTION_SMOTE:
            # call smote func
            x, y = not_implemented(x, y)
        elif self.imbalance_option == IMBALANCE_OPTION_OTHER:
            x, y = not_implemented(x, y)

        if self.cost_option == COST_OPTION_REJECTION_SAMPLING:
            x, y = not_implemented(x, y)

        return x, y

    def __get_model(self):

        if self.explain_option == EXPLAIN_OPTION_WHITE_BOX:
            if self.cost_option == COST_OPTION_MODEL:
                return CostSensitiveDecisionTreeClassifier()
            else:
                return DecisionTreeClassifier()
        else:
            if self.cost_option == COST_OPTION_MODEL:
                return CostSensitiveRandomForestClassifier()
            else:
                return RandomForestClassifier()

    def create_model(self):
        return MetaModel(configuration=self, ml_model=self.__get_model())


class MetaModel:
    configuration = ModelConfiguration(Cost(0, 0))
    ml_model = None

    def __init__(self, configuration, ml_model):
        self.configuration = configuration
        self.ml_model = ml_model

    def fit(self, x, y):
        if isinstance(self.ml_model, CostSensitiveDecisionTreeClassifier):
            costs = []

            for current_y in y:
                costs_array = self.configuration.cost.costcla_cost_array(current_y)
                costs.append(costs_array)

            costs = np.asarray(costs)

            self.ml_model.fit(x, y, cost_mat=costs)
        elif isinstance(self.ml_model, CostSensitiveRandomForestClassifier):
            costs = []

            for current_y in y:
                costs_array = self.configuration.cost.costcla_cost_array(current_y)
                costs.append(costs_array)

            costs = np.asarray(costs)

            self.ml_model.fit(x, y, cost_mat=costs)
        elif isinstance(self.ml_model, DecisionTreeClassifier):
            self.ml_model.fit(x, y)
        elif isinstance(self.ml_model, RandomForestClassifier):
            self.ml_model.fit(x, y)
        else:  # try to call fit unsafely, will raise error with wrong class
            self.ml_model.fit(x, y)

    def predict(self, x):
        if isinstance(self.ml_model, CostSensitiveDecisionTreeClassifier):
            return self.ml_model.predict(x)
        elif isinstance(self.ml_model, CostSensitiveRandomForestClassifier):
            return self.ml_model.predict(x)
        elif isinstance(self.ml_model, DecisionTreeClassifier):
            return self.ml_model.predict(x)
        elif isinstance(self.ml_model, RandomForestClassifier):
            return self.ml_model.predict(x)
        else:  # try to call predict unsafely, will raise error with wrong class
            return self.ml_model.predict(x)

    def print_metrics(self, y_test, y_pred):
        if self.configuration.cost_option == COST_OPTION_MODEL:
            costs = []

            for current_y in y:
                costs_array = self.configuration.cost.costcla_cost_array(current_y)
                costs.append(costs_array)

            costs = np.asarray(costs)

            cost_loss = cost_metrics.cost_loss(y_test, y_pred, costs)
            print("\nCost loss: %f" % cost_loss)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred, average="macro")
        precision = metrics.precision_score(y_test, y_pred, average="macro")
        f1 = metrics.f1_score(y_test, y_pred, average="macro")

        print("\tAccuracy: %f" % accuracy)
        print("\tRecall: %f" % recall)
        print("\tPrecision: %f" % precision)
        print("\tF1: %f" % f1)

    def save_model(self, path):
        pickle.dump(self.ml_model, open(path, "wb"))
        return

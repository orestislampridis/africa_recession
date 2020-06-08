import numpy as np

from costcla import CostSensitiveDecisionTreeClassifier, CostSensitiveRandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from cost_sensitive import Cost

IMBALANCE_OPTION_SMOTE = "kImbalanceSMOTE"
IMBALANCE_OPTION_OTHER = "kImbalanceOther"  # TODO: Please replace with final selected methods

COST_OPTION_REJECTION_SAMPLING = "kCostRejectionSample"
COST_OPTION_MODEL = "kCostModel"

EXPLAIN_OPTION_WHITE_BOX = "kExplainableWhiteBox"
EXPLAIN_OPTION_BLACK_BOX = "kExplainableBlackBox"


class ModelConfiguration:
    cost = Cost(0, 0)
    imbalance_option = ""
    cost_option = ""
    explain_option = ""

    def __init__(self, cost: Cost, imbalance_option: str = IMBALANCE_OPTION_SMOTE,
                 cost_option: str = COST_OPTION_MODEL, explain_option: str = EXPLAIN_OPTION_WHITE_BOX):
        self.cost = cost
        self.imbalance_option = imbalance_option
        self.cost_option = cost_option
        self.explain_option = explain_option

    def transform_dataset(self, x, y):
        # todo add actual impl
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

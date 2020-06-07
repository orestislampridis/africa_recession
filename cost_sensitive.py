import pandas as pd
from costcla.sampling import cost_sampling


class Cost:
    # classes => 1 = Recession, 0 = No_Recession
    recession = 1
    growth = 0

    cost_recession_predicted_recession = 0
    cost_recession_predicted_growth = 0
    cost_growth_predicted_recession = 0
    cost_growth_predicted_growth = 0

    def __init__(self, cost_recession_predicted_growth: int, cost_growth_predicted_recession: int,
                 cost_recession_predicted_recession: int = 0, cost_growth_predicted_growth: int = 0):
        self.cost_recession_predicted_recession = cost_recession_predicted_recession
        self.cost_recession_predicted_growth = cost_recession_predicted_growth
        self.cost_growth_predicted_recession = cost_growth_predicted_recession
        self.cost_growth_predicted_growth = cost_growth_predicted_growth

    def cost_matrix(self):
        return [
            [self.cost_growth_predicted_growth, self.cost_recession_predicted_growth],
            [self.cost_growth_predicted_recession, self.cost_recession_predicted_recession]
        ]

    def cost_growth(self):
        cost = 0

        for row in self.cost_matrix():
            growth_cost = row[0]
            cost += growth_cost

        return cost

    def cost_recession(self):
        cost = 0

        for row in self.cost_matrix():
            recession_cost = row[1]
            cost += recession_cost

        return cost

    def recession_weight(self, x: list, y: list):
        n = 0
        n_recession = 0
        n_growth = 0

        for example_y in y:
            if example_y == self.growth:
                n_growth += 1
            elif example_y == self.recession:
                n_recession += 1
            else:
                raise AssertionError("More than 2 classes found in dataset.")

            n += 1

        cost_recession = self.cost_recession()
        cost_growth = self.cost_growth()

        return n * cost_recession / ((n_growth * cost_growth) + (n_recession * cost_recession))

    def growth_weight(self, dataset: pd.DataFrame):
        n = 0
        n_recession = 0
        n_growth = 0

        for row in dataset.values:
            example_class = row[-1]

            if example_class == self.growth:
                n_growth += 1
            elif example_class == self.recession:
                n_recession += 1
            else:
                raise AssertionError("More than 2 classes found in dataset.")

            n += 1

        cost_recession = self.cost_recession()
        cost_growth = self.cost_growth()

        return n * cost_growth / ((n_growth * cost_growth) + (n_recession * cost_recession))


def cost_sensitive_data_re_balance(x: list, y: list, cost: Cost):
    cost_matrix = cost.cost_matrix()
    # Flatten & Reshape cost matrix
    # costcla => [false positives, false negatives, true positives, true negatives]
    flat_cost_matrix = [cost_matrix[0][1], cost_matrix[1][0], cost_matrix[1][1], cost_matrix[0][0]]
    return cost_sampling(x, y, flat_cost_matrix)

#   http://albahnsen.github.io/CostSensitiveClassification/Models.html

def whitebox_model():
    return "decision tree"


def blackbox_model():
    return "random forest"

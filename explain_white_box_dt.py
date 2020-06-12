import pickle

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'dataset/africa_recession.csv', error_bad_lines=False)
class_names = ["no recession", "recession"]

X = df.drop(columns=['growthbucket'])
y = df.growthbucket

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
X_resampled, y_resampled = SMOTE(random_state=0).fit_resample(X_train, y_train)

# Filename of white box dt model to explain
filename = "tomek_undersample_no_cost_decision_tree_classifier.model"

# Load the model from disk
loaded_model = pickle.load(open("models/" + filename, 'rb'))

y_train_pred = loaded_model.predict(X_train)
y_predicted = loaded_model.predict(X_test)

# Print evaluation metrics
print("accuracy: ", accuracy_score(y_test, y_predicted))
print("precision: ", precision_score(y_test, y_predicted))
print("recall: ", recall_score(y_test, y_predicted))
print("f1 score: ", f1_score(y_test, y_predicted))

importances = loaded_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking of the decision tree
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, df.columns[f], importances[indices[f]]))

# Plot the top 10 feature importances of the decision tree
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, 14.5])
plt.show()

# Visualize the tree to obtain global explanation and export as pdf
dot_data = tree.export_graphviz(loaded_model, out_file=None,
                                feature_names=X.columns,
                                class_names=class_names,
                                filled=True, rounded=True,
                                special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("explanations/" + filename + "_global_explanation")

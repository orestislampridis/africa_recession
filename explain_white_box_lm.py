import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'dataset/africa_recession.csv', error_bad_lines=False)
class_names = ["no recession", "recession"]

X = df.drop(columns=['growthbucket'])
y = df.growthbucket

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
X_resampled, y_resampled = SMOTE(random_state=0).fit_resample(X_train, y_train)

# Filename of white box linear model to explain
filename = "tomek_undersample_no_cost_logistic_regression_classifier.model"

# Load the model from disk
loaded_model = pickle.load(open("models/" + filename, 'rb'))

y_train_pred = loaded_model.predict(X_train)
y_predicted = loaded_model.predict(X_test)

# Print evalution metrics
print("accuracy: ", accuracy_score(y_test, y_predicted))
print("precision: ", precision_score(y_test, y_predicted))
print("recall: ", recall_score(y_test, y_predicted))
print("f1 score: ", f1_score(y_test, y_predicted))

# Plot the weights assigned by the linear model for each feature
weights = loaded_model.coef_
model_weights = pd.DataFrame({'features': list(X.columns), 'weights': list(weights[0])})
model_weights = model_weights.reindex(model_weights['weights'].abs().sort_values(ascending=False).index)
model_weights = model_weights[(model_weights["weights"] != 0)]
print("Number of features:", len(model_weights.values))
f = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
sns.barplot(x="weights", y="features", data=model_weights)
plt.title("Intercept (Bias): " + str(loaded_model.intercept_[0]), loc='right')
plt.xticks(rotation=90)
plt.show()
f.savefig("explanations/" + filename + "_global_explanation.pdf", bbox_inches='tight')

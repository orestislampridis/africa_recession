import graphviz
import pandas as pd
from sklearn import tree
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def scale_data(dataframe):
    minmax = MinMaxScaler()
    columns = list(df.columns.values)
    scaled_df = minmax.fit_transform(dataframe)
    scaled_df = pd.DataFrame(scaled_df, columns=columns)
    return scaled_df


df = pd.read_csv(r'dataset/africa_recession.csv', error_bad_lines=False)
df = scale_data(df)

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

class_names = ["no recession", "recession"]

print(X)
print(y)
print("test")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)

model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=X.columns,
                                class_names=class_names,
                                filled=True, rounded=True,
                                special_characters=True)

print("accuracy: ", accuracy_score(y_test, y_predicted))
print("precision: ", precision_score(y_test, y_predicted))
print("recall: ", recall_score(y_test, y_predicted))
print("f1 score: ", f1_score(y_test, y_predicted))

graph = graphviz.Source(dot_data)
graph.render("Global_explanation")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss, OneSidedSelection, \
    NeighbourhoodCleaningRule
from imblearn.combine import SMOTETomek


# function to bar plot imbalance
def show_imbalance(X, y, resample="no resample"):
    target_count = y.value_counts()
    print('Examples in class 0:', target_count[0])
    print('Examples in class 1:', target_count[1])
    plt.figure(figsize=(9, 4))
    title = "Counts in each class with {}".format(resample)
    plt.title(title, fontsize=18)
    plt.ylabel('Number of examples', fontsize=12)
    plt.xlabel('Classes ', fontsize=12)
    target_count.plot(kind='bar')
    plt.show()


# function to scatter plot after pca
def pca_scatter(x, y, n=2):
    pca = PCA(n_components=n)
    x = pca.fit_transform(x)
    colors = ['green', 'blue']
    for l, c in zip(np.unique(y), colors):
        plt.scatter(
            x[y == l, 0],
            x[y == l, 1],
            c=c, label=l
        )
    plt.title('2 PCA components imbalanced dataset distribution')
    plt.legend(loc='upper left')
    plt.show()


# function to scale data between [0,1]
def scale_data(df):
    minmax = MinMaxScaler()
    columns = list(df.columns.values)
    scaled_df = minmax.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=columns)
    return scaled_df


# function to get best features
def get_best_features(df, k=10):
    best = SelectKBest(k=k)
    fit = best.fit(df.iloc[:, :-1], df.iloc[:, -1])
    feature_scores = pd.DataFrame(columns=['features', 'score'])
    feature_scores['score'] = fit.scores_
    feature_scores['features'] = df.iloc[:, :-1].columns
    feature_scores.sort_values(by=['score'], inplace=True, ascending=False)
    return feature_scores.iloc[:k, :]


# function to compare classifiers
def compare_classifiers(X_train, X_test, y_train, y_test):
    names = ["Nearest Neighbors", "RBF SVM",
             "Decision Tree", "Random Forest", "AdaBoost",
             "Naive Bayes"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5,
                               n_estimators=100,
                               max_features=1),
        AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                           n_estimators=600,
                           learning_rate=1.5,
                           algorithm="SAMME"),
        GaussianNB()]

    best = 0
    for clf, name in zip(classifiers, names):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        #     print("Classifier:",name)
        #     print("Accuracy:",accuracy_score(y_test, y_pred))
        #     print("Precission:",precision_score(y_test, y_pred))
        #     print("Recal:",recall_score(y_test, y_pred))
        #     print("f1_score:",f1_score(y_test, y_pred))
        #     print("\n")

        if best < f1_score(y_test, y_pred):
            best_name = name
            best = f1_score(y_test, y_pred)

    print("Best classifier", best_name, " f1 score:", best)


# Over-sampling methods
def random_over_sampler(x, y):
    print("----Random Oversampling----\n")
    sampler = RandomOverSampler(random_state=42)
    X, y = sampler.fit_sample(X_train, y_train)
    return X, y


def smote(x, y):
    print("----SMOTE----\n")
    sampler = SMOTE(random_state=42)
    X, y = sampler.fit_sample(X_train, y_train)
    return X, y


def borderline_smote(x, y):
    print("----Borderline SMOTE----\n")
    sampler = BorderlineSMOTE(random_state=42)
    X, y = sampler.fit_sample(X_train, y_train)
    return X, y


def kmeans_smote(x, y):
    print("----KMeans SMOTE----\n")
    sampler = KMeansSMOTE(random_state=42)
    X, y = sampler.fit_sample(X_train, y_train)
    return X, y


def adacyn(x, y):
    print("----ADASYN----\n")
    sampler = ADASYN(random_state=42)
    X, y = sampler.fit_sample(X_train, y_train)
    return X, y


# Under-sampling methods
def random_under_sampler(x, y):
    print("----Random Undersampling----\n")
    sampler = RandomUnderSampler(random_state=42)
    X, y = sampler.fit_sample(X_train, y_train)
    return X, y


def tomek_links(x, y):  # use with other resampler
    print("----TOMEK----\n")
    sampler = TomekLinks()
    X, y = sampler.fit_resample(X_train, y_train)
    return X, y


def near_miss(x, y, v=1):
    print("----Near Miss----\n")
    print("Version:", v)
    sampler = NearMiss(version=v)
    X, y = sampler.fit_resample(X_train, y_train)
    return X, y


def one_sided_selection(x, y):
    print("----One Sided Selection----\n")
    sampler = OneSidedSelection()
    X, y = sampler.fit_resample(X_train, y_train)
    return X, y


def neighbourhood_cleaning(x, y):
    print("----Neighbourhood Cleaning Rule----\n")
    sampler = NeighbourhoodCleaningRule()
    X, y = sampler.fit_resample(X_train, y_train)
    return X, y


# Combination of over- and under-sampling methods
def smotetomek(x, y):
    print("----Smote Tomek----\n")
    sampler = SMOTETomek()
    X, y = sampler.fit_resample(X_train, y_train)
    return X, y


# load data
df = pd.read_csv(r'dataset/africa_recession.csv', error_bad_lines=False)
print(df.head(3))

# minmax scale data
df = scale_data(df)

# uncomment to get 10 best features
# best_features = get_best_features(df,10)
# best_df_columns=best_features['features'].values.tolist()
# best_df_columns.append('growthbucket')
# df = df[df.columns.intersection(best_df_columns)]
# print("Best features")
# print(df.head(3))

# feature target split
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# no resampler
print("\n----No resample----\n")
show_imbalance(X_train, y_train)
pca_scatter(X_train, y_train)
compare_classifiers(X_train, X_test, y_train, y_test)
print("\n")

# apply smote
X_sm, y_sm = smote(X_train, y_train)
show_imbalance(X_sm, y_sm, resample="Smote")
pca_scatter(X_sm, y_sm)
compare_classifiers(X_sm, X_test, y_sm, y_test)
print("\n")

# apply random over sampling
X_ros, y_ros = random_over_sampler(X_train, y_train)
show_imbalance(X_ros, y_ros, resample="Random Oversampling")
pca_scatter(X_ros, y_ros)
compare_classifiers(X_ros, X_test, y_ros, y_test)
print("\n")

# apply smotetomek
X_st, y_st = smotetomek(X_train, y_train)
show_imbalance(X_st, y_st, resample="SmoteTomek")
pca_scatter(X_st, y_st)
compare_classifiers(X_st, X_test, y_st, y_test)
print("\n")
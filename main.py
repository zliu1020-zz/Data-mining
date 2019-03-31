import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import pydot
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO

balance_data = pd.read_csv(r"/Users/Ziyan/Desktop/ECE 356/Lab/Lab4/query_result.csv")
# get rid of playerID
X = balance_data.values[:, 1:-1]
# access the indicated column
Y = balance_data.values[:,-1]
X[np.isnan(X.tolist())] = 0

Y = Y.reshape(-1,1)
Y = Y.tolist()

features = ['G','AB','R','H','2b','3B',	'HR','RBI',	'SB',
            'CS','BB','SO','IBB','HBP','SH','SF','GIDP','W','L',
            'G2','GS','CG','SHO','SV','IPouts','H2','ER',
            'HR2','BB2','SO2','BAOpp','ERA','IBB2',	'WP',
            'HBP2','BK','BFP','GF','R2','SH2','SF2','GIDP2']

def generate_graph(tree, idx, criterion):
    dot = StringIO()
    export_graphviz(tree,
                    out_file=dot,
                    class_names=['Nominated', 'Elected'],
                    filled=True,
                   # feature_names=features,
                    rounded=True,
                    special_characters=True)
    tree_snapshot = pydot.graph_from_dot_data(dot.getvalue())
    tree_snapshot[0].write_png(str(idx) + "_" + criterion + "_snapshot.png")


def get_accuracy(y_expected, y_actual):
    return accuracy_score(y_expected, y_actual)*100


def print_results(criterion, y_test, y_pred):
    print (criterion +  " accuracy is " + str(get_accuracy(y_test, y_pred)) + "%")
    print("Confusion matrix = ")
    print(confusion_matrix(y_test, y_pred))


def generate_csv(file_name, arr):
    #arr = np.column_stack((col1, col2))
    np.savetxt(file_name, arr, delimiter=',', fmt='%s')

gini_accuracy_dataset = ['Dataset']
gini_accuracies = ['Accuracy']

gini_classifications_dataset = ['Dataset']
gini_classifications = ['Classifications']
gini_predictions = ['Predictions']

entropy_accuracy_dataset = ['Dataset']
entropy_accuracies = ['Accuracy']

entropy_classifications_dataset = ['Dataset']
entropy_classifications = ['Classifications']
entropy_predictions = ['Predictions']

for i in range(5):
    print("--------------------------------------------------")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    ###### GINI
    clf_gini = DecisionTreeClassifier(criterion = "gini")
    clf_gini.fit(X_train, y_train)
    generate_graph(clf_gini, i, "gini")

    y_pred_gini = clf_gini.predict(X_test)
    print_results("Gini", y_test, y_pred_gini)

    gini_accuracy_dataset.append(str(i+1))
    gini_accuracies.append(str(get_accuracy(y_test, y_pred_gini)) + "%")

    gini_classifications = np.append(gini_classifications, y_test)
    gini_predictions = np.append(gini_predictions, y_pred_gini)
    dataset_num = i + 1
    length = len(y_test)
    repeated_idx = np.repeat(str(dataset_num), length)
    gini_classifications_dataset = np.append(gini_classifications_dataset, repeated_idx)

    ###### ENTROPY
    clf_entropy = DecisionTreeClassifier(criterion = "entropy")
    clf_entropy.fit(X_train, y_train)
    generate_graph(clf_entropy, i, "entropy")

    y_pred_entropy = clf_entropy.predict(X_test)
    print_results("Entropy", y_test, y_pred_entropy)
    entropy_accuracy_dataset.append(str(i+1))
    entropy_accuracies.append(str(get_accuracy(y_test, y_pred_entropy)) + "%")

    entropy_classifications = np.append(entropy_classifications, y_test)
    entropy_predictions = np.append(entropy_predictions, y_pred_entropy)
    dataset_num = i + 1
    length = len(y_test)
    repeated_idx = np.repeat(str(dataset_num), length)
    entropy_classifications_dataset = np.append(entropy_classifications_dataset, repeated_idx)

generate_csv("g_9_DT__gini_accuracy.csv", np.column_stack((gini_accuracy_dataset, gini_accuracies)))
generate_csv("g_9_DT__entropy_accuracy.csv", np.column_stack((entropy_accuracy_dataset, entropy_accuracies)))
generate_csv("g_9_DT__gini_predictions.csv",
             np.column_stack((gini_classifications_dataset, gini_classifications, gini_predictions)))
generate_csv("g_9_DT__entropy_predictions.csv",
             np.column_stack((entropy_classifications_dataset, entropy_classifications, entropy_predictions)))
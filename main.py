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

X = balance_data.values[:, 1:-1]
Y = balance_data.values[:,-1]

yearID = X[:, 0]

Y = Y.reshape(-1,1)
Y = Y.tolist()

yearID = yearID.reshape(-1,1)

enc = OneHotEncoder(handle_unknown='ignore')

yearID = enc.fit_transform(yearID).toarray()

numerical_value = X[:, 1:]
numerical_value = np.append(numerical_value, yearID, axis=1)
numerical_value[np.isnan(numerical_value.tolist())] = 0

for i in range(5):
    print("--------------------------------------------------")
    X_train, X_test, y_train, y_test = train_test_split(numerical_value, Y, test_size = 0.2)

    clf_gini = DecisionTreeClassifier(criterion = "gini")
    clf_gini.fit(X_train, y_train)
    # dot_data = export_graphviz(clf_gini, out_file=None)
    # graph = export_graphviz.Source(dot_data)
    # graph.render("%i_gini" %i)

    dot_data = StringIO()
    export_graphviz(clf_gini, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_png("%i_gini.png"%i)


    y_pred_gini = clf_gini.predict(X_test)
    print ("Gini Accuracy is ", accuracy_score(y_test,y_pred_gini)*100)

    print("Confusion matrix = ")
    print(confusion_matrix(y_test, y_pred_gini))

    clf_entropy = DecisionTreeClassifier(criterion = "entropy")
    clf_entropy.fit(X_train, y_train)
    # dot_data = export_graphviz(clf_entropy, out_file=None)
    # graph = export_graphviz.Source(dot_data)
    #graph.render("%i_entropy" % i)
    dot_data = StringIO()
    export_graphviz(clf_entropy,
                    out_file=dot_data,
                    feature_names= feature,
                    class_names = ['nominated','elected'],
                    filled=True,
                    rounded=True,
                    special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_png("%i_entropy.png"%i)

    y_pred_entropy = clf_entropy.predict(X_test)
    print ("Entropy Accuracy is ", accuracy_score(y_test,y_pred_entropy)*100)
    print("Confusion matrix = ")
    print(confusion_matrix(y_test, y_pred_entropy))

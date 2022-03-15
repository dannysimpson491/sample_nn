# Neural Network project (3)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout

df = pd.read_csv("HR_comma_sep.csv", delimiter=',')
df.rename(columns={"sales": "department"}, inplace=True)

cat_features = ["department", "salary"]
df_final = pd.get_dummies(df, columns = cat_features, drop_first=True)

X = df_final.drop(["left"],axis=1).values
y = df_final["left"].values

num_features = X.shape[1]
num_output = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def make_classifier():
    
    classifier = Sequential()
    
    classifier.add(Dense(
        int((num_features+num_output)/2), 
        kernel_initializer = "uniform", 
        activation = "relu", 
        input_dim = num_features))
    
    classifier.add(Dropout(rate=0.5))
    
    classifier.add(Dense(
        num_output, 
        kernel_initializer = "uniform", 
        activation = "sigmoid"))
    
    classifier.compile(
        optimizer = "adam", 
        loss = "binary_crossentropy", 
        metrics = ["accuracy"])
    
    return classifier

classifier = KerasClassifier(build_fn = make_classifier,
                             batch_size=10,
                             nb_epoch=1)

parameters = {"batch_size": [2,4,8,16],
              "epochs": [2,3]}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv=2)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best parameters:", best_parameters, ", Accuracy =", best_accuracy)

"""
accuracies = cross_val_score(estimator = classifier, 
                             X = X_train, 
                             y = y_train, 
                             cv=10, 
                             n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.var()
print("Mean =", mean, ", Variance =", variance)

"""

#%% RUN THE OPTIMIZED MODEL

classifier.fit(X_train, 
               y_train, 
               batch_size = best_parameters["batch_size"],
               epochs = best_parameters["epochs"])

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", str(round(((cm[0,0]+cm[1,1])/len(y_test)*100),2)) + "%")

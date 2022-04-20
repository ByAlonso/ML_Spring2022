import dabl
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from mlxtend.evaluate import paired_ttest_kfold_cv
from preprocessing import *
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import sklearn.linear_model as lm
from sklearn import metrics

#1 explain that we have chosen to classify as a binary PAss / No pass, reason? It was easier and we thought it would be more accurate
# as the results are not close to a normal distribution

#2 we are gonna uuse h that are the numbers of hidden layers for the parameter in exercise 2

#ANN
cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
def ANN(X_train,y_train,X_test,y_test,cv_inner,cv_outer,aux=0,k=0, h = False):
    final_val = dict()
    aux = 0
    layers = [20]
    if h:
        for x in range(0, k):
            layers.append(20)
    model = MLPClassifier(hidden_layer_sizes=tuple(layers), activation='relu', solver='adam', random_state=1,
                         max_iter=500)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(pred)
    # Measure this fold's RMSE
    score = accuracy_score(y_test, pred)
    print('Accuracy: {:.2f}'.format(score))
    '''score = np.sqrt(metrics.mean_squared_error(pred, y_test))
    print(f"Fold score (RMSE): {score}")'''
    '''if not h:
        k += 1
        model.save('.\\regression_models_k\\model_k{}.h5'.format(k))
    else:
        aux += 1
        if aux % 2 == 0:
            k += 1
        model.save('.\\regression_models_h\\model_h{}_{}.h5'.format(aux, k))
    '''
    final_val[score] = model


    '''y_labeled = [np.argmax(x) for x in y_test]
    print(y_labeled)
    #this would be nice to get but I am done with trying
    fig = plot_confusion_matrix(model, X_test, pred)
    fig.figure_.suptitle("Confusion Matrix for Winequality Dataset")
    plt.show()'''
    return final_val

    '''pred_train = model.predict(X_train)
    scores = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

    pred_test = model.predict(X_test)
    scores2 = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))'''

def LogisticRegression(X_train,y_train,cv_inner,cv_outer):

    lambdas = np.power(10., range(-5, 9))
    scores = dict()
    y_train = [np.argmax(x) for x in y_train]

    for alpha in lambdas:
        model = lm.LogisticRegression(max_iter=200)
        model = model.fit(X, y)
        scores[(max(cross_val_score(model, X_train, y_train, scoring='accuracy',
                                      cv=cv_inner, n_jobs=-1)))] = [alpha, model]
    return {max(scores):scores[max(scores)]}


def Baseline(X_train,y_train,X_test,y_test, cv_inner, cv_outer):
    final_val = dict()
    model = DummyClassifier(strategy='most_frequent',random_state = 1)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)
    print('Accuracy: {:.2f}'.format(score))
    score = max(cross_val_score(model, X_train, y_train, scoring='accuracy',
                                  cv=cv_inner, n_jobs=-1))

    final_val[score] = model
    return final_val


k = 0
aux = 0
baseline = []
linear = []
ANN_ = []
h = True
for train, test in cv_outer.split(X):
    X_train = X.iloc[train]
    y_train = y_cat[train]
    X_test = X.iloc[test]
    y_test = y_cat[test]
    ANN_.append(ANN(X_train,y_train,X_test, y_test, cv_inner, cv_outer,aux, k, h))
    baseline.append(Baseline(X_train, y_train,X_test,y_test, cv_inner, cv_outer))
    linear.append(LogisticRegression(X_train, y_train, cv_inner, cv_outer))

    if not h:
        k += 1
    else:
        aux += 1
        if aux % 2 == 0:
            k += 1

ANN_ = dict(pair for d in ANN_ for pair in d.items())
baseline = dict(pair for d in baseline for pair in d.items())
linear = dict(pair for d in linear for pair in d.items())
print(ANN_)
print(baseline)
print(linear)
def estimator(X,y,clf1,clf2):
    t, p = paired_ttest_kfold_cv(estimator1=clf1,
                                  estimator2=clf2,
                                  X=X, y=y,
                                  random_seed=1)

    print('t statistic: %.3f' % t)
    print('p value: %.3f' % p)


min_ANN = ANN_[max(ANN_)]
min_linear= linear[max(linear)]
min_baseline = baseline[max(baseline)]
y_cat = pd.DataFrame([np.argmax(x) for x in y_cat])
print("ANN vs Logistic")
estimator(X,y_cat,min_ANN,min_linear[1])
print("Logistic vs Baseline")
estimator(X,y_cat,min_linear[1],min_baseline)
print("ANN vs Baseline")
estimator(X,y_cat,min_ANN,min_baseline)
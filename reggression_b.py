import dabl as dabl
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from preprocessing import *
from keras.layers import Dense
from keras import Sequential
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge
from mlxtend.evaluate import paired_ttest_kfold_cv
from sklearn.neural_network import MLPRegressor


def predict(path):
    model = keras.models.load_model(path)
    test_value = pd.DataFrame(np.array([[1, 1.0309000226943976, 1.30920583242787, 1.5395279876548638,
                                         -0.17151424526772402, 0.6932502514632959, -0.543136405254532,
                                         -0.9969257236476314, -0.3707562300992627, 0.07337676810107183, 1, 0, 0, 0, 0,
                                         0, 0, 0, 0, 1, 1, 0, 0, 0],  # 11
                                        [1, 0.20997472854879742, -0.453604510405507, -0.2787680341613449,
                                         -0.17151424526772402, -1.0077683239608213, -0.543136405254532,
                                         -0.21833992984821768, -0.3707562300992627, 3.736570806377657, 0, 0, 1, 0, 0, 0,
                                         0, 1, 0, 0, 0, 0, 0, 1],  # 13
                                        [1, 0.20997472854879742, 1.30920583242787, -0.2787680341613449,
                                         -0.17151424526772402, 0.6932502514632959, -0.543136405254532,
                                         0.560245863951196, 1.0121218925123545, -0.35758723640205586, 0, 0, 1, 0, 0, 0,
                                         0, 1, 0, 0, 1, 0, 0, 0],  # 14
                                        [1, 0.20997472854879742, -0.453604510405507, -0.2787680341613449,
                                         0.7798767391660624, -1.0077683239608213, 3.78196208566806, 2.1174174515500237,
                                         0.32068283120654595, -0.7885512409051836, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                                         0, 0]  # 16
                                        ]))
    print(test_value)
    print(model.predict(test_value))
cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
def ANN(X_train,y_train,X_test,y_test,cv_inner,cv_outer,aux=0,k=0, h = False):
    final_val = dict()
    model = Sequential()
    model.add(Dense(40, input_dim=X.shape[1], activation='relu'))
    if h:
        for x in range(0, k):
            model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=0,epochs=200)

    pred = model.predict(X_test)

    model.evaluate(X_test, y_test)

    # Measure this fold's RMSE
    score = np.sqrt(metrics.mean_squared_error(pred, y_test))
    print(f"Fold score (RMSE): {score}")
    if not h:
        k += 1
        model.save('.\\regression_models_k\\model_k{}.h5'.format(k))
    else:
        aux += 1
        if aux % 2 == 0:
            k += 1
        model.save('.\\regression_models_h\\model_h{}_{}.h5'.format(aux, k))
    final_val[score] = model
    return final_val

def ANN2(X_train,y_train,X_test,y_test,cv_inner,cv_outer,aux=0,k=0, h = False):
    final_val = dict()
    aux = 0
    layers = [20]
    if h:
        for x in range(0, k):
            layers.append(20)
    model = MLPRegressor(hidden_layer_sizes =tuple(layers),activation='relu', solver='adam',random_state=1, max_iter=500)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    # Measure this fold's RMSE
    score = np.sqrt(metrics.mean_squared_error(pred, y_test))
    print(f"Fold score (RMSE): {score}")
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
    return final_val

def Baseline(X_train,y_train,X_test,y_test,cv_inner,cv_outer):
    final_val = dict()
    model = LinearRegression()
    model.fit(X_train, y_train)
    f = lambda x: np.sqrt(abs(x))
    score = min(f(cross_val_score(model, X_test, y_test, scoring='neg_mean_squared_error',
                         cv=cv_inner, n_jobs=-1)))

    final_val[score] = model
    return final_val

def LinearRegressionn(X_train,y_train,cv_inner,cv_outer):
    lambdas = np.power(10., range(-5, 9))
    scores = dict()
    for alpha in lambdas:
        model = Ridge(alpha=alpha)
        model.fit(X_train,y_train)
        f = lambda x: np.sqrt(abs(x))
        scores[(min(f(cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error',
                                      cv=cv_inner, n_jobs=-1))))] = [alpha, model]
    return {min(scores):scores[min(scores)]}



k = 0
aux = 0
baseline = []
linear = []
ANN_ = []
h = True
for train, test in cv_outer.split(X):
    X_train = X.iloc[train]
    y_train = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]
    ANN_.append(ANN2(X_train,y_train,X_test, y_test, cv_inner, cv_outer,aux, k, h))
    baseline.append(Baseline(X_train, y_train,X_test,y_test, cv_inner, cv_outer))
    linear.append(LinearRegressionn(X_train, y_train, cv_inner, cv_outer))

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


min_ANN = ANN_[min(ANN_)]
min_linear= linear[min(linear)]
min_baseline = baseline[min(baseline)]
print("ANN vs Linear")
estimator(X,y,min_ANN,min_linear[1])
print("Linear vs Baseline")
estimator(X,y,min_linear[1],min_baseline)
print("ANN vs Baseline")
estimator(X,y,min_ANN,min_baseline)

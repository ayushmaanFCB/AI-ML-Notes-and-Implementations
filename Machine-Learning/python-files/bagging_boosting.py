from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=101)


# BAGGING
bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),
                        n_estimators=10, random_state=101)
bag.fit(xtrain, ytrain)
preds = bag.predict(xtest)
print(preds[0:10])


# ADABOOST
abo = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),
                         n_estimators=10, random_state=101)
abo.fit(xtrain, ytrain)
preds = abo.predict(xtest)
print(preds[0:10])


# GRADIENT BOOST
gbo = GradientBoostingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),
                                 n_estimators=10, random_state=101)
gbo.fit(xtrain, ytrain)
preds = gbo.predict(xtest)
print(preds[0:10])

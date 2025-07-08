from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle



iris = load_iris()
X, y = iris.data, iris.target


clf = RandomForestClassifier()
clf.fit(X, y)


with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)
    
print("model is trained !!!")
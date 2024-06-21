## Decision Trees
 
#Decision Trees are used in both classification and regression tasks
#They make decisions based on feature values, leading to a tree-like structure 
 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
 
#Load the iris dataset 
iris = load_iris()
X = iris.data
y = iris.target
 
# Train a decision tree classifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X, y)

# Display the decision tree rules
tree_rules = export_text(tree_clf, feature_names=iris.feature_names)
print("Decision Tree Rules:\n", tree_rules)
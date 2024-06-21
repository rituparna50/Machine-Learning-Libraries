## Decision Trees
 
#Decision Trees are used in both classification and regression tasks
#They make decisions based on feature values, leading to a tree-like structure 


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
#Load the iris dataset 
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Train a decision tree classifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X, y)

# Make predictions on the test set
y_pred = tree_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the decision tree rules
tree_rules = export_text(tree_clf, feature_names=iris.feature_names)
print("Decision Tree Rules:\n", tree_rules)

# Save accuracy to the accuracies.md file
with open("/Users/stanford/Desktop/FDL Prep/Accuracies.md", "a") as f:
    f.write("# Decision Tree\n")
    f.write(f"Accuracy: {accuracy}\n\n")


# Display the decision tree rules
tree_rules = export_text(tree_clf, feature_names=iris.feature_names)
print("Decision Tree Rules:\n", tree_rules)



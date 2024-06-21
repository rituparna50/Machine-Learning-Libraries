## Support Vector Machines (SVM)

'''Supervised machine learning 
    
    These are powerful algorithms for classification and regression tasks. They find the optimal hyperplane that 
    best separates data points in differnet classes.Best known for their effectiveness in classification problems. 
    
SVM aims to find the best boundary or hyperplane that separates different classes of data points with the maximum margin. 
In otehr words, SVM tries to find the line (in 2D), plane (in 3D), or hyperplane (in higher dimensions)
that best separates the classes of data

Support Vectors:

Support vectors are the data points that are closest to the hyperplane.
These points are critical because they define the position and orientation of the hyperplane.
The hyperplane is determined by these points, and not by the points farther away.
'''

''' In the Iris dataset, if we are trying to classify the species of the flower based on sepal and petal measurements,
a linear SVM would find the line that best separates the different species.'''
     


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Load the iris data 
iris = datasets.load_iris()
X = iris.data
y = iris.target

#Split the data into traininf and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=None, random_state=42)

#Train a Support vector Machine classifier 
svm_clf = SVC(kernel='linear', C=1)
svm_clf.fit(X_train, y_train)

#Make predictions on the test set 
y_pred = svm_clf.predict(X_test)

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Support Vector machines accuracy: {accuracy}")
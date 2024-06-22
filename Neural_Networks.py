# Neural Networks 

'''Neural Networks inspired by the human brain, are powerful moddels capable of learning complex patterns. 
Deep learning, a subset of neural networks, involvws multiple layers.'''

#importing necessary libraries 
import tensorflow as tf     # TensorFlow is an open-source library for numerical computation and ML

from tensorflow.keras.models import Sequential
    #Sequential is a linear stack of layers in Keras, a high level API for building and training DL models 

from tensorflow.keras.layers import Dense        
    #Dense is a layer type in neural networks where each neuron is connected to every neuron in the previous layer 


# Loading the MNIST dataset 
mnist = tf.keras.datasets.mnist    
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Flatten the images (from 28x28 to 784) since the Dense layer expects a 1D array of inputs 
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)


# Building the Neural Network 
model = Sequential() 
'''This initializes the neural network model as a sequence of layers'''

model.add(Dense(128, input_shape=(784,), activation='relu'))
'''Dense(128) - Adds a dense (fully connected) layer with 128 neurons
    The input layer expects 784 features (input dimensions)'''
    
model.add(Dense(10, activation='softmax'))
'''Adds another dense layer with 3 neurons.
    Uses the softmax activation function, which is typically used in the output layer for classification problem. 
    It converts the output into probabilities'''


## Compiling the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
'''loss='sparse_categorical_function -->  This is the loss function used for multi-class classification problems.
where the labels are integers. It measures how well the model's predictions match the true labels.
Optimizer = 'adam' --> Thus optimizer adjusts the weights of the neural network t minimize the loss funtion.
Adam is a popular and efficient optimizer.
Accuracy - Used as a metric to evaluate the model's poerformance. '''


# Training the model 
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
'''X_train - Thge training data (features)
   y_train - The training labels (target values) 
   epochs - the number of times the training algorithm will work through the entire training dataset
   batch_size - The number of samples processed before the model is updated
   verbose = 2 - Provides detailed output of the training process'''

# Evaluate the model 
_, accuracy_nn = model.evaluate(X_test, y_test)
print(f"Neural Networks Accuracy: {accuracy_nn}")

'''# Save accuracy to the accuracies.md file 
with open('/Users/stanford/Desktop/FDL Prep/Accuracies.md', 'a') as f:
    f.write(f"\n# Neural Network\nAccuracy: {accuracy_nn}\n"
'''

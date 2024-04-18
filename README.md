# neural-network-challenge-1
## Overview
Herein the repository is a recommendation system to assist with student loan options. It utilizes ML techniques on students, and loans to provide a personalized recommendation for an individual student.

## Pt 1
### Step 1
Read the `student-loans.csv` file into a Pandas DataFrame. Review the DataFrame, looking for columns that could eventually define your features and target variables. Check for the credit worthiness

### Step 2
Creating the features (`X`) and target (`y`) datasets. The target dataset is defined by the preprocessed DataFrame column “credit_ranking”. The remaining columns are the features.

### Step 3
Split the features and target sets into training and testing datasets.

### Step 4
 Use scikit-learn's `StandardScaler` to scale the features data. I used the fit_transform function.

## Pt 2
### Step 1
Creating a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.

Number of Inputs (Features):
The number of input features is determined by the shape of (X_train). This value represents the number of input nodes in the neural network's input layer, corresponding to the number of features used for prediction.
Number of Hidden Nodes:
Two hidden layers are added to the neural network model, each with a predefined number of hidden nodes (hidden_nodes_layer_a = 10 and hidden_nodes_layer_b = 5). These hidden nodes enable the nueral network to learn complex relationships within the data.
Number of Neurons in the Output Layer:
The output layer of the neural network contains a single neuron, representing the probability of loan repayment success. The activation function used in the output layer is sigmoid for binary classification tasks.

### Step 2
Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.

The model.compile() function is used to compile the neural network model.
The binary cross-entropy loss function is chosen, which is commonly used for binary classification tasks. This loss function measures the difference between the actual and predicted probabilities of loan repayment success.
The Adam optimizer is selected as the optimization algorithm. Adam is an efficient optimization algorithm that adapts learning rates for each parameter and is widely used in neural network training.
The evaluation metric chosen is accuracy, which measures the proportion of correctly classified instances out of the total number of instances.
Model Training:
The model.fit() function is used to train the compiled neural network model using the training data (X_train_scaled and y_train).
The epochs parameter specifies the number of training epochs, which determines the number of times the entire training dataset is passed through the network for learning.
In this code section, the model is trained for 50 epochs, meaning that the training process iterates over the entire training dataset 50 times.
The training process updates the model's parameters (weights and biases) iteratively to minimize the loss function and improve the model's performance on the training data.

### Step 3
Evaluate the model using the test data to determine the model’s loss and accuracy.

### Step 4
Save and export our model to a keras file, and name the file `student_loans.keras`.

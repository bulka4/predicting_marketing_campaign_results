import numpy as np
import tensorflow as tf


# model which takes as an input both categorical and continuous x variables. At first one neural network takes as an input
# encoded categorical x variables, then output of that network is concatenated with the continuous x variables and this is
# an input to the next neural network

class Model(tf.keras.layers.Layer):
    def __init__(
        self,
        categorical_no_neurons, # list with number of neurons for each layer of categorical input
        categorical_activations, # list with activation functions for each layer of categorical input
        continuous_no_neurons, # list with number of neurons for each layer of continuous input
        continuous_activations, # list with activation functions for each layer of continuous input
        output_shape
    ):
        super().__init__()
        
        # layers for categorical input
        self.categoricalLayers = []
        for no_neurons, activation in zip(categorical_no_neurons, categorical_activations):
            self.categoricalLayers.append(tf.keras.layers.Dense(no_neurons, activation = activation))
        
        # layers for continuous input
        self.continuousLayers = []
        for no_neurons, activation in zip(continuous_no_neurons, continuous_activations):
            self.continuousLayers.append(tf.keras.layers.Dense(no_neurons, activation = activation))
            
        self.continuousLayers.append(tf.keras.layers.Dense(output_shape, activation = 'softmax'))
        
    # @tf.function
    def call(self, x_categorical, x_continuous):
        """
        x_categorical.shape = (batch_size, no_categorical_features)
        x_continuous.shape = (batch_size, no_continuous_features)
        
        x_categorical is an input with categorical variables which are already encoded
        with use of for example one hot encoding
        
        x_continuous is an input with continuous variables
        """
        # make sure that data types are correct
        x_categorical = tf.cast(x_categorical, tf.float32)
        x_continuous = tf.cast(x_continuous, tf.float32)
        
        for layer in self.categoricalLayers:
            x_categorical = layer(x_categorical)
            
        # x_categorical.shape = (batch_size, categorical_no_neurons[-1])
        
        # concatenate output from categoricalLayers with an input with continuous variables
        x = tf.concat([x_categorical, x_continuous], axis = 1)
        
        # x.shape = (batch_size, categorical_no_neurons[-1] + no_continuous_features)
        
        for layer in self.continuousLayers:
            x = layer(x)
            
        return x
    

# @tf.function
def train_step(
    x_categorical,
    x_continuous,
    y,
    model,
    loss_function, 
    optimizer
):
    
    # make sure that the types are correct
    x_categorical = tf.cast(x_categorical, tf.float32)
    x_continuous = tf.cast(x_continuous, tf.float32)
    y = tf.cast(y, tf.float32)
    
    batch_loss = 0
    
    with tf.GradientTape() as tape:
        prediction = model(x_categorical, x_continuous)
        loss = loss_function(y, prediction)
        
        batch_loss += loss
            
    variables = model.trainable_variables
    gradients = tape.gradient(batch_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss


# model which take an average from predictions from all models trained on different subsets of the main dataset
def finalModel(models, x_categorical, x_continuous):
    # models = [models[i] for i in [0,2]]
    
    preds = []
    for model in models:
        pred = model(x_categorical, x_continuous).numpy()
        preds.append(pred)
        
    return np.mean(preds, axis = 0)


# predictions using average from different models
def evaluateFinalModel(models, x_categorical, x_continuous, y, threshold):
    pred = finalModel(models, x_categorical, x_continuous)
    pred = np.array([0 if p[0] > threshold else 1 for p in pred])
    
    # calculating true positives, true negatives, false negatives and false positives
    tp = len(np.where((y == pred) & (pred == 1))[0])
    tn = len(np.where((y == pred) & (pred == 0))[0])
    fp = len(np.where((y != pred) & (pred == 1))[0])
    fn = len(np.where((y != pred) & (pred == 0))[0])
    
    return tp, tn, fp, fn



# predictions from 1 model
def evaluateSingleModel(model, x_categorical, x_continuous, y, threshold):
    pred = model(x_categorical, x_continuous)
    pred = np.array([0 if p[0] > threshold else 1 for p in pred])
    
    # calculating true positives, true negatives, false negatives and false positives
    tp = len(np.where((y == pred) & (pred == 1))[0])
    tn = len(np.where((y == pred) & (pred == 0))[0])
    fp = len(np.where((y != pred) & (pred == 1))[0])
    fn = len(np.where((y != pred) & (pred == 0))[0])
    
    return tp, tn, fp, fn
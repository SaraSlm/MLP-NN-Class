# MLP-NN-Class
Multi layer perceptron (MLP) is a supplement of feed forward neural network. It consists of three types of layers; the input layer, output layer and hidden layer. The input layer receives the input signal to be processed. The required task such as prediction and classification is performed by the output layer. An arbitrary number of hidden layers that are placed in between the input and output layer are the true computational engine of the MLP. 

#Dependencies
  . Python3
  . numpy
  . pandas
  . seaborn
  . operator
  . matplotlib

# Structure
  1. load your data as numpy array with np.load
  2. if your data is sorted, you have to shuffle it.you can do it using unison_shuffle
  3. split your data to train, test, validation with train_test_validation_split and pass your data and percent of test and validation
  4. create your network by calling Neural_Network class with two parametres:train data and learning rate
  5. add layers with add function in Neural_Network class and pass neuron_numbers,weights(if there is a layer before this layer pass the output of that layer otherwise pass empty array),activation_function(it supported softmax,relu,sigmoid),has_bias:True or False
  6. start train model by calling train function and pass values of train and validation data(just remember that the labels of data should be in one hot format) and epochs and batch_size
  7. you can also plot your results with plot function
  
 this code supported plot confusion matrix and calculate f1-score,recall,precision just by little changes

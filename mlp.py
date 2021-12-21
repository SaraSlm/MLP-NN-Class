import numpy as np
import operator
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def unison_shuffle(a, b):
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]


def train_test_validation_split(data, test_percent, validation_percent):
    test_data = data[0:round(len(data) * test_percent)]
    validation_data = data[round(len(data) * test_percent + 1):round(
        len(data) * test_percent + 1 + len(data) * validation_percent)]
    train_data = data[round(len(data) * test_percent + 1 + len(data) * validation_percent + 1):]
    return train_data, validation_data, test_data


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return x * (x > 0)


def softmax(x):
    B = np.exp(x - np.max(x, axis=-1, keepdims=True))
    C = np.sum(B, axis=-1, keepdims=True)
    return B / C


def sigmoid_deritive(x):
    return x * (1 - x)


def relu_deritive(x):
    return (x > 0) * 1


def softmax_deritive(x):
    return x * (1 - x)


def CE_loss(output, label):
    return -((label * np.log(output)).sum(axis=1)).mean()


def plot(title, data1, label_data1, data2, label_data2):
    plt.plot(data1, 'g', label=label_data1)
    plt.plot(data2, 'b', label=label_data2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.legend()
    plt.show()


class Neural_Network:
    def __init__(self, n_in, learning_rate):
        self.n_x = n_in
        self.layers_weight = []
        self.bias = []
        self.functions = []
        self.learning_rate = learning_rate
        self.losses = []
        self.accuracy = []
        self.validation_losses = []
        self.validation_accuracy = []

    def add(self, n_nouron, weights, activationa_function, has_bias):
        self.functions.append(activationa_function)
        if (len(weights) > 0):
            if (has_bias):
                self.layers_weight.append(
                    [np.random.randn(n_nouron, weights.shape[0] + 1) / (n_nouron + weights.shape[0] + 1)])
                this_layer_weight = np.random.randn(n_nouron, weights.shape[0] + 1) / (n_nouron + weights.shape[0] + 1)
            else:
                self.layers_weight.append([np.random.randn(n_nouron, weights.shape[0]) / (n_nouron + weights.shape[0])])
                this_layer_weight = np.random.randn(n_nouron, weights.shape[0]) / (n_nouron + weights.shape[0])
        else:
            if (has_bias):
                self.layers_weight.append(
                    [np.random.randn(n_nouron, len(self.n_x[1]) + 1) / (n_nouron + len(self.n_x[1]) + 1)])
                this_layer_weight = np.random.randn(n_nouron, len(self.n_x[1]) + 1) / (n_nouron + len(self.n_x[1]) + 1)
            else:
                self.layers_weight.append([np.random.randn(n_nouron, len(self.n_x[1])) / (n_nouron + len(self.n_x[1]))])
                this_layer_weight = np.random.randn(n_nouron, len(self.n_x[1])) / (n_nouron + len(self.n_x[1]))
        return this_layer_weight

    def activation_functions(self, x, activation_function):
        if (activation_function == "sigmoid"):
            return sigmoid(x)
        elif (activation_function == "softmax"):
            return softmax(x)
        elif (activation_function == "relu"):
            return relu(x)

    def derivative_functions(self, x, activation_function):
        if (activation_function == "sigmoid"):
            return sigmoid_deritive(x)
        elif (activation_function == "softmax"):
            return softmax_deritive(x)
        elif (activation_function == "relu"):
            return relu_deritive(x)

    def forward(self, X):
        outputs = [X.copy()]
        for i in range(len(self.layers_weight)):
            input_arr = np.concatenate((X, np.ones((X.shape[0], 1), dtype=np.float128)), axis=1)
            weight_bias = np.array(self.layers_weight[i])[0].T
            self.Z1 = np.dot(input_arr, weight_bias)
            X = self.activation_functions(self.Z1, self.functions[i])
            outputs.append(X)
        return outputs

    def back_prop(self, outputs, Y):
        errors = [np.subtract(Y, outputs[-1])]
        for i in range(len(self.layers_weight) - 1):
            delta = np.dot(errors[i], np.array(self.layers_weight[-1 - i])[0][:, :-1])
            delta = delta * self.derivative_functions(outputs[-2 - i], self.functions[-2 - i])
            errors.append(delta)

        for i in range(len(self.layers_weight)):
            delta_w = np.dot(errors[i].T, np.concatenate((outputs[-2 - i],
                                                          np.ones((outputs[-2 - i].shape[0], 1))
                                                          ), axis=1)) * self.learning_rate
            delta_w = delta_w / Y.shape[0]
            self.layers_weight[-1 - i][0] += delta_w

        return self.layers_weight

    def train(self, X, Y, validation_x, validation_y, epochs, batch_size=-1, ):
        for e in range(epochs):
            if batch_size == -1:
                outputs = self.forward(X)
                self.back_prop(outputs, Y)
                train_loss = CE_loss(outputs[-1], Y)
                train_acc = np.mean(outputs[-1].argmax(axis=-1) == Y.argmax(axis=-1))
                print("epoch-----------------------------------------------------------------------", e)
                print(train_loss, train_acc)
                self.losses.append(train_loss)
                self.accuracy.append(train_acc)
            else:
                for j in range(0, X.shape[0], batch_size):
                    X_batch = X[j: min(j + batch_size, X.shape[0])]
                    Y_batch = Y[j: min(j + batch_size, X.shape[0])]

                    outputs = self.forward(X_batch)
                    self.back_prop(outputs, Y_batch)

                outputs = self.forward(X)
                loss = CE_loss(outputs[-1], Y)
                acc = np.mean(outputs[-1].argmax(axis=-1) == Y.argmax(axis=-1))
                self.losses.append(loss)
                self.accuracy.append(acc)
                print("epoch-----------------------------------------------------------------------", e)
                print('loss:', loss, 'acc:', acc)
            if (len(validation_x) > 0):
                validation_output = self.forward(validation_x)
                validation_loss = CE_loss(validation_output[-1], validation_y)
                validation_acc = np.mean(validation_output[-1].argmax(axis=-1) == validation_y.argmax(axis=-1))
                print('validation_loss:', validation_loss, 'validation_acc:', validation_acc)
                self.validation_losses.append(validation_loss)
                self.validation_accuracy.append(validation_acc)
        return self.losses, self.accuracy, self.validation_losses, self.validation_accuracy

    def predict(self, inp):
        output = self.forward(inp)[-1]
        outputs = [dict(enumerate(output[i])) for i in range(len(output))]
        output_classes = [max(output.items(), key=operator.itemgetter(1))[0] for output in outputs]
        out_probs = [outputs[out_class] for out_class in output_classes]
        return output_classes, out_probs

    def predict_class(self, inp):
        output = self.forward(inp)[-1]
        output = dict(enumerate(output[0]))
        out_class = max(output.items(), key=operator.itemgetter(1))[0]
        out_prob = output[out_class]
        return out_class, out_prob


x_input_data = np.load('path_to_x_input_data .npy')
y_input_data = np.load('path_to_y_input_data .npy')
x_input_data, y_input_data = unison_shuffle(x_input_data, y_input_data)

train_data_x, validation_data_x, test_data_x = train_test_validation_split(x_input_data, 0.2, 0.2)
train_data_y, validation_data_y, test_data_y = train_test_validation_split(y_input_data, 0.2, 0.2)

nn = Neural_Network(train_data_x, 0.1)
weight = nn.add(200, [], 'relu', True)
nn.add(10, weight, 'softmax', True)
# one hot train y
train_data_y = np.array(train_data_y)
categorical_train_y = np.zeros((train_data_y.size, train_data_y.max() + 1))
categorical_train_y[np.arange(train_data_y.size), train_data_y] = 1

validation_data_y = np.array(validation_data_y)
categorical_validation_y = np.zeros((validation_data_y.size, validation_data_y.max() + 1))
categorical_validation_y[np.arange(validation_data_y.size), validation_data_y] = 1
train_loss, train_acc, validation_loss, validation_acc = nn.train(train_data_x, categorical_train_y,
                                                                  validation_x=validation_data_x,
                                                                  validation_y=categorical_validation_y, epochs=5,
                                                                  batch_size=32)

plot("training and validation loss", train_loss, "train_loss", validation_loss, "validation_loss")
plot("training and validation accuracy", train_acc, "train_acc", validation_acc, "validation_acc")

test_out = nn.predict(test_data_x)

confusion_matrix = np.ones((10, 10))
confusion_matrix[0:10, 0:10] = 0
for i in range(len(test_data_y)):
    confusion_matrix[test_out[0][i]][test_data_y[i]] += 1

# put class names instead ABCFGHVWLP
df_cm = pd.DataFrame(confusion_matrix, index=[i for i in "ABCFGHVWLP"],
                     columns=[i for i in "ABCFGHVWLP"])
df_cm = df_cm.astype(int)

plt.figure(figsize=(10, 10))
sn.heatmap(df_cm, annot=True, fmt="d")
plt.show()
true_pos = np.diag(df_cm)
false_pos = np.sum(df_cm, axis=0) - true_pos
false_neg = np.sum(df_cm, axis=1) - true_pos

precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
F1 = 2 * (precision * recall) / (precision + recall)

print("precision", precision)
print("recall", recall)
print("f1score", F1)


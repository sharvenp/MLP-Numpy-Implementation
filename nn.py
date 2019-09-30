
# MLP Neural Net Implementation in using Numpy

import numpy as np
import mnist # The Handwritten Digits Dataset
import json

class NeuralNet:

    def __init__(self, structure, random_init_bound=1, model_path=''):

        self.structure = structure
        self.n = len(self.structure)

        if model_path == '': # Init New Model
            self._construct_model(random_init_bound)
        else: # Load Model
            self.load(model_path)

    def _construct_model(self, random_bound):

        # Init the Weights, Biases and Structure of NN

        self.weights, self.biases = [], []

        for i in range(self.n):
            neurons, activation = self.structure[i]
            if activation == '*': # Its the input layer
                self.weights.append('*')
                self.biases.append('*')
            else:
                self.weights.append( ((2*random_bound) * np.random.random((self.structure[i - 1][0], neurons))) - random_bound)
                self.biases.append( (random_bound * np.random.random((1, neurons))) - (random_bound/2) ) 

    def _cost_function(self, method, y, y_hat, deriv):
        
        # The Loss Function of the NN

        if method == 'MSE': # Mean Squared Error
            if deriv:
                return (1/y_hat.shape[0]) * (y_hat - y)
            else:
                return np.sum((y - y_hat)**2) / (2 * y_hat.shape[0])

    def _activation_function(self, function, deriv, x):

        # The activation function

        if function == 'sigmoid':
            
            if deriv:
                ex = self._activation_function('sigmoid', False, x)
                return ex*(1 - ex) 
            else:
                return 1/(1 + np.exp(-x))

        elif function == 'relu':

            if deriv:
                x[x<=0] = 0
                x[x>0] = 1
                return x
            else:
                return np.maximum(0, x)

        elif function == 'softmax':

            if deriv:
                pass
            else:
                shiftx = x - np.max(x)
                exps = np.exp(shiftx)
                return exps / np.sum(exps)

    def _back_prop(self, loss_method, Y_Data, layers, activations, lr):
        
        # Back Propagation

        curr_delta = self._cost_function(loss_method, Y_Data, activations[-1], True)

        deltas = [curr_delta]

        for i in range(2, self.n):

            k = self.n - i

            curr_layer = layers[k]
            curr_activation = activations[k]

            w = self.weights[k + 1]
            gradient = np.dot(w, curr_delta.T)
            curr_delta = self._activation_function(self.structure[k][1], True, curr_layer) * gradient.T

            deltas.append(curr_delta)

        for i in range(len(deltas)):
            
            k = self.n - i - 1
            dw = np.dot(deltas[i].T, activations[k - 1])
            db = np.sum(deltas[i], axis=0, keepdims=True)
            self.weights[k] -= dw.T * lr
            self.biases[k] -= db * lr

    def load(self, load_path):

        # Load a json file with model data

        with open(load_path) as model_data:
            data = json.load(model_data)
            w, b = data[0], data[1]

            self.weights, self.biases = [], [] 

            for i in range(len(w)):
                if i > 0:
                    self.weights.append(np.array(w[i]))
                    self.biases.append((np.array(b[i])))
                else:
                    self.weights.append('*')
                    self.biases.append('*')

            print("Loaded Successfully from {}.".format(load_path))

    def save(self, save_path):

        # Save this model to <save_path> as a json file

        with open(save_path, 'w') as saved_model:

            w, b = [], []

            for i in range(len(self.weights)):

                if i > 0:
                    w.append(self.weights[i].tolist())
                    b.append(self.biases[i].tolist())
                else:
                    w.append('*')
                    b.append('*')

            json.dump([w, b], saved_model)
            print("Saved Successfully to {}.".format(save_path))

    def fit(self, X_Data, Y_Data, loss_method, lr, lr_decay, epochs, batch_size, print_mode=0):
        
        # Train the NN with labels <X_Data, Y_Data>.

        if print_mode:
            print(f"Training for {epochs} epochs:")

        m = X_Data.shape[0] // batch_size
        prev_loss = 0

        curr_lr = lr

        for e in range(epochs):

            p = np.random.permutation(X_Data.shape[0])

            X_Data = X_Data[p, :]
            Y_Data = Y_Data[p, :]

            batch_loss = 0

            for i in range(m):

                X = X_Data[i*batch_size : (i + 1)*batch_size, :]
                Y = Y_Data[i*batch_size : (i + 1)*batch_size, :]

                layers, activations = self.forward_prop(X)

                batch_loss += self._cost_function(loss_method, Y, activations[-1], False)
                self._back_prop(loss_method, Y, layers, activations, curr_lr)

            # Decay Learning Rate
            curr_lr = lr / (1 + lr_decay * e)

            if print_mode:
                loss = batch_loss/m
                print("Epoch: {:0>3d} - Loss: {:.10f} - delta: {:+.5f} - lr: {:.6f}".format(e, loss, loss - prev_loss, curr_lr))
                prev_loss = loss

    def test(self, X_Data, Y_Data, loss_method):

        # Test the NN with X_Data and Expected Y_Data

        print(60*"=")
        print("Testing:")

        output = self.forward_prop(X_Data)[1][-1]
        samples = X_Data.shape[0]
        correct = 0

        for i in range(len(output)):
            if np.argmax(output[i]) == np.argmax(Y_Data[i]):
                correct += 1

        print("Accuracy: {:2d}/{:5d} - {:.2f} %".format(correct, samples, ((correct/samples)*100)))
        print("Loss: {:.10f}".format(self._cost_function(loss_method, Y_Data, output, False)))
        print(60*"=")        

    def forward_prop(self, x):
        
        # Forward Propagate

        curr_layer = x

        layers = [curr_layer]
        activations = [curr_layer]

        for i in range(1, self.n):
            w = self.weights[i]
            b = self.biases[i]
            curr_layer = np.dot(curr_layer, w) + b
            layers.append(curr_layer)
            activation = self._activation_function(self.structure[i][1], False, curr_layer)
            activations.append(activation)

        return layers, activations

    def evaluate(self, x):

        output = self.forward_prop(x)[1][-1]
        return np.argmax(output)
        
def main():

    # Numpy Stuff
    # np.random.seed(1)
    # np.set_printoptions(threshold=np.inf)

    # Toy Data for testing

    # X_Data = np.array([[0, 0, 0],
    #                    [0, 0, 1],
    #                    [0, 1, 0],
    #                    [0, 1, 1],
    #                    [1, 0, 0],
    #                    [1, 0, 1],
    #                    [1, 1, 0],
    #                    [1, 1, 1]])

    # Y_Data = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
    #                    [0, 1, 0, 0, 0, 0, 0, 0],
    #                    [0, 0, 1, 0, 0, 0, 0, 0],
    #                    [0, 0, 0, 1, 0, 0, 0, 0],
    #                    [0, 0, 0, 0, 1, 0, 0, 0],
    #                    [0, 0, 0, 0, 0, 1, 0, 0],
    #                    [0, 0, 0, 0, 0, 0, 1, 0],
    #                    [0, 0, 0, 0, 0, 0, 0, 1]])

    structure = [(784, '*'), (38, 'sigmoid'), (10, 'sigmoid')]
    nn = NeuralNet(structure, random_init_bound=0.05)

    # nn.load('models/Neon.json') # Load a model

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    X_Test_Data = (test_images.reshape(test_images.shape[0], 784))/255.0
    Y_Test_Data = np.zeros((10,10000))

    X_Train_Data = (train_images.reshape(train_images.shape[0], 784))/255.0
    Y_Train_Data = np.zeros((10,60000))
    
    for i in range(Y_Train_Data.shape[1]):
        Y_Train_Data[train_labels[i]][i] = 1.0

    for i in range(Y_Test_Data.shape[1]):
        Y_Test_Data[test_labels[i]][i] = 1.0

    nn.fit(X_Train_Data, Y_Train_Data.T, 'MSE', 0.01, 0.05, 50, 50, print_mode=1)
    nn.test(X_Test_Data, Y_Test_Data.T, 'MSE')

    nn.save('models/Sodium.json') # Save model

if __name__ == "__main__":
    main()
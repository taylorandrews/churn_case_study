from __future__ import division
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import theano
import itertools
from eda import load_and_clean_data
from sklearn.cross_validation import train_test_split

def load_data():
    ''' loads and shapes data '''
    dfn, dfc = load_and_clean_data()
    dfn.pop('last_trip_date')
    dfn.pop('signup_date')
    dfc.pop('last_trip_date')
    dfc.pop('signup_date')
    y_n = dfn.pop('churn').values
    X_n = dfn.values
    y_c = dfc.pop('churn').values
    X_c = dfc.values
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_n, y_n, test_size=0.3)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.3)
    theano.config.floatX = 'float32'
    X_train = X_train_n.astype(theano.config.floatX)
    X_test = X_test_n.astype(theano.config.floatX)
    y_train_ohe = np_utils.to_categorical(y_train_n)
    return X_train, y_train_n, X_test, y_test_n, y_train_ohe

def define_nn_mlp_model(X_train, y_train_ohe, num_neurons_in_layer_1, num_neurons_in_layer_2, init_wts_layer_1, init_wts_layer_2, activation, sgd_lr, sgd_decay, sgd_momentum):
    ''' defines multi-layer-perceptron neural network '''
    model = Sequential() # sequence of layers
    num_inputs = X_train.shape[1] # number of features
    num_classes = 2  # number of classes
    model.add(Dense(input_dim=num_inputs,
                     output_dim=num_neurons_in_layer_1,
                     init=init_wts_layer_1,
                     activation=activation))
    model.add(Dense(input_dim=num_inputs,
                     output_dim=num_neurons_in_layer_1,
                     init=init_wts_layer_1,
                     activation=activation))
    model.add(Dense(input_dim=num_neurons_in_layer_2,
                     output_dim=num_classes,
                     init=init_wts_layer_2,
                     activation='softmax'))
    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=sgd_momentum)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"] )
    return model

def print_output(model, y_train, y_test, rng_seed):
    '''prints model accuracy results'''
    y_train_pred = model.predict_classes(X_train, verbose=0)
    y_test_pred = model.predict_classes(X_test, verbose=0)
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    return test_acc

def manual_grid_search(X_train, y_train, X_test, y_test, y_train_ohe, lst_num_neurons_in_layer_1, lst_num_neurons_in_layer_2, lst_init_wts_layer_1, lst_init_wts_layer_2, lst_activation, lst_sgd_lr, lst_sgd_decay, lst_sgd_momentum, lst_nb_epoch, lst_batch_size, lst_validation_split, rng_seed):
    best_test_acc = 0
    param_sets_list = []
    param_sets_list = list(itertools.product(lst_num_neurons_in_layer_1, lst_num_neurons_in_layer_2, lst_init_wts_layer_1, lst_init_wts_layer_2, lst_activation, lst_sgd_lr, lst_sgd_decay, lst_sgd_momentum, lst_nb_epoch, lst_batch_size, lst_validation_split))
    print param_sets_list
    for (num_neurons_in_layer_1, num_neurons_in_layer_2, init_wts_layer_1, init_wts_layer_2, activation, sgd_lr, sgd_decay, sgd_momentum, nb_epoch, batch_size, validation_split) in param_sets_list:
        model = define_nn_mlp_model(X_train, y_train_ohe, num_neurons_in_layer_1, num_neurons_in_layer_2, init_wts_layer_1, init_wts_layer_2, activation, sgd_lr, sgd_decay, sgd_momentum)
        model.fit(x=X_train, y=y_train_ohe, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=validation_split, verbose=1)
        test_acc = print_output(model, y_train, y_test, rng_seed)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_test_params = [num_neurons_in_layer_1, num_neurons_in_layer_2, init_wts_layer_1, init_wts_layer_2, activation, sgd_lr, sgd_decay, sgd_momentum, nb_epoch, batch_size, validation_split]
    print "best test accuracy = {}".format(best_test_acc)
    print best_test_params

if __name__ == '__main__':
    rng_seed = 2 # set random number generator seed
    np.random.seed(rng_seed)

    X_train, y_train, X_test, y_test, y_train_ohe = load_data()

    lst_num_neurons_in_layer_1 = [3000]
    lst_num_neurons_in_layer_2 = [2000]
    lst_init_wts_layer_1 = ['lecun_uniform']
    lst_init_wts_layer_2 = ['lecun_uniform']
    lst_activation = ['tanh']
    lst_sgd_lr = [0.01]
    lst_sgd_decay = [1e-6]
    lst_sgd_momentum = [0.9]
    lst_nb_epoch = [5]
    lst_batch_size = [4000]
    lst_validation_split = [0]

    manual_grid_search(X_train, y_train, X_test, y_test, y_train_ohe, lst_num_neurons_in_layer_1, lst_num_neurons_in_layer_2, lst_init_wts_layer_1, lst_init_wts_layer_2, lst_activation, lst_sgd_lr, lst_sgd_decay, lst_sgd_momentum, lst_nb_epoch, lst_batch_size, lst_validation_split, rng_seed)

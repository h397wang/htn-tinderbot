import numpy as np
import sys
import os
import time
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import pickle

import theano
import theano.tensor as T

import lasagne
import random

dataset_path = '../../tinder-bot/dataset/'
pickle_dataset = dataset_path+'dataset.pickle'

image = 'result/female/Zoe_Ball_0001.jpg'

PIXELS = 28

def load_dataset():
    with open(pickle_dataset, 'rb') as f:
        dataset = pickle.load(f)
    random.shuffle(dataset)
    num_images = len(dataset)
    num_train = int(num_images*0.8)
    num_val = int((num_images - num_train) / 2)
    num_test = num_images - num_train - num_val
    print("Total: {}, Train: {}, Val {}, Test {}".format(num_images, num_train, num_val, num_test))

    X_train = np.zeros((num_train*3, 1, PIXELS, PIXELS), dtype='float32')
    Y_train = np.zeros(num_train*3, dtype='uint8')
    X_val = np.zeros((num_val, 1, PIXELS, PIXELS), dtype='float32')
    Y_val = np.zeros(num_val, dtype='uint8')
    X_test = np.zeros((num_test, 1, PIXELS, PIXELS), dtype='float32')
    Y_test = np.zeros(num_test, dtype='uint8')

    train_images = dataset[0:num_train-1]
    val_images = dataset[num_train:num_train+num_val-1]
    test_images = dataset[num_train+num_val:]

    positive = 0
    negative = 0
    i = 0
    for image in train_images:
        img = Image.open(dataset_path+image['path'])
        img = ImageOps.fit(img, (PIXELS, PIXELS), Image.ANTIALIAS)
        img = img.convert('L')
        img_blurred = img.filter(ImageFilter.SMOOTH)
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = np.asarray(img, dtype = 'float32') / 255.
        img_blurred = np.asarray(img_blurred, dtype = 'float32') / 255.
        img_flipped = np.asarray(img_flipped, dtype = 'float32') / 255.

        X_train[i][0] = img
        X_train[i+1][0] = img_blurred
        X_train[i+2][0] = img_flipped
        Y_train[i] = int(image['preference'])
        Y_train[i+1] = int(image['preference'])
        Y_train[i+2] = int(image['preference'])
        if Y_train[i] == 1:
            positive +=1
        else:
            negative +=1
        i+=3

    i = 0
    for image in val_images:
        img = Image.open(dataset_path+image['path'])
        img = ImageOps.fit(img, (PIXELS, PIXELS), Image.ANTIALIAS)
        img = img.convert('L')

        img = np.asarray(img, dtype = 'float32') / 255.

        X_val[i][0] = img
        Y_val[i] = int(image['preference'])
        i+=1

    i = 0
    for image in test_images:
        img = Image.open(dataset_path+image['path'])
        # if i == 7:
            # print(image['path'])
            # img.show()
        img = ImageOps.fit(img, (PIXELS, PIXELS), Image.ANTIALIAS)
        img = img.convert('L')

        # if i == 6:
            # img.show()

        img = np.asarray(img, dtype = 'float32') / 255.

        X_test[i][0] = img
        Y_test[i] = int(image['preference'])
        i+=1


    print("Positives: {}, Negatives {}".format(positive, negative))
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)


def build_mlp(input_var=None):
    print("Building network ...")
    l_in = lasagne.layers.InputLayer(shape=(None, 1, PIXELS, PIXELS), input_var=input_var)
    # l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    # l_hid = lasagne.layers.DenseLayer(
        # l_in, num_units=800,
        # nonlinearity=lasagne.nonlinearities.sigmoid,
        # W=lasagne.init.Normal())

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # We'll now add dropout of 50%:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.3)

    # Another 800-unit layer:
    l_hid3 = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=100,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    # 50% dropout again:
    l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.5)

    l_out = lasagne.layers.DenseLayer(
            l_hid3_drop, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    # l_in = lasagne.layers.InputLayer(shape=(None, 1, PIXELS, PIXELS), input_var=input_var)
    # l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    # l_hid = lasagne.layers.DenseLayer(
        # l_in_drop, num_units=800,
        # nonlinearity=lasagne.nonlinearities.rectify,
        # W=lasagne.init.GlorotUniform())
    # l_out = lasagne.layers.DenseLayer(
            # l_hid, num_units=2,
            # nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(num_epochs=50):
     # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_mlp(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

     # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
    # updates = lasagne.updates.sgd(
            # loss, params, learning_rate=1)

    # train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # for i in range(1):
        # for batch in iterate_minibatches(X_train, y_train, 1, shuffle=False):
            # inputs, targets = batch
            # train_fn(inputs, targets)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)

    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):

        params = lasagne.layers.get_all_param_values(network)

        # np.set_printoptions(precision=3, threshold=np.inf)
        # for thing in params:
            # print(thing)

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 25, shuffle=True):
            inputs, targets = batch
            # print(inputs.shape)
            # print(targets.shape)
            # print(inputs)
            # print(targets)
            # print(inputs.dtype)
            # print(targets.dtype)
            train_err += train_fn(inputs, targets)
            train_batches += 1
            # print("Params: {}".format(lasagne.layers.get_all_param_values(network)))

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 10, shuffle=True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        # if (val_acc / val_batches * 100) > 80:
            # break

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 10, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    test_prediction_val = lasagne.layers.get_output(network, deterministic=True)
    # predict_fn = theano.function([input_var], T.argmax(test_prediction_val, axis=1))
    predict_fn = theano.function([input_var], test_prediction_val)

    for i in range(len(X_test)):
        X_val = X_test[i:i+1]
        # print(X_val[0][0].shape)
        # print(X_val[0][0])
        y_val = y_test[i:i+1]

        y_predicted = predict_fn(X_val)

        # print("predicted {}, actual {}".format(y_predicted, y_val))
        print("predicted {}, actual {}".format(np.argmax(y_predicted, axis=1), y_val))

    # params = lasagne.layers.get_all_param_values(network)
    # # print("Params: {}".format(params))
    # # print(type(params))

    # np.set_printoptions(precision=3, threshold=np.inf)
    # for thing in params:
        # print(thing)

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

if __name__ == '__main__':
    main()

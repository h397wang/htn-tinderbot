import numpy as np
import sys
import os
import time
from PIL import Image
from PIL import ImageOps

import theano
import theano.tensor as T

import lasagne

image = 'result/female/Zoe_Ball_0001.jpg'

PIXELS = 28
X = np.zeros((1, 1, PIXELS, PIXELS), dtype='float32')
Y = np.zeros(1, dtype='uint8')
def load_dataset():
    Y[0] = 1
    img = Image.open(image)
    img = ImageOps.fit(img, (PIXELS, PIXELS), Image.ANTIALIAS)
    img = img.convert('L')
    # img.show()
    # print(type(img))
    img = np.asarray(img, dtype = 'float32') / 255.
    # print(img)
    # print(img.shape)
    # print(img[0][0])
    # print(img.shape)
    X[0][0] = img
    # print(X)

    return (X, Y, 1,1,1,1)


def build_mlp(input_var=None):
    print("Building network ...")
    load_dataset()
    l_in = lasagne.layers.InputLayer(shape=(None, 1, PIXELS, PIXELS), input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    l_hid = lasagne.layers.DenseLayer(
        l_in_drop, num_units=800,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    l_out = lasagne.layers.DenseLayer(
            l_hid, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

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

def main(num_epochs=500):
     # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    print(X_train.shape)
    print(y_train.shape)


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
    updates = lasagne.updates.sgd(
            loss, params, learning_rate=1)

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
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 1, shuffle=True):
            inputs, targets = batch
            print(inputs.shape)
            print(targets.shape)
            print(inputs)
            print(targets)
            print(inputs.dtype)
            print(targets.dtype)
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # # And a full pass over the validation data:
        # val_err = 0
        # val_acc = 0
        # val_batches = 0
        # for batch in iterate_minibatches(X_val, y_val, 1, shuffle=False):
            # inputs, targets = batch
            # err, acc = val_fn(inputs, targets)
            # val_err += err
            # val_acc += acc
            # val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        # print("  validation accuracy:\t\t{:.2f} %".format(
            # val_acc / val_batches * 100))

    return

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

if __name__ == '__main__':
    main()

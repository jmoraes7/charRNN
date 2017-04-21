from __future__ import print_function
import os
import time

import tensorflow as tf 
import numpy as np 

vocab = (
        " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "\\^_abcdefghijklmnopqrstuvwxyz{|}")

DATA_PATH = 'input.txt'
HIDDEN_SIZE = 200
BATCH_SIZE = 64
SKIP_STEP = 5
NUM_STEPS = 50
NUM_OUTPUT = len(vocab)
NUM_LAYERS = 2
EPOCHS = 100
TEMPERATURE = 0.7
LEN_GENERATED = 300
LR = 0.003

keep_prob = .5

def vocab_encode(text):
    return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array):
    return ''.join([vocab[x - 1] for x in array])

def read_data(filename, window=NUM_STEPS, overlap=NUM_STEPS // 2):
    for text in open(filename):
        text = vocab_encode(text)
        for start in range(0, len(text) - window, overlap):
            chunk = text[start: start + window]
            chunk += [0] * (window - len(chunk))
            yield chunk

def read_batch(stream, batch_size=BATCH_SIZE):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch

def create_rnn(batch):
    cell = tf.contrib.rnn.GRUCell(num_units=HIDDEN_SIZE, activation = tf.nn.relu)

    in_state = tf.placeholder_with_default(
            cell.zero_state(tf.shape(batch)[0], tf.float32), [None, HIDDEN_SIZE])

    #dropout_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
    #multi_layer_cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS)
    rnn_output, out_state = tf.nn.dynamic_rnn(cell, batch, initial_state=in_state, dtype=tf.float32)

    return rnn_output, in_state, out_state


def create_model(batch, temp, num_steps):
    with tf.device("/cpu:0"):
        batch = tf.one_hot(batch, len(vocab))
        
        rnn_output, in_state, out_state = create_rnn(batch)
        stacked_rnn_output = tf.reshape(rnn_output, [-1, HIDDEN_SIZE])
        stacked_output = tf.contrib.layers.fully_connected(stacked_rnn_output,
                                                            NUM_OUTPUT, activation_fn=None)

        logits = tf.reshape(stacked_output, [-1, num_steps, NUM_OUTPUT])
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1], labels=batch[:, 1:]))

        # sample the next character from Maxwell-Boltzmann Distribution with temperature temp
        sample = tf.multinomial(tf.exp(logits[:, -1] / temp), 1)[:, 0] 

    return loss, logits, in_state, out_state, sample


def training(X, loss, optimizer, global_step, logits, sample, temp, in_state, out_state, num_steps):
    saver = tf.train.Saver()
    start = time.time()

    logits = tf.argmax(logits, axis=2)
    char2vec = tf.argmax(tf.one_hot(X, len(vocab)), axis=2)


    with tf.Session() as sess:
        writer = tf.summary.FileWriter('graphs/shakespeare', sess.graph)
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/shakespeare/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(EPOCHS):
            iteration = global_step.eval()
            for batch in read_batch(read_data(DATA_PATH)):
                sess.run([optimizer], feed_dict={X: batch})
                
                print(iteration)
                if (iteration + 1) % SKIP_STEP == 0:
                    batch_loss = sess.run([loss], feed_dict={X: batch})
                    print('Iter {}. \n    Loss {}. Time {}'.format(iteration, batch_loss, time.time() - start))

                    start = time.time()
                    saver.save(sess, 'checkpoints/shakespeare/char-rnn', iteration)

                    a, b= sess.run([logits[:,:-1], char2vec[:, 1:]], feed_dict={X: batch})
                    
                    print('----------------------------------')
                    print('Expected Output: ' + vocab_decode(b[0]))
                    print("                 " + vocab_decode(b[1]))
                    print("                 " + vocab_decode(b[2]))
                    print("                 " + vocab_decode(b[3]))

                    print('Prediction:      ' + vocab_decode(a[0]))
                    print('          :      ' + vocab_decode(a[1]))
                    print('          :      ' + vocab_decode(a[2]))
                    print('          :      ' + vocab_decode(a[3]))

                    generateSample(sess, X, sample, temp, in_state, out_state, num_steps)

                iteration += 1

def getSampleHiddenState(sess, seed, X, out_state):
    seed = [vocab_encode(seed[:-1])]
    state = sess.run(out_state, feed_dict = {X : seed})
    return state

def generateSample(sess, X, sample, temp, in_state, out_state, num_steps, seed='T'):
    sentence = seed
    state = None
    if (len(seed) > 1): state = getSampleHiddenState(sess, seed, X, out_state)

    for _ in range(LEN_GENERATED):
        batch = [vocab_encode(sentence[-1])]
        feed = {X: batch, temp: TEMPERATURE, num_steps: 1}
        # for the first decoder step, the state is None
        if state is not None:
            feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += vocab_decode(index)
    print('Sample')
    print(sentence)
    print('')


def main():
    X = tf.placeholder(tf.int32, [None, None])
    temp = tf.placeholder(tf.float32)
    num_steps = tf.placeholder_with_default(NUM_STEPS, None)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    loss, logits, in_state, out_state, sample = create_model(X, temp, num_steps)
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss, global_step=global_step)

    training(X, loss, optimizer, global_step, logits, sample, temp, in_state, out_state, num_steps)


if __name__ == '__main__':
    main()

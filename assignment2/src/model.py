import numpy as np
import tensorflow as tf
import sys

class Model():
    def __init__(self, params):

        layer_size = params.layer_size
        batch_size = params.batch_size
        seqlen = params.seqlen
        num_layers = params.num_layers
        lr = params.lr
        keep_prob = params.keep_prob
        wv_size = params.wv_size

        self.train_x  = tf.placeholder(tf.float32, [batch_size,seqlen,wv_size])
        self.train_y  = tf.placeholder(tf.float32, [batch_size,1])

        def new_layer(layer_size):
            return tf.contrib.rnn.BasicLSTMCell(layer_size, state_is_tuple=False)

        cell = tf.contrib.rnn.MultiRNNCell([new_layer(layer_size) for _ in range(num_layers)])
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        self.cell = cell

        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        self.input_state = tf.identity(init_state)

        model_input = tf.unstack(self.train_x, axis=1)

        net_output, output_state = tf.contrib.legacy_seq2seq.rnn_decoder(model_input, init_state, cell, loop_function=None)
        self.output_state = tf.identity(output_state)

        class_weights = tf.get_variable("class_weights",shape=[layer_size,1])
        class_biases = tf.get_variable("class_biases",shape=[1])

        class_input = tf.reshape(tf.concat(axis=1, values=net_output), [-1,layer_size])
        pred = tf.nn.sigmoid(tf.nn.xw_plus_b(class_input, class_weights, class_biases))


        self.loss = tf.div(tf.nn.l2_loss(pred-train_y),batch_size )

        ## Updates
        # train_var = tf.trainable_variables()
        # gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_var), 10.0)
        # self.updates = tf.train.AdamOptimizer(lr).apply_gradients(zip(gradients, train_var))
        self.updates = tf.train.AdamOptimizer(lr).minimize(self.loss)


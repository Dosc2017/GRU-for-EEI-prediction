
# Dependency:
# Python 3.6
# TensorFlow 1.7

# Import the libraries
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# Ignore some warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import data
train_data = np.array(pd.read_excel("ccpp_train.xlsx"))
test_data = np.array(pd.read_excel("ccpp_test.xlsx"))


# Split the data into batches
def generate_batch(arrays, time_size, batch_size):
    batches = np.zeros((batch_size, time_size, 5))
    for i in range(batch_size):
        for j in range(time_size):
            batches[i, j, :] = arrays[i + j*90, :]
    return batches


train = generate_batch(train_data, 100, 90)
test = generate_batch(test_data, 100, 90)


# Z-score standardization
def std_scale(array, batch_size):
    std_array = np.zeros((batch_size, 100, 4))
    for i in range(batch_size):
        temp_array = array[i, :, :]
        std_array[i, :, :] = preprocessing.scale(temp_array)
    return std_array


X_train = train[:, :, :4]
X_train = std_scale(X_train, 90)
Y_train = train[:, :, 4:]
X_test = test[:, :, :4]
X_test = std_scale(X_test, 90)
Y_test = test[:, :, 4:]

batch_size = 90
time_size = 100


# Define the model
class GRU:

    def __init__(self, input_dimensions, hidden_size, dtype=tf.float64):
        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size

        self.Wr = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=3),
            name='Wz')
        self.Wz = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=3),
            name='Wz')
        self.Wh = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=3),
            name='Wh')

        self.Ur = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=1),
            name='Ur')
        self.Uz = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=1),
            name='Uz')
        self.Uh = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=1),
            name='Uh')

        self.br = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=1),
                              name='br')
        self.bz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=1),
                              name='bz')
        self.bh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=1),
                              name='bh')

        # Define the input_layer placeholder
        self.input_layer = tf.placeholder(dtype=tf.float64, shape=(None, None, input_dimensions), name='input')

        # Transpose the input layer
        self.x_t = tf.transpose(self.input_layer, [1, 0, 2], name='x_t')

        # Define the initial hidden state h_0
        self.h_0 = tf.matmul(self.x_t[0, :, :], tf.zeros(dtype=tf.float64, shape=(input_dimensions, hidden_size)),
                             name='h_0')

        # Perform the scan operator
        self.h_t_transposed = tf.scan(self.forward_pass, self.x_t, initializer=self.h_0, name='h_t_transposed')

        # Transpose the result back
        self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')

    def forward_pass(self, h_tm1, x_t):
            # Define GRU cell
            # Definitions of z_t and r_t
            z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
            r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)

            # Definition of h~_t
            h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)

            # Compute the next hidden state
            h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

            return h_t


# Initialize and train the model.
input_dimensions = 4
hidden_size = 10

# Initialize a session
session = tf.Session()

# Create a new instance of the GRU model
gru = GRU(input_dimensions, hidden_size)

# Add an additional layer on top of each of the hidden state outputs
W_output = tf.Variable(tf.truncated_normal(dtype=tf.float64, shape=(hidden_size, 1), mean=0, stddev=5))
b_output = tf.Variable(tf.truncated_normal(dtype=tf.float64, shape=(1,), mean=0, stddev=1))
output = tf.map_fn(lambda h_t: tf.matmul(h_t, W_output) + b_output, gru.h_t)

# Create a placeholder for the expected output
expected_output = tf.placeholder(dtype=tf.float64, shape=(batch_size, time_size, 1), name='expected_output')

# Use quadratic loss
loss = tf.reduce_sum(tf.pow(output - expected_output, 2)) / float(batch_size)

# Use the Adam optimizer for training
train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# Initialize all the variables
init_variables = tf.global_variables_initializer()
session.run(init_variables)

# Initialize the losses
train_losses = []
validation_losses = []


for epoch in range(6000):
    # Compute the losses
    _, train_loss = session.run([train_step, loss], feed_dict={gru.input_layer: X_train, expected_output: Y_train})
    validation_loss = session.run(loss, feed_dict={gru.input_layer: X_test, expected_output: Y_test})

    # Log the losses
    train_losses += [train_loss]
    validation_losses += [validation_loss]
    # Plot the final loss
    if epoch == 5999:
        plt.plot(train_losses, '-b', label='Train loss')
        plt.plot(validation_losses, '-r', label='Validation loss')
        plt.legend(loc=0)
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
        print('Iteration: %d, train loss: %.4f, test loss: %.4f' % (epoch, train_loss, validation_loss))


# Save the model
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_variables)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path: ", save_path)


# A simple comparision of prediction and actual data of one batch
X_pred = np.zeros((1, 100, 4))
X_pred = X_test[1:2, :, :]
Y_expect = np.reshape(Y_test[1, :, :], -1)
Y_pred = np.reshape(session.run(output, feed_dict={gru.input_layer: X_pred}), -1)


plt.plot(Y_expect, '-b', label='expected output')
plt.plot(Y_pred, '-r', label='predicted output')
plt.show()













import tensorflow as tf
import numpy as np
from PIL import Image

### Fetch the data ###

# Import the MNIST dataset and store the image data in the variable mnist.
# The MNIST dataset is a set of 28x28 pixel images of handwritten digits.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Count the images in each of the subsets of the dataset.
number_of_train_images = mnist.train.num_examples # 55,000
number_of_validation_images = mnist.validation.num_examples  # 5000
number_of_test_images = mnist.test.num_examples  # 10,000

print 'Number of images in the train subset:' + number_of_train_images
print 'Number of images in the validation subset:' + number_of_validation_images
print 'Number of images in the test subset:' + number_of_test_images

### Define the architecture of the neural network (number layers, number of neurons ("units") in each layer, etc). ###

number_of_units_in_input_layer = 784 # Input layer (28x28 pixel image flattened as a vector/array of 784 values representing the greyscale of each pixel as integers between 0 and 255).
number_of_units_in_hidden1_layer = 512 # 1st hidden layer (hidden layers are the ones between the input and output layer).
number_of_units_in_hidden2_layer = 256 # 2nd hidden layer.
number_of_units_in_hidden3_layer = 128 # 3rd hidden layer.
number_of_units_in_output_layer = 10 # Output layer (0-9 digits).

# Hyperparameters (constants/settings) configuring the learning process.
learning_rate = 1e-4 # How much the parameters will adjust after each step of the learning process, to tune the weights to reduce loss.
number_of_training_iterations = 1000 # How many times to go through the training step.
batch_size = 128 # How many training examples to use at each step.
dropout = 0.5 # Threshold at which to eliminate some units at random. Gives each unit a 50% chance of being eliminated at every training step. Used in the final hidden layer.

### Build the neural network/graph in Tensorflow ###

# Placeholder tensors.
# Tensors are data structures, like an array. They get passed through the network, and updated as the network learns.
# None = any amount.
X = tf.placeholder('float', [None, number_of_units_in_input_layer])
Y = tf.placeholder('float', [None, number_of_units_in_output_layer])
keep_probability = tf.placeholder(tf.float32)

# The weights and biases are where the network learning happens. They define the strength of the connections between the units/neurons. They get updated during the training process.
# Initialize the weights as random small decimals. Random so they all change independently, and small so they can go positive or negative.
weights = {
  'hidden1_layer': tf.Variable(tf.truncated_normal([number_of_units_in_input_layer, number_of_units_in_hidden1_layer], stddev=0.1)),
  'hidden2_layer': tf.Variable(tf.truncated_normal([number_of_units_in_hidden1_layer, number_of_units_in_hidden2_layer], stddev=0.1)),
  'hidden3_layer': tf.Variable(tf.truncated_normal([number_of_units_in_hidden2_layer, number_of_units_in_hidden3_layer], stddev=0.1)),
  'output_layer': tf.Variable(tf.truncated_normal([number_of_units_in_hidden3_layer, number_of_units_in_output_layer], stddev=0.1)),
}

# Initialize the biases.
biases = {
  'hidden1_layer': tf.Variable(tf.constant(0.1, shape=[number_of_units_in_hidden1_layer])),
  'hidden2_layer': tf.Variable(tf.constant(0.1, shape=[number_of_units_in_hidden2_layer])),
  'hidden3_layer': tf.Variable(tf.constant(0.1, shape=[number_of_units_in_hidden3_layer])),
  'output_layer': tf.Variable(tf.constant(0.1, shape=[number_of_units_in_output_layer]))
}

# Connect the layers of the network to manipulate the tensors as they pass through the network.
# Each of the hidden layers will execute matrix multiplication on the outputs of the previous layer and the weights of the current layer, and add the bias.
# At the last hidden layer apply a dropout operation using keep_probability.
layer_1 = tf.add(tf.matmul(X, weights['hidden1_layer']), biases['hidden1_layer'])
layer_2 = tf.add(tf.matmul(layer_1, weights['hidden2_layer']), biases['hidden2_layer'])
layer_3 = tf.add(tf.matmul(layer_2, weights['hidden3_layer']), biases['hidden3_layer'])
layer_drop = tf.nn.dropout(layer_3, keep_probability)
output_layer = tf.matmul(layer_3, weights['output_layer']) + biases['output_layer']

# Specify the loss function (cross-entropy) and optimization algorithm to minimize the loss function (Adam optimizer).
# cross-entropy loss function quantifies the difference between the predictions and the correct answer, with zero meaning a perfect classification.
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(
    labels=Y,
    logits=output_layer
  )
)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

### Training the neural network ###
# Training means running the training subset of the dataset through the network, and updating the parameters after each batch to optimize the loss function.

# Compare which image inputs are being predicted correctly by looking at the output_layer (predictions from the network) and Y (labels/correct answers), casted to an array of booleans.
correct_predictions = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))

# Cast the correct predictions array to floats and calculate the mean to get the total accuracy score.
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Initialize the session for running the training data through the graph/network.
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

# While training, the goal is to optimize the loss function to minimize the difference between the network's predictions and the actual values.
# Do this by propagating values through the network, calculating the loss, then propagating the values backward through the network, updating the parameters.
# Train in mini batches.
for i in range(number_of_training_iterations):
  batch_x, batch_y = mnist.train.next_batch(batch_size)
  session.run(train_step, feed_dict={
    X: batch_x,
    Y: batch_y,
    keep_probability: dropout
  })

  # Print loss and accuracy on a running basis.
  if i % 100 == 0:
    mini_batch_loss, mini_batch_accuracy = session.run(
      [cross_entropy, accuracy],
      feed_dict={X: batch_x, Y: batch_y, keep_probability: 1.0}
    )
    print(
      "Iteration",
      str(i),
      "\t| Loss =",
      str(mini_batch_loss),
      "\t| Accuracy =",
      str(mini_batch_accuracy)
    )

### Testing the neural network ###
# Testing means running the testing subset of the dataset through the trained network, and tracking the correct predictions to determine how accurate the network is.
# Accuracy should be about 92%.
test_accuracy = session.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_probability: 1.0})
print("\nAccuracy on test subset:", test_accuracy)

### Test the network with manual input ###
# input_image = np.invert(Image.open('/path/to/test_img.png').convert('L')).ravel()
# prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [input_image]})
# print ("Prediction for input image:", np.squeeze(prediction))

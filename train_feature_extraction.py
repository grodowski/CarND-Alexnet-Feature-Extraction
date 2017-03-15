import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

# TODO: Load traffic signs data.
with open('./train.p', 'rb') as f:
    data = pickle.load(f)

# inspect data
for k,v in data.items():
    print(k, v.shape)

nb_classes = 43

# temp limit features for testing on CPU
tmp_x, tmp_y = shuffle(data['features'], data['labels'], random_state=42)
# data_x = tmp_x[0:2000]
# data_y = tmp_y[0:2000]
data_x = tmp_x
data_y = tmp_y

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(
    data_x,
    data_y,
    test_size=0.33,
    random_state=42
)

print('Training: {}'.format(X_train.shape))
print('Test: {}'.format(X_test.shape))

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
y = tf.placeholder(tf.int64, None) # one_hot

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes) # use this shape for the weight matrix
fc8_w = tf.Variable(tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1))
fc8_b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8_w, fc8_b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot_y = tf.one_hot(y, nb_classes)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fc8_w, fc8_b])

# declare accuracy operation and evaluate
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(sess, X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print('Training...')
EPOCHS = 10
BATCH_SIZE = 256
for i in range(EPOCHS): # run epochs print('Starting epoch {}'.format(i + 1))
    t0 = time.time()
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, len(X_train), BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
    # evaluate
    print("Time: {:.4f} seconds".format(time.time() - t0))
    print('Epoch {} accuracy: {:.4f}'.format(i + 1, evaluate(sess, X_test, y_test)))

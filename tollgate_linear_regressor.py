import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os


# travel time.
# travel_training_df = pd.read_csv('dataSets/training/training_travel_time_dataset.csv', dtype=np.float32, index_col=0)
# travel_training_df = travel_training_df.dropna()
# travel_training_df.wind_direction[travel_training_df.wind_direction > 360.0] = 0.0
# travel_test_df = pd.read_csv('dataSets/testing-phase1/test1_travel_time_dataset.csv', dtype=np.float32, index_col=0)
# travel_test_df = travel_test_df.dropna()
# travel_test_df.wind_direction[travel_test_df.wind_direction > 360.0] = 0.0
# submission_df = pd.read_csv('dataSets/testing-phase1/submission_travel_time_dataset.csv', dtype=np.float32, index_col=0)
# print('before filtered dirty rows, submission size: ', submission_df.shape)
# submission_df = submission_df.dropna()
# submission_df.wind_direction[submission_df.wind_direction > 360.0] = 0.0
# print('after filtered dirty rows, submission size: ', submission_df.shape)
# submission_sample = 'dataSets/testing-phase1/submission_sample_travelTime.csv'
# ylimit = 1.0

# volume.
travel_training_df = pd.read_csv('dataSets/training/training_volume_dataset.csv', dtype=np.float32, index_col=0)
travel_training_df = travel_training_df.dropna()
travel_training_df.wind_direction[travel_training_df.wind_direction > 360.0] = 0.0
travel_test_df = pd.read_csv('dataSets/testing-phase1/test1_volume_dataset.csv', dtype=np.float32, index_col=0)
travel_test_df = travel_test_df.dropna()
travel_test_df.wind_direction[travel_test_df.wind_direction > 360.0] = 0.0
submission_df = pd.read_csv('dataSets/testing-phase1/submission_volume_dataset.csv', dtype=np.float32, index_col=0)
print('before filtered dirty rows, submission size: ', submission_df.shape)
submission_df = submission_df.dropna()
submission_df.wind_direction[submission_df.wind_direction > 360.0] = 0.0
print('after filtered dirty rows, submission size: ', submission_df.shape)
submission_sample = 'dataSets/testing-phase1/submission_sample_volume.csv'
ylimit = 10.0


def feature_normalize(dataset, mu=None, sigma=None):
    if mu is None:
        mu = np.mean(dataset, axis=0)
    if sigma is None:
        sigma = np.std(dataset, axis=0) + 0.1  # in case zero division.
    return (dataset - mu) / sigma, mu, sigma


travel_training_dataset, mu, sigma = feature_normalize(travel_training_df.as_matrix(
    columns=travel_training_df.columns[: -1]))
travel_training_labels = travel_training_df.as_matrix(
    columns=[travel_training_df.columns[-1]])
travel_test_dataset, _, _ = feature_normalize(travel_test_df.as_matrix(
    columns=travel_test_df.columns[:-1]), mu=mu, sigma=sigma)
travel_test_labels = travel_test_df.as_matrix(
    columns=[travel_test_df.columns[-1]])

submission_dataset, _, _ = feature_normalize(submission_df.as_matrix(
    columns=submission_df.columns[:-1]), mu=mu, sigma=sigma)
submission_rst = submission_sample.split('_')[0] + '_' + submission_sample.split('_')[2]

train_dataset = travel_training_dataset
print('training dataset size: ', train_dataset.shape)
train_labels = np.reshape(travel_training_labels, newshape=[-1, 1])
test_dataset = travel_test_dataset
print('test dataset size: ', test_dataset.shape)
test_labels = np.reshape(travel_test_labels, newshape=[-1, 1])

n_dim = train_dataset.shape[1]
num_epochs = 10
batch_size = travel_training_dataset.shape[0] // 1000
# testing
# num_steps = 10
num_steps = travel_training_dataset.shape[0] // batch_size
report_interval = 5

cost_history = []
cost_epoch_history = []
loss_history = []

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(dtype=tf.float32, shape=[None, n_dim])
    tf_train_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    tf_valid_dataset = tf.constant(test_dataset)
    tf_valid_labels = tf.constant(test_labels)
    tf_test_dataset = tf.constant(test_dataset)
    tf_test_labels = tf.constant(test_labels)

    global_step = tf.Variable(0)  # count the number of steps taken.
    initial_learning_rate = 0.05
    final_learning_rate = 0.01
    decay_rate = 0.96
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_rate=decay_rate,
                                               decay_steps=num_steps / (
                                               np.log(final_learning_rate / initial_learning_rate) / np.log(
                                                   decay_rate)))

    weights = tf.get_variable('weights', [n_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.Variable(tf.ones(shape=[1]))

    # MSE loss
    # loss = tf.reduce_mean(tf.square(tf.divide(tf.nn.xw_plus_b(tf_train_dataset, weights, bias) - tf_train_labels,
    #                                           tf_train_labels)))
    predicts = tf.nn.xw_plus_b(tf_train_dataset, weights, bias)
    loss = tf.reduce_mean(tf.square(predicts - tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    # MSPE metric
    training_metric = tf.reduce_mean(tf.abs(tf.divide(tf.nn.xw_plus_b(tf_train_dataset, weights, bias) - tf_train_labels,
                                                      tf_train_labels)))
    validation_metric = tf.reduce_mean(tf.abs(tf.divide(tf.nn.xw_plus_b(tf_valid_dataset, weights, bias) - tf_valid_labels,
                                                        tf_valid_labels)))
    test_metric = tf.reduce_mean(tf.abs(tf.divide(tf.maximum(tf.nn.xw_plus_b(tf_test_dataset, weights, bias), 0.0) - tf_test_labels,
                                                  tf_test_labels)))

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for epoch in range(num_epochs):
            shuffle = np.random.permutation(train_dataset.shape[0])
            train_dataset = train_dataset[shuffle]
            train_labels = train_labels[shuffle]
            for step in range(num_steps):
                offset = batch_size * step % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                _, l, tm = sess.run(fetches=[optimizer, loss, training_metric], feed_dict=feed_dict)
                if step % report_interval:
                    print('Minibatch loss at step %d: %.4f' % (step, l))
                    print('Minibatch metric: %.4f' % tm)
                    print('Validation metric: %.4f\n' % validation_metric.eval())
                cost_history.append(tm)
                loss_history.append(l)
            print('Test metric: %.4f' % test_metric.eval())
            cost_epoch_history.append(test_metric.eval())

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3)
        ax1.plot(range(len(loss_history)), loss_history)
        ax1.set_xlim([0, len(loss_history)])
        ax1.set_ylim([0, np.max(loss_history)])
        ax2.plot(range(len(cost_history)), cost_history)
        ax2.set_xlim([0, len(cost_history)])
        ax2.set_ylim([0, ylimit])
        ax3.scatter(range(len(cost_epoch_history)), cost_epoch_history)
        ax3.set_xlim([0, len(cost_epoch_history)])
        ax3.set_ylim([0, 1.0])
        plt.show()

        # predict
        preds, = sess.run([predicts], feed_dict={tf_train_dataset: submission_dataset})
        # generate submission
        with open(submission_sample, 'r') as f_in:
            with open(submission_rst, 'w') as f_out:
                idx = 0
                for line in f_in:
                    if idx == 0:
                        f_out.write(line.rstrip() + os.linesep)
                    else:
                        line = line.rstrip().rsplit(',', maxsplit=1)[0]
                        pre = str.format('%.2f' % preds[idx-1, 0])
                        f_out.write(line + ',' + pre + os.linesep)
                    idx += 1
        print('finished prediction.')


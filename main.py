import preprop as preprop
import logging
from gensim.models import Word2Vec
import numpy as np
import analysis
from sklearn.model_selection import KFold
import trainHelper

import tensorflow as tf
from tensorflow.contrib import rnn

# control flags
RE_TRAIN_EMBBED = True

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

file = open('../dtrain.csv', 'r')
file = preprop.decoding(file)
labels = preprop.getLabel(file)

file_test = open('../test.csv', 'r')
file_test = preprop.decoding(file_test)

# generate necessary data for word embedding
uni_gram = preprop.genNgram(file, 1)
bi_gram = preprop.genNgram(file, 2)
tri_gram = preprop.genNgram(file, 3)

uni_gram_test = preprop.genNgram(file_test, 1)
bi_gram_test = preprop.genNgram(file_test, 2)
tri_gram_test = preprop.genNgram(file_test, 3)

if RE_TRAIN_EMBBED:
    w2v_uni = Word2Vec(sentences=uni_gram+uni_gram_test, size=10, window=5, min_count=1, workers=2,
                 sg=1, iter=10)
    w2v_uni.save("w2v_uni10.model")
    w2v_bi = Word2Vec(sentences=bi_gram+bi_gram_test, size=50, window=5, min_count=1, workers=2,
                       sg=1, iter=10)
    w2v_bi.save("w2v_bi50.model")
    w2v_tri = Word2Vec(sentences=tri_gram+tri_gram_test, size=100, window=5, min_count=1, workers=2,
                       sg=1, iter=10)
    w2v_tri.save("w2v_tri100.model")

w2v_uni = Word2Vec.load("w2v_uni10.model")
w2v_bi = Word2Vec.load("w2v_bi50.model")
w2v_tri = Word2Vec.load("w2v_tri100.model")

# below block used to visualize t-SNE
# bi_gram_keys = set()
# for line in uni_gram:
#     for key in line:
#         bi_gram_keys.add(key)
# n_gram_keys = sorted(list(bi_gram_keys))
# n_gram_keys = [(key, idx) for (idx,key) in enumerate(n_gram_keys)]
# n_gram_keys = dict(n_gram_keys)
# val = len(n_gram_keys.keys())

# use word2vec model to obtain each embedded key
# X = np.array([w2v_uni.wv[key] for key in n_gram_keys])
# analysis.t_SNE_visualization(X, n_gram_keys)

uni_gram = preprop.ngram2embedded(uni_gram, w2v_uni)
bi_gram = preprop.ngram2embedded(bi_gram, w2v_bi)
tri_gram = preprop.ngram2embedded(tri_gram, w2v_tri)
uni_gram_test = preprop.ngram2embedded(uni_gram_test, w2v_uni)
bi_gram_test = preprop.ngram2embedded(bi_gram_test, w2v_bi)
tri_gram_test = preprop.ngram2embedded(tri_gram_test, w2v_tri)

uni_max_timestep = max([len(x) for x in uni_gram+uni_gram_test])
bi_max_timestep = max([len(x) for x in bi_gram+bi_gram_test])
tri_max_timestep = max([len(x) for x in tri_gram+tri_gram_test])

uni_gram = preprop.ngram2padded(uni_gram, uni_max_timestep)
bi_gram = preprop.ngram2padded(bi_gram, bi_max_timestep)
tri_gram = preprop.ngram2padded(tri_gram, tri_max_timestep)

uni_gram_test = preprop.ngram2padded(uni_gram_test, uni_max_timestep)
bi_gram_test = preprop.ngram2padded(bi_gram_test, bi_max_timestep)
tri_gram_test = preprop.ngram2padded(tri_gram_test, tri_max_timestep)

print('necessary data ready now')

# training LSTM, cross validation
tmp = np.zeros((len(uni_gram), 1))
kf = KFold(n_splits=10, shuffle=True)
for train_idx, test_idx in kf.split(tmp):

    tf.reset_default_graph()

    # hyper-parameters
    lr = 1e-2  #3.5e-3
    LSTM_hidden_size = 100  # LSTM state dimension
    # ==== some parameters =====
    # input feature dim for every time step, varies for ngrams
    # eg unigram=10, bigram=100, .. (ie, same as embbeding dim)
    unigram_timestep = len(uni_gram[0])
    unigram_dim_per_step = len(uni_gram[0][0])
    bigram_timestep = len(bi_gram[0])
    bigram_dim_per_step = len(bi_gram[0][0])
    trigram_timestep = len(tri_gram[0])
    trigram_dim_per_step = len(tri_gram[0][0])

    # class[0] non-brazil [1] brazil
    class_num = 2
    batch_size_training = 2048
    # ===== end parameters =====

    # convert labels to one-hot encoding
    label_onehot = np.zeros((len(labels), class_num ))
    label_onehot[np.arange(len(labels)), labels] = 1
    # this just a filler for testing label
    label_Test = np.zeros((len(bi_gram_test), class_num))

    X_train_uni = uni_gram[train_idx]
    X_train_bi = bi_gram[train_idx]
    X_train_tri = tri_gram[train_idx]
    X_test_uni = uni_gram[test_idx]
    X_test_bi = bi_gram[test_idx]
    X_test_tri = tri_gram[test_idx]

    Y_train = label_onehot[train_idx]
    Y_test = label_onehot[test_idx]



    # when training and testing, batch_size is different (128 vs test set)
    # type need to be int32
    batch_size = tf.placeholder(tf.int32, [])
    _X_uni = tf.placeholder(tf.float32, [None, unigram_timestep, unigram_dim_per_step])
    _X_bi = tf.placeholder(tf.float32, [None, bigram_timestep, bigram_dim_per_step])
    _X_tri = tf.placeholder(tf.float32, [None, trigram_timestep, trigram_dim_per_step])
    y = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)

    ####################################################################
    # step 1: rnn input shape = (batch_size, timestep_size, input_size_per_step)


    # step 2：single layer LSTM_cell, only need to specify hidden state size




    # step 3: add dropout layer, this wrapper can drop input weight & output weight
    #         in the paper proposal arch, only output dropout is used.


    lstm_cell_uni = rnn.BasicLSTMCell(num_units=LSTM_hidden_size, forget_bias=1.0, state_is_tuple=True, name="uni")
    init_state_uni = lstm_cell_uni.zero_state(batch_size, dtype=tf.float32)
    lstm_cell_uni = rnn.DropoutWrapper(cell=lstm_cell_uni, input_keep_prob=1.0, output_keep_prob=keep_prob)
    outputs_uni, state_uni = tf.nn.dynamic_rnn(lstm_cell_uni, inputs=_X_uni, initial_state=init_state_uni, time_major=False)
    h_state_uni = outputs_uni[:, -1, :]

    lstm_cell_bi = rnn.BasicLSTMCell(num_units=LSTM_hidden_size, forget_bias=1.0, state_is_tuple=True, name="bi")
    init_state_bi = lstm_cell_bi.zero_state(batch_size, dtype=tf.float32)
    lstm_cell_bi = rnn.DropoutWrapper(cell=lstm_cell_bi, input_keep_prob=1.0, output_keep_prob=keep_prob)
    outputs_bi, state_bi = tf.nn.dynamic_rnn(lstm_cell_bi, inputs=_X_bi, initial_state=init_state_bi, time_major=False)
    h_state_bi = outputs_bi[:, -1, :]

    lstm_cell_tri = rnn.BasicLSTMCell(num_units=LSTM_hidden_size, forget_bias=1.0, state_is_tuple=True, name="tri")
    init_state_tri = lstm_cell_tri.zero_state(batch_size, dtype=tf.float32)
    outputs_tri, state_tri = tf.nn.dynamic_rnn(lstm_cell_tri, inputs=_X_tri, initial_state=init_state_tri, time_major=False)
    lstm_cell_tri = rnn.DropoutWrapper(cell=lstm_cell_tri, input_keep_prob=1.0, output_keep_prob=keep_prob)
    h_state_tri = outputs_tri[:, -1, :]
    # step 5: init internal state for LSTM

    # step 6： run forward prop along time
    #          when time_major==False, outputs.shape = [batch_size, timestep_size, hidden_size]
    # to obtain last timestep output, just use h_state = outputs[:, -1, :]
    #           state.shape = [layer_num, 2, batch_size, hidden_size],
    # final output shape is [batch_size, hidden_size]




    # step 7: TODO, concate features, connect to a FC layer
    hidden_layer_output_concat = tf.concat([h_state_uni, h_state_bi, h_state_tri], 1)

    # step 8: hidden layer and then output need a softmax
    W = tf.Variable(tf.truncated_normal([3*LSTM_hidden_size, class_num], stddev=0.1), dtype=tf.float32)
    #                                    ^^^ this is because directly use LSTM output currently
    bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(hidden_layer_output_concat, W) + bias)


    # step 9: after obtaining softmax output, use cross-entropy loss and adam optimizer
    cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    output_trainbatchaccur = []
    output_testaccur = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_cnt = 0
        batch_total = 0
        for i in range(500):
            _batch_size = batch_size_training
            if  batch_cnt == batch_total: # next batch idx not exist, regen batch idx
                batch_idx = trainHelper.batch_gen(X_train_bi, _batch_size) # doesn't matter which train set
                batch_total = len(batch_idx)
                batch_uni = (X_train_uni[batch_idx[0]], 0)
                batch_bi = (X_train_bi[batch_idx[0]], Y_train[batch_idx[0]])
                batch_tri= (X_train_tri[batch_idx[0]], 0)
                batch_cnt = 1
            else:
                batch_uni = (X_train_uni[batch_idx[batch_cnt]], 0)
                batch_bi = (X_train_bi[batch_idx[batch_cnt]], Y_train[batch_idx[batch_cnt]])
                batch_tri = (X_train_tri[batch_idx[batch_cnt]], 0)
                batch_cnt = batch_cnt + 1

            if (i+1)%10 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={
                    _X_uni:batch_uni[0], _X_bi:batch_bi[0], _X_tri:batch_tri[0], y: batch_bi[1],
                    keep_prob: 1.0, batch_size: _batch_size})
                print("Iter%d, training accuracy %g on batch %d" % ( i, train_accuracy, batch_cnt))
                output_trainbatchaccur.append(1.-train_accuracy)

                test_accuracy = sess.run(accuracy, feed_dict={
                    _X_uni: X_test_uni, _X_bi: X_test_bi, _X_tri: X_test_tri, y: Y_test,
                    keep_prob: 1.0, batch_size: X_test_bi.shape[0]})
                print("Iter%d, testing accuracy %g " % (i, test_accuracy))
                output_testaccur.append(1.-test_accuracy)

            sess.run(train_op, feed_dict={_X_uni:batch_uni[0], _X_bi:batch_bi[0], _X_tri:batch_tri[0],
                                          y: batch_bi[1], keep_prob: 0.5, batch_size: _batch_size})

        # test accuracy
        # _X_bi: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]}))
        trainHelper.write_accuracy_output(output_trainbatchaccur, 'train_err_lr1e-2_emddown.csv')
        trainHelper.write_accuracy_output(output_testaccur, 'test_err_lr1e-2embdown.csv')

        class_score = sess.run(y_pre, feed_dict={_X_uni: uni_gram_test, _X_bi: bi_gram_test,
            _X_tri: tri_gram_test, y: label_Test, keep_prob: 1.0, batch_size: bi_gram_test.shape[0]})
        class_score = np.argmax(class_score, axis=1)
        trainHelper.write_pred_output(class_score)

    break
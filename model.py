import os

import numpy as np
from tensorflow.nn.rnn_cell import GRUCell
from modules import *
from dmon.utils import *
from utils.gat import *
from utils.rnn_dien import dynamic_rnn as dynamic_rnn_dien
from utils.rnn_cell_implement import VecAttGRUCell
from dmon.dmon import *
from dmon.gcn import *
from dmon.utils import *
from settransformer import SetTransformer
import copy
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering

# baseline
class Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
    # def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, reg_attend, reg_contrast, reg_construct, flag="DNN"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.neg_num = 10
        self.reg_attend = 0.1
        self.reg_contrast = 0.1
        self.reg_construct = 0.1
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.lr = tf.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(input_tensor=self.mask, axis=-1), dtype=tf.float32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, embedding_dim], trainable=True)
            self.mid_embeddings_bias = tf.get_variable("bias_lookup_table", [n_mid], initializer=tf.zeros_initializer(),
                                                       trainable=False)
            self.mid_batch_embedded = tf.nn.embedding_lookup(params=self.mid_embeddings_var, ids=self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(params=self.mid_embeddings_var,
                                                                 ids=self.mid_his_batch_ph)

        self.item_eb = self.mid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))

    def build_sampled_softmax_loss(self, item_emb, user_emb):
        self.loss = tf.reduce_mean(input_tensor=tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_attend):
    #     self.loss = tf.reduce_mean(input_tensor=tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid))
    #
    #     self.loss = self.loss + self.reg_attend * loss_attend
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_cont):
    #     self.loss = tf.reduce_mean(input_tensor=tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid))
    #
    #     self.loss = self.loss + self.reg_contrast * loss_cont
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_attend, loss_contrast):
    #     self.loss = tf.reduce_mean(input_tensor=tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid))
    #
    #     self.loss = self.loss + self.reg_attend * loss_attend + self.reg_contrast * loss_contrast
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_attend, loss_contrast, loss_construct):
    #     self.loss = tf.reduce_mean(
    #         input_tensor=tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias,
    #                                                 tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb,
    #                                                 self.neg_num * self.batch_size, self.n_mid))
    #
    #     self.loss = self.loss + self.reg_attend * loss_attend + self.reg_contrast * loss_contrast + self.reg_construct * loss_construct
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
            self.lr: inps[4]
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, hist_item, hist_mask):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.mid_his_batch_ph: hist_item,
            self.mask: hist_mask,
            # self.mid_batch_ph: item_id
        })
        return user_embs

    def output_variable(self, sess, hist_item, hist_mask):
        item_his_eb, user_eb = sess.run([self.item_his_eb, self.user_eb],
            feed_dict={self.mid_his_batch_ph: hist_item,
                      self.mask: hist_mask
                      })
        return item_his_eb, user_eb

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)


class Model2(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len, flag="DNN"):
    # def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len, re_cont, flag="DNN"):
        self.batch_size = batch_size
        self.num_items = n_mid
        self.neg_num = 10
        self.dim = embedding_dim
        self.reg_LM = 0.1
        self.reg_cont_item = 0.03
        self.reg_cont_interest = 0.05
        self.reg_cont_readout_in = 0.1
        self.reg_clu_loss = 0.1
        self.reg_balance = 0.1
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.uid_batch = tf.placeholder(tf.int32, [None, ], name='user_id_batch')
        self.itemid_batch = tf.placeholder(tf.int32, [None, ], name='target_item_id_batch')
        self.his_itemid_batch = tf.placeholder(tf.int32, [None, seq_len], name='his_item_id_batch')
        self.mask = tf.placeholder(tf.float32, [None, seq_len], name='his_mask_batch')
        self.adj_matrix = tf.placeholder(tf.float32, [None, seq_len, seq_len + 2], name='item_adjacent_batch')
        self.timestamps = tf.placeholder(tf.int32, [None, seq_len], name='item_time_interval_batch')
        self.lr = tf.placeholder(tf.float64, [], name='learning_rate')

        mask = tf.expand_dims(self.mask, -1)

        with tf.variable_scope("item_embedding", reuse=None):
            self.item_id_embeddings_var = tf.get_variable("item_id_embedding_var", [self.num_items, self.dim],
                                                          trainable=True)
            self.item_id_embeddings_bias = tf.get_variable("bias_lookup_table", [self.num_items],
                                                           initializer=tf.zeros_initializer(), trainable=False)
            self.item_eb = tf.nn.embedding_lookup(params=self.item_id_embeddings_var, ids=self.itemid_batch)
            self.his_itemid_batch_embedded = tf.nn.embedding_lookup(params=self.item_id_embeddings_var,
                                                                    ids=self.his_itemid_batch)

            self.item_list_emb = tf.reshape(self.his_itemid_batch_embedded, [-1, seq_len, self.dim])
            self.item_list_emb *= mask

            absolute_pos = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(input=self.item_list_emb)[1]), 0),
                        [tf.shape(input=self.item_list_emb)[0], 1]),
                vocab_size=seq_len, num_units=hidden_size, scope="abs_pos", reuse=None)

            self.item_list_add_pos = self.item_list_emb + absolute_pos
            self.item_list_add_pos *= mask

            # self.time_matrix_emb = embedding(self.timestamps, vocab_size=time_span + 1, num_units=hidden_size,
            #                                  scope="time_matrix", reuse=None)
            # self.time_W = tf.get_variable("time_W_var", [hidden_size, 1], trainable=True)
            # time_matrix_attention = tf.reshape(
            #     tf.matmul(tf.reshape(self.time_matrix_emb, [-1, hidden_size]), self.time_W),
            #     [-1, seq_len, seq_len, 1])
            #
            # time_mask = tf.expand_dims(tf.tile(tf.expand_dims(self.mask, axis=1), [1, seq_len, 1]), axis=-1)
            # time_paddings = tf.ones_like(time_mask) * (-2 ** 32 + 1)
            #
            # time_matrix_attention = tf.where(tf.equal(time_mask, 0), time_paddings, time_matrix_attention)
            #
            # time_matrix_attention = tf.nn.softmax(time_matrix_attention, axis=-2)
            # time_matrix_attention = tf.transpose(a=time_matrix_attention, perm=[0, 1, 3, 2])
            # time_emb = tf.squeeze(tf.matmul(time_matrix_attention, self.time_matrix_emb), axis=2)
            #
            # self.item_list_add_pos_time = self.item_list_add_pos + time_emb
            #
            # self.item_list_add_pos_time *= mask

    # def build_sampled_softmax_loss(self, item_emb, user_emb):
    #     self.loss = tf.reduce_mean(input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
    #                                                           tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
    #                                                           self.neg_num * self.batch_size, self.num_items))
    #
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_balance):
    #     self.loss = tf.reduce_mean(input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
    #                                                           tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
    #                                                           self.neg_num * self.batch_size, self.num_items))
    #     self.loss += self.reg_balance * loss_balance
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_LM):
    #         self.loss = tf.reduce_mean(
    #             input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
    #                                                     tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
    #                                                     self.neg_num * self.batch_size, self.num_items))
    #
    #         self.loss = self.loss + self.reg_LM * loss_LM
    #         self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def build_sampled_softmax_loss(self, item_emb, user_emb, loss_cont_readout_in):
        self.loss = tf.reduce_mean(
            input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
                                                    tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
                                                    self.neg_num * self.batch_size, self.num_items))

        self.loss = self.loss + self.reg_cont_readout_in * loss_cont_readout_in
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_cont_item, loss_cont_interest):
    #     self.loss = tf.reduce_mean(
    #         input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
    #                                                 tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
    #                                                 self.neg_num * self.batch_size, self.num_items))
    #
    #     self.loss = self.loss + self.reg_cont_item * loss_cont_item + self.reg_cont_interest * loss_cont_interest
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_LM, loss_balance):
    #     self.loss = tf.reduce_mean(
    #         input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
    #                                                 tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
    #                                                 self.neg_num * self.batch_size, self.num_items))
    #
    #     self.loss = self.loss + self.reg_LM * loss_LM + self.reg_balance * loss_balance
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_cont_item_interest, loss_balance):
    #     self.loss = tf.reduce_mean(
    #         input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
    #                                                 tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
    #                                                 self.neg_num * self.batch_size, self.num_items))
    #
    #     self.loss = self.loss + self.reg_cont_item_interest * loss_cont_item_interest + self.reg_balance * loss_balance
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_LM, loss_cont):
    #     self.loss = tf.reduce_mean(
    #         input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
    #                                                 tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
    #                                                 self.neg_num * self.batch_size, self.num_items))
    #
    #     self.loss = self.loss + self.reg_cont * loss_cont + self.reg_LM * loss_LM
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_LM, loss_cont, loss_balance):
    #     self.loss = tf.reduce_mean(input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
    #                                                           tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
    #                                                           self.neg_num * self.batch_size, self.num_items))
    #
    #     self.loss = self.loss + self.reg_LM * loss_LM + self.reg_cont * loss_cont + self.reg_balance * loss_balance
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, inps):
        feed_dict = {self.uid_batch: inps[0], self.itemid_batch: inps[1],
                     self.his_itemid_batch: inps[2], self.mask: inps[3], self.lr: inps[4],
                     self.is_training: True}
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss


    def output_item(self, sess):
        item_embs = sess.run(self.item_id_embeddings_var)
        return item_embs

    def output_user(self, sess, hist_item, hist_mask):
        user_embs = sess.run(self.user_eb, feed_dict={self.his_itemid_batch: hist_item,
                                                      self.mask: hist_mask,
                                                      self.is_training: False})
        return user_embs

    def output_user_mip(self, sess, hist_item, time_matrix, hist_mask):
        user_embs = sess.run(self.user_eb, feed_dict={self.his_itemid_batch: hist_item,
                                                      self.mask: hist_mask,
                                                      self.is_training: False})
        return user_embs

    def output_variable(self, sess, hist_item, hist_mask):
        gat, X, mask_0, mask_1, mask_2, mask_3  = sess.run([self.graph_gat, self.item_list_emb,
                                     self.mask_0, self.mask_1, self.mask_2, self.mask_3],
                                    feed_dict={self.his_itemid_batch: hist_item,
                                               self.mask: hist_mask,
                                               self.is_training: False})
        return gat, X, mask_0, mask_1, mask_2, mask_3

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)


class Model_DNN(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_DNN, self).__init__(n_mid, embedding_dim, hidden_size,
                                        batch_size, seq_len, flag="DNN")

        masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(embedding_dim)], axis=-1)

        self.item_his_eb_mean = tf.reduce_sum(input_tensor=self.item_his_eb, axis=1) / (
                    tf.reduce_sum(input_tensor=tf.cast(masks, dtype=tf.float32), axis=1) + 1e-9)
        self.user_eb = tf.layers.dense(self.item_his_eb_mean, hidden_size, activation=None)   # [B, d]

        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)


class Model_GRU4REC(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_GRU4REC, self).__init__(n_mid, embedding_dim, hidden_size,
                                            batch_size, seq_len, flag="GRU4REC")
        with tf.name_scope('rnn_1'):
            self.sequence_length = self.mask_length
            rnn_outputs, final_state1 = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_his_eb,
                                                          sequence_length=self.sequence_length, dtype=tf.float32,
                                                          scope="gru1")

        self.user_eb = final_state1
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)


class CapsuleNetwork(tf.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def call(self, item_his_emb, item_eb, mask):
        with tf.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim, activation=None, bias_initializer=None)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim * self.num_interest, activation=None,
                                               bias_initializer=None)
            else:
                w = tf.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer())
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(input_tensor=w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(a=item_emb_hat, perm=[0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(
                tf.random.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(input_tensor=tf.square(interest_capsule), axis=-1, keepdims=True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(a=interest_capsule, perm=[0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(input_tensor=tf.square(interest_capsule), axis=-1, keepdims=True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.layers.dense(interest_capsule, self.dim, activation=tf.nn.relu, name='proj')

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]),
                                tf.argmax(input=atten, axis=1, output_type=tf.int32) + tf.range(
                                    tf.shape(input=item_his_emb)[0]) * self.num_interest)
        else:
            readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout


class Model_MIND(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, hard_readout=True,
                 relu_layer=True):
        super(Model_MIND, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="MIND")

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=0, num_interest=num_interest,
                                         hard_readout=hard_readout, relu_layer=relu_layer)
        self.user_eb, self.readout = capsule_network(item_his_emb, self.item_eb, self.mask)
        self.build_sampled_softmax_loss(self.item_eb, self.readout)


class Model_ComiRec_DR(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, hard_readout=True,
                 relu_layer=False):
        super(Model_ComiRec_DR, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len,
                                               flag="ComiRec_DR")

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=2, num_interest=num_interest,
                                         hard_readout=hard_readout, relu_layer=relu_layer)
        self.user_eb, self.readout = capsule_network(item_his_emb, self.item_eb, self.mask)
        self.build_sampled_softmax_loss(self.item_eb, self.readout)


class Model_ComiRec_SA(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=True):
        super(Model_ComiRec_SA, self).__init__(n_mid, embedding_dim, hidden_size,
                                               batch_size, seq_len, flag="ComiRec_SA")

        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim])

        if add_pos:
            self.position_embedding = \
                tf.get_variable(
                    shape=[1, seq_len, embedding_dim],
                    name='position_embedding')
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding,
                                                        [tf.shape(input=item_list_emb)[0], 1, 1])
        else:
            item_list_add_pos = item_list_emb

        num_heads = num_interest
        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, hidden_size * 4, activation=tf.nn.tanh)
            item_att_w = tf.layers.dense(item_hidden, num_heads, activation=None)
            item_att_w = tf.transpose(a=item_att_w, perm=[0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1])
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            item_att_w = tf.nn.softmax(item_att_w)

            interest_emb = tf.matmul(item_att_w, item_list_emb)

        self.user_eb = interest_emb

        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1))

        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]),
                            tf.argmax(input=atten, axis=1, output_type=tf.int32) + tf.range(
                                tf.shape(input=item_list_emb)[0]) * num_heads)

        self.build_sampled_softmax_loss(self.item_eb, readout)


# class Model_Set(Model2):
#     def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len, n_layers=1, n_head=1, relu_layer=False, hard_readout=True):
#         super(Model_Set, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len, flag="Set")
#         # item_his_emb = self.item_list_add_pos_time
#         item_his_emb = self.item_list_emb
#         self.n_layers = n_layers
#         self.n_head = n_head
#         self.num_interest = num_interest
#         self.dim = embedding_dim
#         self.relu_layer = relu_layer
#         self.hard_readout = hard_readout
#
#         # set_transformer_1 = SetTransformer(self.n_layers, self.n_head, hidden_size, embedding_dim, num_interest, layer_norm=True)
#         # set_transformer_2 = SetTransformer(self.n_layers + 1, self.n_head, hidden_size, embedding_dim, num_interest, layer_norm=True)
#         # set_transformer_3 = SetTransformer(self.n_layers + 2, self.n_head, hidden_size, embedding_dim, num_interest, layer_norm=True)
#         #
#         # interest_capsule_1 = set_transformer_1(item_his_emb)
#         # interest_capsule_2 = set_transformer_2(item_his_emb)
#         # interest_capsule_3 = set_transformer_3(item_his_emb)
#         # interest_capsule = (interest_capsule_1 * 4.5 + interest_capsule_2 * 2 + interest_capsule_3 * 1) / (4.5 + 2 + 1)
#
#         set_transformer = SetTransformer(self.n_layers, self.n_head, hidden_size, embedding_dim, num_interest, layer_norm=True)
#         interest_capsule = set_transformer(item_his_emb)
#         interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])
#
#         if self.relu_layer:
#             interest_capsule = tf.layers.dense(interest_capsule, self.dim, activation=tf.nn.relu, name='proj')
#         self.user_eb = interest_capsule
#         atten = tf.matmul(interest_capsule, tf.reshape(self.item_eb, [-1, self.dim, 1]))
#         atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))
#
#         if self.hard_readout:
#             readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]), tf.argmax(input=atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(input=item_his_emb)[0]) * self.num_interest)
#         else:
#             readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
#             readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])
#         self.build_sampled_softmax_loss(self.item_eb, readout)

def sa(X, mask, i, num_interest, hidden_size):
    seq_len = get_shape(X)[1]
    embedding_dim = get_shape(X)[2]
    item_list_emb = tf.reshape(X, [-1, seq_len, embedding_dim])

    add_pos = True
    if add_pos:
        position_embedding = \
            tf.get_variable(
                shape=[1, seq_len, embedding_dim],
                name='position_embedding' + str(i))
        item_list_add_pos = item_list_emb + tf.tile(position_embedding, [tf.shape(input=item_list_emb)[0], 1, 1])
    else:
        item_list_add_pos = item_list_emb
    # item_list_add_pos = item_list_emb
    num_heads = num_interest
    with tf.variable_scope("self_atten" + str(i), reuse=tf.AUTO_REUSE) as scope:
        item_hidden = tf.layers.dense(item_list_add_pos, hidden_size * 4, activation=tf.nn.tanh)
        item_att_w = tf.layers.dense(item_hidden, num_heads, activation=None)
        item_att_w = tf.transpose(a=item_att_w, perm=[0, 2, 1])

        atten_mask = tf.cast(tf.tile(tf.expand_dims(mask, axis=1), [1, num_heads, 1]), tf.float32)
        paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

        item_att_w_pad = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
        item_att_w_pad = tf.nn.softmax(item_att_w_pad)

        interest_emb = tf.matmul(item_att_w_pad, item_list_emb)

    return interest_emb


# def time_attention(time_matrix_emb, time_W, hidden_size, seq_len, mask):
#
#     time_matrix_attention = tf.reshape(tf.matmul(tf.reshape(time_matrix_emb, [-1, hidden_size]), time_W),
#                                        [-1, seq_len, seq_len, 1])
#
#     time_mask = tf.expand_dims(tf.tile(tf.expand_dims(mask, axis=1), [1, seq_len, 1]), axis=-1)
#     time_paddings = tf.ones_like(time_mask) * (-2 ** 32 + 1)
#
#     time_matrix_attention = tf.where(tf.equal(time_mask, 0), time_paddings, time_matrix_attention)
#
#     time_matrix_attention = tf.nn.softmax(time_matrix_attention, axis=-2)
#     time_matrix_attention = tf.transpose(a=time_matrix_attention, perm=[0, 1, 3, 2])
#     time_emb = tf.squeeze(tf.matmul(time_matrix_attention, time_matrix_emb), axis=2)
#
#     return time_emb


class Re4(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=True):
        super(Re4, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="Re4")

    # def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, reg_attend, reg_contrast, reg_construct, seq_len=256,  add_pos=True):
    #     super(Re4, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, reg_attend, reg_contrast, reg_construct, flag="Re4")

        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim])
        add_pos = True
        if add_pos:
            position_embedding = tf.get_variable(shape=[1, seq_len, embedding_dim], name='position_embedding')
            item_list_emb_pos = item_list_emb + tf.tile(position_embedding, [tf.shape(input=item_list_emb)[0], 1, 1])
        else:
            item_list_emb_pos = item_list_emb

        # self.W1 = tf.Variable(tf.random.normal([hidden_size*4, hidden_size]), trainable=True, name="W1")
        # self.W1_2 = tf.Variable(tf.random.normal([num_interest, hidden_size*4]), trainable=True, name="W1_2")
        # self.W2 = tf.Variable(tf.random.normal([hidden_size, hidden_size]), trainable=True, name="W2")
        # self.W3 = tf.Variable(tf.random.normal([hidden_size, hidden_size]), trainable=True, name="W3")
        # self.W3_2 = tf.Variable(tf.random.normal([seq_len, hidden_size]), trainable=True, name="W3_2")
        # self.W5 = tf.Variable(tf.random.normal([hidden_size, hidden_size]), trainable=True, name="W5")

        # multi-interest
        # proposals_weight = tf.matmul(self.W1_2, K.tanh(tf.matmul(self.W1, tf.transpose(item_list_emb_pos, [0, 2, 1]))))
        # atten_mask = tf.cast(tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_interest, 1]), tf.float32)
        # paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)
        #
        # proposals_weight_logits = tf.where(tf.equal(atten_mask, 0), paddings, proposals_weight)
        # proposals_weight = tf.nn.softmax(proposals_weight_logits, axis=2)
        # watch_interests = tf.matmul(proposals_weight, tf.matmul(item_list_emb, self.W2))

        num_heads = num_interest
        with tf.variable_scope("self_attention_1", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_emb_pos, hidden_size * 4, activation=tf.nn.tanh)
            proposals_weight = tf.layers.dense(item_hidden, num_heads, activation=None)
            proposals_weight = tf.transpose(a=proposals_weight, perm=[0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1])
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            proposals_weight_logits = tf.where(tf.equal(atten_mask, 0), paddings, proposals_weight)
            proposals_weight = tf.nn.softmax(proposals_weight_logits)

            watch_interests = tf.matmul(proposals_weight, item_list_emb)

        # re-attend
        product = tf.matmul(watch_interests, tf.transpose(item_list_emb, [0, 2, 1]))  # [B, k, L]
        product = tf.where(tf.equal(atten_mask, 0), paddings, product)
        re_att = tf.nn.softmax(product, axis=2)
        att_pred = tf.nn.log_softmax(proposals_weight_logits, axis=-1)
        loss_attend = -tf.reduce_sum(re_att * att_pred) / tf.reduce_sum(re_att)

        # re-contrast
        # X = item_list_emb_pos
        # self.gat_item = GraphAttention1(embedding_dim)
        # # X = self.item_list_add_pos_time
        # self.float_mask = tf.cast(self.mask, tf.float32)
        # self.real_sequence_length = tf.reduce_sum(self.mask, 1)
        # self.metric_heads_item = 1
        # self.relative_threshold_item = 0.5
        # with tf.name_scope('item_graph'):
        #     ## Node similarity metric learning
        #     S_item = []
        #     for i in range(self.metric_heads_item):
        #         # weighted cosine similarity
        #         self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), get_shape(X)[-1], use_bias=False)
        #         X_fts = X * tf.expand_dims(self.weighted_tensor, 0)
        #         X_fts = tf.nn.l2_normalize(X_fts, dim=2)
        #         S_item_one = tf.matmul(X_fts, tf.transpose(X_fts, (0, 2, 1)))  # B*L*L
        #         # min-max normalization for mask
        #         S_item_min = tf.reduce_min(S_item_one, -1, keepdims=True)
        #         S_item_max = tf.reduce_max(S_item_one, -1, keepdims=True)
        #         # S_one = (S_one - S_min) / ((S_max - S_min) + 1)
        #         S_item_one = (S_item_one - S_item_min) / (S_item_max - S_item_min)
        #         S_item_one = tf.where(tf.math.is_nan(S_item_one), tf.zeros_like(S_item_one), S_item_one)
        #         S_item += [S_item_one]
        #     S_item = tf.reduce_mean(tf.stack(S_item, 0), 0)
        #     # mask invalid nodes  S = B*L*L * B*L*1 * B*1*L = B*L*L
        #     S_item = S_item * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
        #
        #     ## Graph sparsification via seted sparseness
        #     S_flatten = tf.reshape(S_item, [get_shape(S_item)[0], -1])  # B*[L*L]
        #     sorted_S_flatten = tf.sort(S_flatten, direction='DESCENDING', axis=-1)  # B*L -> B*L
        #     # relative ranking strategy of the entire graph
        #     num_edges = tf.cast(tf.count_nonzero(S_item, [1, 2]), tf.float32)  # B
        #     to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.relative_threshold_item), tf.int32)
        #     threshold_score = tf.gather_nd(sorted_S_flatten,
        #                                    tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1),
        #                                    batch_dims=1)  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
        #     A_item = tf.cast(tf.greater(S_item, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)
        #
        #
        # self.w = tf.layers.dense(tf.ones([1, 1]), get_shape(X)[-1], use_bias=False)
        # X_clu = X * tf.expand_dims(self.w, 0)
        # mu = tf.slice(X_clu, [0, 0, 0], [get_shape(X)[0], num_interest, get_shape(X)[2]])
        # mu = mu * tf.expand_dims(self.w, 0)
        # mu_iter = tf.stop_gradient(mu, name='mu_iter')
        # # mu_iter = mu
        # max_iter = 3
        # c = 1e10
        # for _ in range(max_iter):
        #     sim = tf.matmul(X_clu, tf.transpose(mu_iter, [0, 2, 1]))  # [B, len, dim] * [B, k, dim] = [B, len, k]
        #     S = tf.arg_max(sim, dimension=2)  # [B, len]
        #     # S = tf.cast(tf.reduce_max(tf.nn.softmax(sim / c), axis=-1), tf.int32)  # [B, len]
        #     mu_temp = []
        #     for i in range(num_interest):
        #         X_i = tf.multiply(X_clu,
        #                           tf.cast(tf.equal(tf.tile(tf.expand_dims(S, axis=-1), [1, 1, embedding_dim]), i),
        #                                   tf.float32))
        #         count_i = tf.tile(tf.expand_dims(tf.reduce_sum(tf.cast(tf.equal(S, i), tf.float32), axis=1), axis=-1),
        #                           [1, embedding_dim])
        #         count_i = tf.where(tf.equal(count_i, 0), tf.ones_like(count_i), count_i)
        #         mu_i = tf.reduce_sum(X_i, axis=1) / count_i  # [B, dim]
        #         mu_temp += [mu_i]
        #     mu_iter = tf.stack(mu_temp, axis=1)
        #
        # A_item = tf.ones((get_shape(X)[0], seq_len, seq_len))
        # user_eb = []
        # self.graph_gat = 0
        # for i in range(num_interest):
        #     mask_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
        #     t = tf.expand_dims(tf.equal(S, i), axis=-1)
        #     A_i = tf.multiply(tf.cast(tf.logical_and(t, tf.transpose(t, perm=[0, 2, 1])), tf.float32), A_item)
        #     graph_gat_i = self.gat_item([X, A_i])
        #     gat_mask_i = tf.tile(tf.expand_dims(mask_i, axis=-1), [1, 1, hidden_size])
        #     paddings_i = tf.cast(tf.zeros_like(gat_mask_i), tf.float32)
        #     graph_gat_i = tf.where(tf.equal(gat_mask_i, 0), paddings_i, graph_gat_i)
        #     # user_eb_i = sa(X, mask_i, i + 1, num_interest=1, hidden_size=hidden_size)  # [B, 1, d]
        #     # user_eb += [user_eb_i]
        #     self.graph_gat += graph_gat_i
        #
        # self.graph_gat = tf.nn.l2_normalize(self.graph_gat, dim=2)  # [B, len, dim]
        # watch_interests = tf.nn.l2_normalize(watch_interests, dim=2)
        #
        # self.loss_cont = 0
        # self.matrix2 = tf.exp(tf.matmul(self.graph_gat, tf.transpose(watch_interests, [0, 2, 1])))  # [B, len, k]
        # # self.matrix2 = tf.matmul(gat_norm, tf.transpose(self.user_eb, [0, 2, 1]))   # [B, len, k]
        # for i in range(num_interest):
        #     mask_pos_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
        #     matrix_exp = tf.expand_dims(self.matrix2[:, :, i], axis=-1)
        #     mask_neg_i = self.mask
        #     mask_pos_i = tf.expand_dims(mask_pos_i, axis=-1)
        #     mask_neg_i = tf.expand_dims(mask_neg_i, axis=-1)
        #     cont_pos = tf.squeeze(tf.multiply(matrix_exp, mask_pos_i)) + 1e-8  # [B, len]
        #     cont_neg = tf.squeeze(tf.multiply(matrix_exp, mask_neg_i)) + 1e-8  # [B, len]
        #     self.loss_cont += -tf.reduce_mean(
        #         tf.log(cont_pos / tf.tile(tf.expand_dims(tf.reduce_sum(cont_neg, axis=-1), axis=-1), [1, seq_len])))

        # t_cont = 1
        # # A_item = tf.ones((get_shape(X)[0], seq_len, seq_len))
        # watch_interests_norm = tf.nn.l2_normalize(watch_interests, dim=-1)
        # item_list_emb_norm = tf.nn.l2_normalize(item_list_emb, dim=-1)
        # cos_sim = tf.matmul(watch_interests_norm, tf.transpose(item_list_emb_norm, [0, 2, 1]))  # [B, k, L]
        # gate = tf.repeat(1 / (self.mask_length * 1.0), seq_len, axis=0)  # 两种方案
        # gate = tf.reshape(gate, [get_shape(item_list_emb)[0], 1, seq_len])  # [B, 1, L]
        # positive_weight_idx = tf.cast(tf.greater(proposals_weight, gate), tf.float32)
        #
        # cos = tf.where_v2(tf.greater(cos_sim, tf.tile(gate, [1, num_interest, 1])), tf.ones_like(cos_sim), tf.zeros_like(cos_sim))
        # # self.a = cos_i = cos[:, i, :]
        # # print("hsm" * 100)
        # # print(self.a)
        # self.graph_gat = 0
        # for i in range(num_interest):
        #     cos_i = cos[:, i, :]    # [B, L]  mask_i
        #     cos_i = tf.expand_dims(cos_i, axis=1)   # [B, 1, L]
        #     A_i = tf.multiply(tf.matmul(cos_i, tf.transpose(cos_i, [0, 2, 1])), A_item)
        #     graph_gat_i = self.gat_item([X, A_i])
        #     gat_mask_i = tf.tile(tf.transpose(cos_i, [0, 2, 1]), [1, 1, hidden_size])
        #     paddings_i = tf.cast(tf.zeros_like(gat_mask_i), tf.float32)
        #     graph_gat_i = tf.where(tf.equal(gat_mask_i, 0), paddings_i, graph_gat_i)
        #     self.graph_gat += graph_gat_i
        #
        # self.graph_gat = tf.nn.l2_normalize(self.graph_gat, dim=2)  # [B, len, dim]
        # self.user_eb = tf.nn.l2_normalize(watch_interests, dim=2)
        # self.loss_cont = 0
        # self.matrix = tf.exp(tf.matmul(self.graph_gat, tf.transpose(watch_interests, [0, 2, 1])))  # [B, len, k]
        # for i in range(num_interest):
        #     mask_pos_i = tf.cast(cos[:, i, :], tf.float32)
        #     matrix_exp = tf.expand_dims(self.matrix[:, :, i], axis=-1)
        #     mask_neg_i = self.mask
        #     mask_pos_i = tf.expand_dims(mask_pos_i, axis=-1)
        #     mask_neg_i = tf.expand_dims(mask_neg_i, axis=-1)
        #     cont_pos = tf.squeeze(tf.multiply(matrix_exp, mask_pos_i)) + 1e-8  # [B, len]
        #     cont_neg = tf.squeeze(tf.multiply(matrix_exp, mask_neg_i)) + 1e-8  # [B, len]
        #     self.loss_cont += -tf.reduce_mean(
        #         tf.log(cont_pos / tf.tile(tf.expand_dims(tf.reduce_sum(cont_neg, axis=-1), axis=-1), [1, seq_len])))

        # loss_contrast = self.loss_cont

        t_cont = 1
        watch_interests_norm = tf.nn.l2_normalize(watch_interests, dim=-1)
        item_list_emb_norm = tf.nn.l2_normalize(item_list_emb, dim=-1)
        cos_sim = tf.matmul(watch_interests_norm, tf.transpose(item_list_emb_norm, [0, 2, 1]))  # [B, k, L]
        gate = tf.repeat(1 / (self.mask_length * 1.0), seq_len, axis=0)  # 两种方案
        gate = tf.reshape(gate, [get_shape(item_list_emb)[0], 1, seq_len])  # [B, 1, L]
        positive_weight_idx = tf.cast(tf.greater(proposals_weight, gate), tf.float32)
        mask_cos = tf.where(tf.equal(atten_mask, 0), paddings, cos_sim)
        pos_cos = tf.where(tf.equal(positive_weight_idx, 1), mask_cos, paddings)

        cons_pos = tf.exp(pos_cos / t_cont)
        cons_neg = tf.reduce_sum(tf.exp(mask_cos / t_cont), axis=2)

        in2in = tf.matmul(watch_interests_norm, tf.transpose(watch_interests_norm, [0, 2, 1]))  # [B, k, k]
        paddings = tf.ones_like(in2in) * (-2 ** 32 + 1)
        in2in = tf.where(tf.equal(tf.tile(tf.expand_dims(tf.eye(num_interest), axis=0),
                                          [get_shape(watch_interests)[0], 1, 1]), 1), paddings, in2in)
        cons_neg = cons_neg + tf.reduce_sum(tf.exp(in2in / t_cont), axis=2)

        item_rolled = tf.roll(item_list_emb_norm, 1, axis=0)
        in2i = tf.matmul(watch_interests_norm, tf.transpose(item_rolled, [0, 2, 1]))  # [B, k, d] * [B, d, L]
        in2i_mask = tf.roll(tf.equal(self.mid_his_batch_ph, 0), 1, axis=0)  # [B, L]
        paddings = tf.ones_like(in2i) * (-2 ** 32 + 1)
        in2i = tf.where(tf.tile(tf.expand_dims(in2i_mask, axis=1), [1, num_interest, 1]), paddings, in2i)
        cons_neg = cons_neg + tf.reduce_sum(tf.exp(in2i / t_cont), axis=2)

        cons_div = cons_pos / tf.expand_dims(cons_neg, axis=-1)
        cons_div = tf.where(tf.equal(tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_interest, 1]), 0),
                            tf.ones_like(cons_div), cons_div)
        cons_div = tf.where(tf.equal(positive_weight_idx, 1), cons_div, tf.ones_like(cons_div))

        loss_contrast = -tf.log(cons_div)
        loss_contrast = tf.reduce_mean(loss_contrast)

        # re-construct
        recons_item = tf.layers.dense(watch_interests, hidden_size * seq_len)  # [B, k, d*L]
        recons_item = tf.reshape(recons_item,
                                 [get_shape(watch_interests)[0] * num_interest, seq_len, embedding_dim])  # [B*k, L, d]

        with tf.variable_scope("self_attention_2", reuse=tf.AUTO_REUSE) as scope:
            recons_hidden = tf.layers.dense(recons_item, hidden_size, activation=tf.nn.tanh)
            recons_weight = tf.layers.dense(recons_hidden, seq_len, activation=None)
            recons_weight = tf.transpose(a=recons_weight, perm=[0, 2, 1])

            recons_weight = tf.reshape(recons_weight, [get_shape(watch_interests)[0], num_interest, seq_len, seq_len])
            paddings = tf.ones_like(recons_weight) * (-2 ** 32 + 1)

            recons_weight_logits = tf.reshape(tf.where(tf.tile(tf.reshape(tf.equal(self.mid_his_batch_ph, 0),
                                                                          [get_shape(watch_interests)[0], 1, 1,
                                                                           seq_len]), [1, num_interest, seq_len, 1]),
                                                       paddings, recons_weight), [-1, seq_len, seq_len])
            recons_weight = tf.nn.softmax(recons_weight_logits, axis=-1)

            recons_item = tf.reshape(tf.matmul(recons_weight, recons_item),
                                     [get_shape(watch_interests)[0], num_interest, seq_len, -1])

        target_emb = tf.tile(tf.expand_dims(item_list_emb, axis=1), [1, num_interest, 1, 1])
        loss_construct = tf.square(recons_item - target_emb)  # [B, k, L, d]
        loss_construct = tf.where(tf.tile(tf.expand_dims(tf.equal(positive_weight_idx, 0), axis=-1),
                                          [1, 1, 1, hidden_size]), tf.zeros_like(loss_construct), loss_construct)
        loss_construct = tf.where(tf.equal(tf.tile(tf.expand_dims(tf.expand_dims(self.mask, axis=1), axis=-1),
                                                   [1, num_interest, 1, hidden_size]), 0),
                                  tf.zeros_like(loss_construct), loss_construct)
        loss_construct = tf.reduce_mean(loss_construct)

        # recons_weight = tf.matmul(self.W3_2,     # [B, L, d] * ([B, d, d] * [B*k, d, L])
        #                              K.tanh(tf.matmul(self.W3, tf.transpose(recons_item, [0, 2, 1]))))  # [B*k, L, L]
        # recons_weight = tf.reshape(recons_weight, [get_shape(watch_interests)[0], num_interest, seq_len, seq_len])
        # paddings = tf.ones_like(recons_weight) * (-2 ** 32 + 1)
        # recons_weight = tf.reshape(tf.where(tf.tile(tf.reshape(tf.equal(self.mid_his_batch_ph, 0),
        #                 [get_shape(watch_interests)[0], 1, 1, seq_len]), [1, num_interest, seq_len, 1]),
        #                                     paddings, recons_weight), [-1, seq_len, seq_len])
        # recons_weight = tf.nn.softmax(recons_weight, axis=-1)
        # recons_item = tf.reshape(tf.matmul(recons_weight, tf.matmul(recons_item, self.W5)), [get_shape(watch_interests)[0],
        #                                                                                      num_interest, seq_len, -1])

        # watch_interests = K.tanh(tf.layers.dense(watch_interests, embedding_dim, activation=None))
        self.user_eb = watch_interests

        num_heads = num_interest
        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1))

        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]),
                            tf.argmax(input=atten, axis=1, output_type=tf.int32) + tf.range(
                                tf.shape(input=item_list_emb)[0]) * num_heads)

        self.build_sampled_softmax_loss(self.item_eb, readout, loss_attend, loss_contrast, loss_construct)
        # self.build_sampled_softmax_loss(self.item_eb, readout, loss_contrast)


# class SURGE(Model):
#     def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
#         super(SURGE, self).__init__(n_mid, embedding_dim, hidden_size,
#                                            batch_size, seq_len, flag="SURGE")
#         """Initialization of variables or temp hyperparameters
#         Args:
#             hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
#             iterator_creator (obj): An iterator to load the data.
#         """
#         self.relative_threshold = 0.5
#         self.metric_heads = 1
#         self.attention_heads = 1
#         self.pool_layers = 1
#         self.layer_shared = True
#         self.pool_length = 10
#         self.hidden_size = hidden_size
#         self.seq_len = seq_len
#
#         X = self.item_his_eb
#         self.float_mask = tf.cast(self.mask, tf.float32)
#         self.real_sequence_length = tf.reduce_sum(self.mask, 1)
#
#         with tf.name_scope('interest_graph'):
#             ## Node similarity metric learning
#             S = []
#             for i in range(self.metric_heads):
#                 # weighted cosine similarity
#                 self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), get_shape(X)[-1], use_bias=False)
#                 X_fts = X * tf.expand_dims(self.weighted_tensor, 0)
#                 X_fts = tf.nn.l2_normalize(X_fts, dim=2)
#                 S_one = tf.matmul(X_fts, tf.transpose(X_fts, (0, 2, 1)))  # B*L*L
#                 # min-max normalization for mask
#                 S_min = tf.reduce_min(S_one, -1, keepdims=True)
#                 S_max = tf.reduce_max(S_one, -1, keepdims=True)
#                 S_one = (S_one - S_min) / (S_max - S_min)
#                 S += [S_one]
#             S = tf.reduce_mean(tf.stack(S, 0), 0)
#             # mask invalid nodes  S = B*L*L * B*L*1 * B*1*L = B*L*L
#             S = S * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
#
#             ## Graph sparsification via seted sparseness
#             S_flatten = tf.reshape(S, [get_shape(S)[0], -1])  # B*[L*L]
#             sorted_S_flatten = tf.sort(S_flatten, direction='DESCENDING', axis=-1)  # B*L -> B*L
#             # relative ranking strategy of the entire graph
#             num_edges = tf.cast(tf.count_nonzero(S, [1, 2]), tf.float32)  # B
#             to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.relative_threshold), tf.int32)
#             threshold_score = tf.gather_nd(sorted_S_flatten,
#                                            tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1),
#                                            batch_dims=1)  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
#             A = tf.cast(tf.greater(S, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)
#
#         with tf.name_scope('interest_fusion_extraction'):
#             for l in range(self.pool_layers):
#                 reuse = False if l==0 else True
#                 X, A, graph_readout, alphas = self._interest_fusion_extraction(X, A, layer=l, reuse=reuse)
#
#         with tf.name_scope('prediction'):
#             # flatten pooled graph to reduced sequence
#             output_shape = self.mask.get_shape()
#             sorted_mask_index = tf.argsort(self.mask, direction='DESCENDING', stable=True, axis=-1) # B*L -> B*L
#             sorted_mask = tf.sort(self.mask, direction='DESCENDING', axis=-1)  # B*L -> B*L
#             sorted_mask.set_shape(output_shape)
#             sorted_mask_index.set_shape(output_shape)
#             X = tf.batch_gather(X, sorted_mask_index)  # B*L*F  < B*L = B*L*F
#             mask = sorted_mask
#             self.reduced_sequence_length = tf.reduce_sum(mask, 1)  # B
#
#             # cut useless sequence tail per batch
#             self.to_max_length = tf.cast(tf.range(tf.reduce_max(self.reduced_sequence_length)), tf.int32)  # l
#             X = tf.gather(X, self.to_max_length, axis=1)  # B*L*F -> B*l*F
#             mask = tf.gather(mask, self.to_max_length, axis=1)  # B*L -> B*l
#             self.reduced_sequence_length = tf.reduce_sum(mask, 1)  # B
#
#             # use cluster score as attention weights in AUGRU
#             _, alphas = self._attention_fcn(self.item_eb, X, 'AGRU', False, return_alpha=True)
#             _, final_state = dynamic_rnn_dien(
#                 VecAttGRUCell(self.hidden_size),
#                 inputs=X,
#                 att_scores=alphas,
#                 sequence_length=self.reduced_sequence_length,
#                 dtype=tf.float32,
#                 scope="gru"
#             )
#             model_output = tf.concat([final_state, graph_readout, self.item_eb, graph_readout*self.item_eb], 1)
#             self.user_eb = tf.layers.dense(model_output, self.hidden_size)
#
#             self.build_sampled_softmax_loss(self.item_eb, self.user_eb)
#
#
#     def _attention_fcn(self, query, key_value, name, reuse, return_alpha=False):
#         """Apply attention by fully connected layers.
#         Args:
#             query (obj): The embedding of target item or cluster which is regarded as a query in attention operations.
#             key_value (obj): The embedding of history items which is regarded as keys or values in attention operations.
#             name (obj): The name of variable W
#             reuse (obj): Reusing variable W in query operation
#             return_alpha (obj): Returning attention weights
#         Returns:
#             output (obj): Weighted sum of value embedding.
#             att_weights (obj):  Attention weights
#         """
#         with tf.variable_scope("attention_fcn"+str(name), reuse=reuse):
#             query_size = query.shape[-1].value            # F
#             boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))
#
#             attention_mat = tf.get_variable(
#                 name="attention_mat"+str(name),
#                 shape=[key_value.shape.as_list()[-1], query_size],
#                 initializer=tf.truncated_normal_initializer(),
#             )   # F * F
#             att_inputs = tf.tensordot(key_value, attention_mat, [[2], [0]]) # [B, L, F]
#
#             if query.shape.ndims != att_inputs.shape.ndims:       # True---->query = self.item_eb
#                 queries = tf.tile(tf.expand_dims(query, axis=1), [1, self.seq_len, 1])
#             else:
#                 queries = query
#
#             last_hidden_nn_layer = tf.concat(
#                 [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1)   # [B, L, 4*F]
#
#             att_fnc_output = tf.layers.dense(last_hidden_nn_layer, self.hidden_size, activation=tf.nn.relu)
#             curr_hidden_nn_layer = tf.layers.batch_normalization(
#                 att_fnc_output,
#                 momentum=0.95,
#                 epsilon=0.0001,
#             )
#             att_fnc_output = tf.layers.dense(curr_hidden_nn_layer, 1, activation=tf.nn.relu)  # [B, L, 1]
#             # att_fnc_output = tf.squeeze(att_fnc_output, -1)
#             mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
#             boolean_mask = tf.expand_dims(boolean_mask, -1)
#             att_weights = tf.nn.softmax(
#                 tf.where(boolean_mask, att_fnc_output, mask_paddings),
#                 name="att_weights",
#             )
#
#             output = key_value * tf.tile(att_weights, [1, 1, self.hidden_size])   # [B, L, F]
#             if not return_alpha:
#                 return output
#             else:
#                 return output, att_weights
#
#
#     def _interest_fusion_extraction(self, X, A, layer, reuse):
#         """Interest fusion and extraction via graph convolution and graph pooling
#         Args:
#             X (obj): Node embedding of graph
#             A (obj): Adjacency matrix of graph
#             layer (obj): Interest fusion and extraction layer
#             reuse (obj): Reusing variable W in query operation
#         Returns:
#             X (obj): Aggerated cluster embedding
#             A (obj): Pooled adjacency matrix
#             graph_readout (obj): Readout embedding after graph pooling
#             cluster_score (obj): Cluster score for AUGRU in prediction layer
#         """
#         with tf.name_scope('interest_fusion'):
#             ## cluster embedding
#             A_bool = tf.cast(tf.greater(A, 0), A.dtype)
#             A_bool = A_bool * (tf.ones([A.shape.as_list()[1], A.shape.as_list()[1]]) - tf.eye(A.shape.as_list()[1])) + tf.eye(A.shape.as_list()[1])
#             D = tf.reduce_sum(A_bool, axis=-1)  # B*L
#             D = tf.sqrt(D)[:, None] + K.epsilon()  # B*1*L
#             A = (A_bool / D) / tf.transpose(D, perm=(0, 2, 1))  # B*L*L / B*1*L / B*L*1 归一化邻接矩阵
#             X_q = tf.matmul(A, tf.matmul(A, X))  # B*L*F      B*L*L * (B*L*L * B*L*F)
#
#             Xc = []
#             for i in range(self.attention_heads):
#                 ## cluster- and query-aware attention
#                 if not self.layer_shared:
#                     _, f_1 = self._attention_fcn(X_q, X, 'f1_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
#                     _, f_2 = self._attention_fcn(self.item_eb, X, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
#                 if self.layer_shared:
#                     _, f_1 = self._attention_fcn(X_q, X, 'f1_shared'+'_'+str(i), reuse, return_alpha=True)
#                     _, f_2 = self._attention_fcn(self.item_eb, X, 'f2_shared'+'_'+str(i), reuse, return_alpha=True)
#
#                 ## graph attentive convolution
#                 E = A_bool * f_1 + A_bool * tf.transpose(f_2, (0, 2, 1)) # B*L*1 x B*L*1 -> B*L*L
#                 E = tf.nn.leaky_relu(E)
#                 boolean_mask = tf.equal(A_bool, tf.ones_like(A_bool))
#                 mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1)
#                 E = tf.nn.softmax(tf.where(boolean_mask, E, mask_paddings), axis=-1)
#                 Xc_one = tf.matmul(E, X)  # B*L*L x B*L*F -> B*L*F
#                 Xc_one = tf.layers.dense(Xc_one, self.hidden_size, use_bias=False)
#                 Xc_one += X
#                 Xc += [tf.nn.leaky_relu(Xc_one)]
#             Xc = tf.reduce_mean(tf.stack(Xc, 0), 0)
#
#         with tf.name_scope('interest_extraction'):
#             ## cluster fitness score
#             X_q = tf.matmul(A, tf.matmul(A, Xc))  # B*L*F
#             cluster_score = []
#             for i in range(self.attention_heads):
#                 if not self.layer_shared:
#                     _, f_1 = self._attention_fcn(X_q, Xc, 'f1_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
#                     _, f_2 = self._attention_fcn(self.item_eb, Xc, 'f2_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
#                 if self.layer_shared:
#                     _, f_1 = self._attention_fcn(X_q, Xc, 'f1_shared'+'_'+str(i), True, return_alpha=True)
#                     _, f_2 = self._attention_fcn(self.item_eb, Xc, 'f2_shared'+'_'+str(i), True, return_alpha=True)
#                 cluster_score += [f_1 + f_2]
#
#             cluster_score = tf.reduce_mean(tf.stack(cluster_score, 0), 0)
#             boolean_mask = tf.expand_dims(tf.equal(self.mask, tf.ones_like(self.mask)), axis=-1)
#             mask_paddings = tf.ones_like(cluster_score) * (-(2 ** 32) + 1)
#             cluster_score = tf.nn.softmax(tf.where(boolean_mask, cluster_score, mask_paddings), axis=-1)
#
#             ## graph pooling
#             num_nodes = tf.reduce_sum(self.mask, 1)  # B
#             boolean_pool = tf.greater(num_nodes, self.pool_length)  # B
#             to_keep = tf.where(boolean_pool,
#                                tf.cast(self.pool_length + (self.real_sequence_length - self.pool_length)/self.pool_layers*(self.pool_layers-layer-1), tf.float32),
#                                num_nodes)  # B
#
#             cluster_score = tf.squeeze(cluster_score, axis=-1) * self.float_mask  # B*L
#             sorted_score = tf.sort(cluster_score, direction='DESCENDING', axis=-1)  # B*L
#             target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1), batch_dims=1) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B)
#             topk_mask = tf.greater(cluster_score, tf.expand_dims(target_score, -1))  # B*L + B*1 -> B*L
#             mask = tf.cast(topk_mask, tf.int32)
#             self.float_mask = tf.cast(mask, tf.float32)
#             self.reduced_sequence_length = tf.reduce_sum(mask, 1)
#
#             ## ensure graph connectivity
#
#             E = E * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
#             A = tf.matmul(tf.matmul(E, A_bool), tf.transpose(E, (0, 2, 1)))  # B*C*L x B*L*L x B*L*C = B*C*C
#             ## graph readout
#             graph_readout = tf.reduce_sum(Xc*tf.expand_dims(cluster_score, -1)*tf.expand_dims(self.float_mask, -1), 1)
#
#         return Xc, A, graph_readout, cluster_score

class SURGE(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(SURGE, self).__init__(n_mid, embedding_dim, hidden_size,
                                    batch_size, seq_len, flag="SURGE")
        """Initialization of variables or temp hyperparameters
        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
        """
        self.relative_threshold = 0.5
        self.metric_heads = 1
        self.attention_heads = 1
        self.pool_layers = 1
        self.layer_shared = True
        self.pool_length = 10
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        X = self.item_his_eb
        self.float_mask = tf.cast(self.mask, tf.float32)
        self.real_sequence_length = tf.reduce_sum(self.mask, 1)

        with tf.name_scope('interest_graph'):
            ## Node similarity metric learning
            S = []
            for i in range(self.metric_heads):
                # weighted cosine similarity
                self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), get_shape(X)[-1], use_bias=False)
                X_fts = X * tf.expand_dims(self.weighted_tensor, 0)
                X_fts = tf.nn.l2_normalize(X_fts, dim=2)
                S_one = tf.matmul(X_fts, tf.transpose(X_fts, (0, 2, 1)))  # B*L*L
                # min-max normalization for mask
                S_min = tf.reduce_min(S_one, -1, keepdims=True)
                S_max = tf.reduce_max(S_one, -1, keepdims=True)
                S_one = (S_one - S_min) / (S_max - S_min)
                S_one = tf.where(tf.math.is_nan(S_one), tf.zeros_like(S_one), S_one)
                S += [S_one]
            S = tf.reduce_mean(tf.stack(S, 0), 0)
            # mask invalid nodes  S = B*L*L * B*L*1 * B*1*L = B*L*L
            S = S * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)

            ## Graph sparsification via seted sparseness
            S_flatten = tf.reshape(S, [get_shape(S)[0], -1])  # B*[L*L]
            sorted_S_flatten = tf.sort(S_flatten, direction='DESCENDING', axis=-1)  # B*L -> B*L
            # relative ranking strategy of the entire graph
            num_edges = tf.cast(tf.count_nonzero(S, [1, 2]), tf.float32)  # B
            to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.relative_threshold), tf.int32)
            threshold_score = tf.gather_nd(sorted_S_flatten,
                                           tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1),
                                           batch_dims=1)  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            A = tf.cast(tf.greater(S, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)

        with tf.name_scope('interest_fusion_extraction'):
            for l in range(self.pool_layers):
                reuse = False if l == 0 else True
                X, A, graph_readout, alphas = self._interest_fusion_extraction(X, A, layer=l, reuse=reuse)

        with tf.name_scope('prediction'):
            # flatten pooled graph to reduced sequence
            output_shape = self.mask.get_shape()
            sorted_mask_index = tf.argsort(self.mask, direction='DESCENDING', stable=True, axis=-1)  # B*L -> B*L
            sorted_mask = tf.sort(self.mask, direction='DESCENDING', axis=-1)  # B*L -> B*L
            sorted_mask.set_shape(output_shape)
            sorted_mask_index.set_shape(output_shape)
            X = tf.batch_gather(X, sorted_mask_index)  # B*L*F  < B*L = B*L*F
            mask = sorted_mask
            self.reduced_sequence_length = tf.reduce_sum(mask, 1)  # B

            # cut useless sequence tail per batch
            self.to_max_length = tf.cast(tf.range(tf.reduce_max(self.reduced_sequence_length)), tf.int32)  # l
            X = tf.gather(X, self.to_max_length, axis=1)  # B*L*F -> B*l*F
            mask = tf.gather(mask, self.to_max_length, axis=1)  # B*L -> B*l
            self.reduced_sequence_length = tf.reduce_sum(mask, 1)  # B

            # use cluster score as attention weights in AUGRU
            # last_item = tf.reduce_mean(X, axis=1)
            # _, alphas = self._attention_fcn(last_item, X, 'AGRU', False, return_alpha=True)   # [B, seq_len, 1]
            alphas = tf.expand_dims(alphas, axis=-1)
            _, final_state = dynamic_rnn_dien(
                VecAttGRUCell(self.hidden_size),
                inputs=X,
                att_scores=alphas,
                sequence_length=self.reduced_sequence_length,
                dtype=tf.float32,
                scope="gru"
            )
            model_output = tf.concat([final_state, graph_readout], 1)
            self.user_eb = tf.layers.dense(model_output, self.hidden_size)

            self.build_sampled_softmax_loss(self.item_eb, self.user_eb)

    def _attention_fcn(self, query, key_value, name, reuse, return_alpha=False):
        """Apply attention by fully connected layers.
        Args:
            query (obj): The embedding of target item or cluster which is regarded as a query in attention operations.
            key_value (obj): The embedding of history items which is regarded as keys or values in attention operations.
            name (obj): The name of variable W
            reuse (obj): Reusing variable W in query operation
            return_alpha (obj): Returning attention weights
        Returns:
            output (obj): Weighted sum of value embedding.
            att_weights (obj):  Attention weights
        """
        with tf.variable_scope("attention_fcn" + str(name), reuse=reuse):
            query_size = query.shape[-1].value  # F
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_mat = tf.get_variable(
                name="attention_mat" + str(name),
                shape=[key_value.shape.as_list()[-1], query_size],
                initializer=tf.truncated_normal_initializer(),
            )  # F * F
            att_inputs = tf.tensordot(key_value, attention_mat, [[2], [0]])  # [B, L, F]

            if query.shape.ndims != att_inputs.shape.ndims:  # True---->query = self.item_eb
                queries = tf.tile(tf.expand_dims(query, axis=1), [1, self.seq_len, 1])
            else:
                queries = query

            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1)  # [B, L, 4*F]

            att_fnc_output = tf.layers.dense(last_hidden_nn_layer, self.hidden_size, activation=tf.nn.relu)
            curr_hidden_nn_layer = tf.layers.batch_normalization(
                att_fnc_output,
                momentum=0.95,
                epsilon=0.0001,
            )
            att_fnc_output = tf.layers.dense(curr_hidden_nn_layer, 1, activation=tf.nn.relu)  # [B, L, 1]
            # att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            boolean_mask = tf.expand_dims(boolean_mask, -1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )

            output = key_value * tf.tile(att_weights, [1, 1, self.hidden_size])  # [B, L, F]
            if not return_alpha:
                return output
            else:
                return output, att_weights

    def _interest_fusion_extraction(self, X, A, layer, reuse):
        """Interest fusion and extraction via graph convolution and graph pooling
        Args:
            X (obj): Node embedding of graph
            A (obj): Adjacency matrix of graph
            layer (obj): Interest fusion and extraction layer
            reuse (obj): Reusing variable W in query operation
        Returns:
            X (obj): Aggerated cluster embedding
            A (obj): Pooled adjacency matrix
            graph_readout (obj): Readout embedding after graph pooling
            cluster_score (obj): Cluster score for AUGRU in prediction layer
        """
        with tf.name_scope('interest_fusion'):
            ## cluster embedding
            A_bool = tf.cast(tf.greater(A, 0), A.dtype)
            A_bool = A_bool * (
                        tf.ones([A.shape.as_list()[1], A.shape.as_list()[1]]) - tf.eye(A.shape.as_list()[1])) + tf.eye(
                A.shape.as_list()[1])
            D = tf.reduce_sum(A_bool, axis=-1)  # B*L
            D = tf.sqrt(D)[:, None] + K.epsilon()  # B*1*L
            A = (A_bool / D) / tf.transpose(D, perm=(0, 2, 1))  # B*L*L / B*1*L / B*L*1 归一化邻接矩阵
            X_q = tf.matmul(A, tf.matmul(A, X))  # B*L*F      B*L*L * (B*L*L * B*L*F)

            Xc = []
            for i in range(self.attention_heads):
                ## cluster- and query-aware attention
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, X, 'f1_layer_' + str(layer) + '_' + str(i), False,
                                                 return_alpha=True)
                    # _, f_2 = self._attention_fcn(self.item_eb, X, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, X, 'f1_shared' + '_' + str(i), reuse, return_alpha=True)
                    # _, f_2 = self._attention_fcn(self.item_eb, X, 'f2_shared'+'_'+str(i), reuse, return_alpha=True)

                ## graph attentive convolution
                # E = A_bool * f_1 + A_bool * tf.transpose(f_2, (0, 2, 1)) # B*L*1 x B*L*1 -> B*L*L
                E = A_bool * f_1 + A_bool
                E = tf.nn.leaky_relu(E)
                boolean_mask = tf.equal(A_bool, tf.ones_like(A_bool))
                mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1)
                E = tf.nn.softmax(tf.where(boolean_mask, E, mask_paddings), axis=-1)
                Xc_one = tf.matmul(E, X)  # B*L*L x B*L*F -> B*L*F
                Xc_one = tf.layers.dense(Xc_one, self.hidden_size, use_bias=False)
                Xc_one += X
                Xc += [tf.nn.leaky_relu(Xc_one)]
            Xc = tf.reduce_mean(tf.stack(Xc, 0), 0)

        with tf.name_scope('interest_extraction'):
            ## cluster fitness score
            X_q = tf.matmul(A, tf.matmul(A, Xc))  # B*L*F
            cluster_score = []
            for i in range(self.attention_heads):
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, Xc, 'f1_layer_' + str(layer) + '_' + str(i), True,
                                                 return_alpha=True)
                    # _, f_2 = self._attention_fcn(self.item_eb, Xc, 'f2_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, Xc, 'f1_shared' + '_' + str(i), True, return_alpha=True)
                    # _, f_2 = self._attention_fcn(self.item_eb, Xc, 'f2_shared'+'_'+str(i), True, return_alpha=True)
                # cluster_score += [f_1 + f_2]
                cluster_score += [f_1]

            cluster_score = tf.reduce_mean(tf.stack(cluster_score, 0), 0)
            boolean_mask = tf.expand_dims(tf.equal(self.mask, tf.ones_like(self.mask)), axis=-1)
            mask_paddings = tf.ones_like(cluster_score) * (-(2 ** 32) + 1)
            cluster_score = tf.nn.softmax(tf.where(boolean_mask, cluster_score, mask_paddings), axis=-1)

            ## graph pooling
            num_nodes = tf.reduce_sum(self.mask, 1)  # B
            boolean_pool = tf.greater(num_nodes, self.pool_length)  # B
            to_keep = tf.where(boolean_pool,
                               tf.cast(self.pool_length + (
                                           self.real_sequence_length - self.pool_length) / self.pool_layers * (
                                                   self.pool_layers - layer - 1), tf.float32),
                               num_nodes)  # B

            cluster_score = tf.squeeze(cluster_score, axis=-1) * self.float_mask  # B*L
            sorted_score = tf.sort(cluster_score, direction='DESCENDING', axis=-1)  # B*L
            target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1),
                                        batch_dims=1) + K.epsilon()  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            topk_mask = tf.greater(cluster_score, tf.expand_dims(target_score, -1))  # B*L + B*1 -> B*L
            mask = tf.cast(topk_mask, tf.int32)
            self.float_mask = tf.cast(mask, tf.float32)
            self.reduced_sequence_length = tf.reduce_sum(mask, 1)

            ## ensure graph connectivity

            E = E * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
            A = tf.matmul(tf.matmul(E, A_bool), tf.transpose(E, (0, 2, 1)))  # B*C*L x B*L*L x B*L*C = B*C*C
            ## graph readout
            graph_readout = tf.reduce_sum(Xc * tf.expand_dims(cluster_score, -1) * tf.expand_dims(self.float_mask, -1),
                                          1)

        return Xc, A, graph_readout, cluster_score


def build_dmon(input_features,
               input_graph,
               input_adjacency):
    """Builds a Deep Modularity Network (DMoN) model from the Keras inputs.
    Args:
      input_features: A dense [n, d] Keras input for the node features.
      input_graph: A sparse [n, n] Keras input for the normalized graph.
      input_adjacency: A sparse [n, n] Keras input for the graph adjacency.
    Returns:
      Built Keras DMoN model.
    """
    output = input_features
    for n_channels in [64, 64, 64]:
        output = GCN(n_channels)([output, input_graph])
    pool, pool_assignment = DMoN(4, collapse_regularization=0.1, dropout_rate=0)([output, input_adjacency])
    return pool, pool_assignment


class SineEnc():
    def __init__(self, dim, max_time_scale=1e4):
        super(SineEnc, self).__init__()
        self.dim = dim * 2
        self.denom = tf.range(0, self.dim, 2) / self.dim
        self.denom = tf.cast(self.denom, dtype=tf.float32)
        self.denom *= -tf.math.log(max_time_scale)
        self.denom = tf.reshape(tf.exp(self.denom), [1, 1, -1])
        self.output_dim = dim

    def __call__(self, timestamps):
        timestamps = tf.cast(tf.tile(tf.expand_dims(timestamps, axis=-1), [1, 1, self.dim // 2]), tf.float32)
        pe = tf.zeros([get_shape(timestamps)[0], get_shape(timestamps)[1], self.dim // 2], dtype=tf.float32)
        # ！！！没有实现交替赋值
        # odd_mask = tf.constant([True, False])
        # odd_mask = tf.tile(tf.expand_dims(tf.expand_dims(odd_mask, axis=-1), axis=-1),
        #                    [get_shape(timestamps)[0] //2, get_shape(timestamps)[1], self.dim // 2])
        # even_mask = tf.logical_not(odd_mask)

        pe = tf.sin(timestamps * self.denom)
        pe += tf.cos(timestamps * self.denom)
        return pe


class Model_My(Model2):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len):
        super(Model_My, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len,
                                       flag="My")

    # def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len, reg_cont):
    #     super(Model_My, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span,
    #                                    seq_len, reg_cont, flag="Set")
        # item_his_emb = self.item_list_add_pos_time
        # self.item_list_emb = tf.nn.l2_normalize(self.item_list_emb, dim=2)
        item_his_emb = self.item_list_emb
        self.num_interest = num_interest
        self.relative_threshold_item = 0.5
        self.relative_threshold_interest = 0.8
        self.metric_heads_item = 1
        self.metric_heads_interest = 1
        self.attention_heads = 1
        self.seq_len = seq_len
        self.gat_item = GraphAttention1(embedding_dim)
        self.gat_interest = GraphAttention2(embedding_dim)
        self.interest_cluster = 20

        X = self.item_list_emb
        # X = self.item_list_add_pos_time
        self.float_mask = tf.cast(self.mask, tf.float32)
        self.real_sequence_length = tf.reduce_sum(self.mask, 1)

        with tf.name_scope('item_graph'):
            ## Node similarity metric learning
            S_item = []
            for i in range(self.metric_heads_item):
                # weighted cosine similarity
                self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), get_shape(X)[-1], use_bias=False)
                X_fts = X * tf.expand_dims(self.weighted_tensor, 0)
                X_fts = tf.nn.l2_normalize(X_fts, dim=2)
                S_item_one = tf.matmul(X_fts, tf.transpose(X_fts, (0, 2, 1)))  # B*L*L
                # min-max normalization for mask
                S_item_min = tf.reduce_min(S_item_one, -1, keepdims=True)
                S_item_max = tf.reduce_max(S_item_one, -1, keepdims=True)
                # S_one = (S_one - S_min) / ((S_max - S_min) + 1)
                S_item_one = (S_item_one - S_item_min) / (S_item_max - S_item_min)
                S_item_one = tf.where(tf.math.is_nan(S_item_one), tf.zeros_like(S_item_one), S_item_one)
                S_item += [S_item_one]
            S_item = tf.reduce_mean(tf.stack(S_item, 0), 0)
            # mask invalid nodes  S = B*L*L * B*L*1 * B*1*L = B*L*L
            S_item = S_item * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)

            ## Graph sparsification via seted sparseness
            S_flatten = tf.reshape(S_item, [get_shape(S_item)[0], -1])  # B*[L*L]
            sorted_S_flatten = tf.sort(S_flatten, direction='DESCENDING', axis=-1)  # B*L -> B*L
            # relative ranking strategy of the entire graph
            num_edges = tf.cast(tf.count_nonzero(S_item, [1, 2]), tf.float32)  # B
            to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.relative_threshold_item), tf.int32)
            threshold_score = tf.gather_nd(sorted_S_flatten,
                                           tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1),
                                           batch_dims=1)  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            A_item = tf.cast(tf.greater(S_item, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)

        # self.graph_gat = self.gat_item([X, A_item])     # [B, L, dim]
        # con = tf.concat([self.graph_gat, X], axis=-1)
        # self.graph_gat = multihead_attention(self.graph_gat, con, con)
        # gat_mask = tf.tile(tf.expand_dims(self.mask, axis=-1), [1, 1, hidden_size])
        # paddings = tf.zeros_like(gat_mask)
        # self.graph_gat = tf.where(tf.equal(gat_mask, 0), paddings, self.graph_gat)

        # with tf.variable_scope("cluster_att", reuse=tf.AUTO_REUSE) as scope:
        #     X_hidden = tf.layers.dense(item_his_emb, hidden_size * 4, activation=tf.nn.tanh)
        #     X_att_w = tf.layers.dense(X_hidden, num_interest, activation=None)
        #     X_att_w = tf.transpose(a=X_att_w, perm=[0, 2, 1])
        #
        #     X_att_w = tf.nn.softmax(X_att_w)
        #
        # S = tf.transpose(a=X_att_w, perm=[0, 2, 1])
        # S = tf.arg_max(S, dimension=2)

        # Kmeans
        self.w = tf.layers.dense(tf.ones([1, 1]), get_shape(X)[-1], use_bias=False)
        X_clu = X * tf.expand_dims(self.w, 0)
        mu = tf.slice(X_clu, [0, 0, 0], [get_shape(X)[0], num_interest, get_shape(X)[2]])
        mu = mu * tf.expand_dims(self.w, 0)
        mu_iter = tf.stop_gradient(mu, name='mu_iter')
        # mu_iter = mu
        max_iter = 3
        c = 1e10
        for _ in range(max_iter):
            sim = tf.matmul(X_clu, tf.transpose(mu_iter, [0, 2, 1]))  # [B, len, dim] * [B, k, dim] = [B, len, k]
            S = tf.arg_max(sim, dimension=2)  # [B, len]
            # S = tf.cast(tf.reduce_max(tf.nn.softmax(sim / c), axis=-1), tf.int32)  # [B, len]
            mu_temp = []
            for i in range(num_interest):
                X_i = tf.multiply(X_clu,
                                  tf.cast(tf.equal(tf.tile(tf.expand_dims(S, axis=-1), [1, 1, embedding_dim]), i),
                                          tf.float32))
                count_i = tf.tile(tf.expand_dims(tf.reduce_sum(tf.cast(tf.equal(S, i), tf.float32), axis=1), axis=-1),
                                  [1, embedding_dim])
                count_i = tf.where(tf.equal(count_i, 0), tf.ones_like(count_i), count_i)
                mu_i = tf.reduce_sum(X_i, axis=1) / count_i  # [B, dim]
                mu_temp += [mu_i]
            mu_iter = tf.stack(mu_temp, axis=1)

        # A_item = tf.ones((get_shape(X)[0], seq_len, seq_len))
        readout_graph = []
        user_eb = []
        self.graph_gat = 0
        for i in range(num_interest):
            mask_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
            t = tf.expand_dims(tf.equal(S, i), axis=-1)
            A_i = tf.multiply(tf.cast(tf.logical_and(t, tf.transpose(t, perm=[0, 2, 1])), tf.float32), A_item)
            graph_gat_i = self.gat_item([X, A_i])
            # # graph_gat_i = GCN(embedding_dim)([X, A_i])
            gat_mask_i = tf.tile(tf.expand_dims(mask_i, axis=-1), [1, 1, hidden_size])
            paddings_i = tf.cast(tf.zeros_like(gat_mask_i), tf.float32)
            graph_gat_i = tf.where(tf.equal(gat_mask_i, 0), paddings_i, graph_gat_i)
            X_i = tf.multiply(X, gat_mask_i)
            user_eb_i = sa(X_i, mask_i, i + 1, num_interest=1, hidden_size=hidden_size)  # [B, 1, d]
            user_eb += [user_eb_i]
            readout_graph += [tf.sigmoid(tf.reduce_mean(graph_gat_i, axis=1))]
            self.graph_gat += graph_gat_i
        readout_graph = tf.stack(readout_graph, 1)
        self.user_eb = tf.squeeze(tf.transpose(tf.stack(user_eb, 0), [1, 0, 2, 3]))

        readout_graph = tf.nn.l2_normalize(readout_graph, dim=2)
        self.graph_gat = tf.nn.l2_normalize(self.graph_gat, dim=2)  # [B, len, dim]
        self.user_eb = tf.nn.l2_normalize(self.user_eb, dim=2)

        num_heads = num_interest
        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_his_emb)[0], embedding_dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_his_emb)[0], num_heads]), 1))

        self.readout = tf.gather(tf.reshape(self.user_eb, [-1, embedding_dim]),
                                 tf.argmax(input=atten, axis=1, output_type=tf.int32) + tf.range(
                                     tf.shape(input=item_his_emb)[0]) * num_heads)

        self.loss_cont_item_interest = 0
        self.matrix2 = tf.exp(tf.matmul(self.graph_gat, tf.transpose(self.user_eb, [0, 2, 1])))  # [B, len, k]
        # self.matrix2 = tf.matmul(gat_norm, tf.transpose(self.user_eb, [0, 2, 1]))   # [B, len, k]
        for i in range(num_interest):
            mask_pos_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
            matrix_exp = tf.expand_dims(self.matrix2[:, :, i], axis=-1)
            mask_neg_i = self.mask
            mask_pos_i = tf.expand_dims(mask_pos_i, axis=-1)
            mask_neg_i = tf.expand_dims(mask_neg_i, axis=-1)
            cont_pos = tf.squeeze(tf.multiply(matrix_exp, mask_pos_i)) + 1e-8  # [B, len]
            cont_neg = tf.squeeze(tf.multiply(matrix_exp, mask_neg_i)) + 1e-8  # [B, len]
            self.loss_cont_item_interest += -tf.reduce_mean(
                tf.log(cont_pos / tf.tile(tf.expand_dims(tf.reduce_sum(cont_neg, axis=-1), axis=-1), [1, seq_len])))

        # self.loss_cont_readout_in = 0
        # self.matrix1 = tf.exp(tf.matmul(self.user_eb, tf.transpose(readout_graph, [0, 2, 1])))   # [B, len, k]
        # for i in range(num_interest):
        #     mask_pos_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
        #     matrix_exp = tf.expand_dims(self.matrix1[:, :, i], axis=-1)
        #     mask_neg_i = self.mask
        #     mask_pos_i = tf.expand_dims(mask_pos_i, axis=-1)
        #     mask_neg_i = tf.expand_dims(mask_neg_i, axis=-1)
        #     cont_pos = tf.squeeze(tf.multiply(matrix_exp, mask_pos_i)) + 1e-8   # [B, len]
        #     cont_neg = tf.squeeze(tf.multiply(matrix_exp, mask_neg_i)) + 1e-8   # [B, len]
        #     self.loss_cont_readout_in += -tf.reduce_mean(tf.log(cont_pos / tf.tile(tf.expand_dims(tf.reduce_sum(cont_neg, axis=-1), axis=-1), [1, seq_len])))

        # self.loss_cont_interest = 0
        # matrix = tf.exp(tf.matmul(self.user_eb, tf.transpose(readout_graph, [0, 2, 1])))     # [B, k, k]
        # for i in range(num_interest):
        #     self.loss_cont_interest += -tf.reduce_mean(tf.log(matrix[:, i, i] / tf.reduce_mean(matrix[:, i, i:], axis=-1)))

        # L_M
        # self.loss_LM = tf.norm(tf.multiply(A, tf.matmul(S, tf.transpose(S, [0, 2, 1]))))       #

        # 消融实验
        # dense    gat
        # self.build_sampled_softmax_loss(self.item_eb, self.readout)

        # loss_balance
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_balance)

        # loss_lm
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_LM)

        # loss_cont
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_cont_readout_in)
        self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_cont_item_interest)

        # loss_LM loss_balance
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_LM, self.loss_balance)

        # loss_cont loss_balance
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_cont_item_interest, self.loss_balance)

        # loss_LM loss_cont
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_LM, self.loss_cont)

        # # loss_lm loss_cont loss_balance
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_LM, self.loss_cont, self.loss_balance)



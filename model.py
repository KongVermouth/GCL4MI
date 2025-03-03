import os
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


# baseline
class Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.neg_num = 10
        self.reg_attend = 0.1
        self.reg_contrast = 0.2
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

    # def build_sampled_softmax_loss(self, item_emb, user_emb):
    #     self.loss = tf.reduce_mean(input_tensor=tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid))
    #
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_attend):
    #     self.loss = tf.reduce_mean(input_tensor=tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid))
    #
    #     self.loss = self.loss + self.reg_attend * loss_attend
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, loss_attend, loss_contrast):
    #     self.loss = tf.reduce_mean(input_tensor=tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias, tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.n_mid))
    #
    #     self.loss = self.loss + self.reg_attend * loss_attend + self.reg_contrast * loss_contrast
    #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def build_sampled_softmax_loss(self, item_emb, user_emb, loss_attend, loss_contrast, loss_construct):
        self.loss = tf.reduce_mean(
            input_tensor=tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias,
                                                    tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb,
                                                    self.neg_num * self.batch_size, self.n_mid))

        self.loss = self.loss + self.reg_attend * loss_attend + self.reg_contrast * loss_contrast + self.reg_construct * loss_construct
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

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
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len, flag="DNN"):
    # def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len, re_cont, flag="DNN"):
        self.batch_size = batch_size
        self.num_items = n_mid
        self.neg_num = 10
        self.dim = embedding_dim
        self.reg_LM = 0.1
        self.reg_cont_item = 0.03
        self.reg_cont_interest = 0.05
        self.reg_cont_item_interest = 0.1
        self.reg_clu_loss = 0.1
        self.reg_balance = 0.1
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.uid_batch = tf.placeholder(tf.int32, [None, ], name='user_id_batch')
        self.itemid_batch = tf.placeholder(tf.int32, [None, ], name='target_item_id_batch')
        self.his_itemid_batch = tf.placeholder(tf.int32, [None, seq_len], name='his_item_id_batch')
        self.mask = tf.placeholder(tf.float32, [None, seq_len], name='his_mask_batch')
        self.adj_matrix = tf.placeholder(tf.float32, [None, seq_len, seq_len + 2], name='item_adjacent_batch')
        self.time_matrix = tf.placeholder(tf.int32, [None, seq_len, seq_len], name='item_time_interval_batch')
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

            self.time_matrix_emb = embedding(self.time_matrix, vocab_size=time_span + 1, num_units=hidden_size,
                                             scope="time_matrix", reuse=None)
            self.time_W = tf.get_variable("time_W_var", [hidden_size, 1], trainable=True)
            time_matrix_attention = tf.reshape(
                tf.matmul(tf.reshape(self.time_matrix_emb, [-1, hidden_size]), self.time_W),
                [-1, seq_len, seq_len, 1])

            time_mask = tf.expand_dims(tf.tile(tf.expand_dims(self.mask, axis=1), [1, seq_len, 1]), axis=-1)
            time_paddings = tf.ones_like(time_mask) * (-2 ** 32 + 1)

            time_matrix_attention = tf.where(tf.equal(time_mask, 0), time_paddings, time_matrix_attention)

            time_matrix_attention = tf.nn.softmax(time_matrix_attention, axis=-2)
            time_matrix_attention = tf.transpose(a=time_matrix_attention, perm=[0, 1, 3, 2])
            time_emb = tf.squeeze(tf.matmul(time_matrix_attention, self.time_matrix_emb), axis=2)

            self.item_list_add_pos_time = self.item_list_add_pos + time_emb

            self.item_list_add_pos_time *= mask

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

    def build_sampled_softmax_loss(self, item_emb, user_emb, loss_cont_item_interest):
        self.loss = tf.reduce_mean(
            input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
                                                    tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
                                                    self.neg_num * self.batch_size, self.num_items))

        self.loss = self.loss + self.reg_cont_item_interest * loss_cont_item_interest
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
        self.user_eb = tf.layers.dense(self.item_his_eb_mean, hidden_size, activation=None)
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

        # [B, k, L]
        # item_att_w_pad = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
        # full_zero_mask = tf.cast(tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_sum(mask, axis=-1), axis=-1), axis=-1), [1, num_heads, seq_len]), tf.int32)
        # paddings_full_zeros = tf.where(tf.equal(full_zero_mask, 0), tf.ones_like(full_zero_mask), tf.zeros_like(full_zero_mask))
        # item_att_w_pad = tf.where(tf.equal(paddings_full_zeros, 1), tf.random_normal(get_shape(paddings_full_zeros), mean=100), item_att_w_pad)
        # item_att_w_pad = tf.nn.softmax(item_att_w)

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

        target_emb = tf.tile(tf.expand_dims(item_list_emb, axis=1), [1, num_interest, 1, 1])
        loss_construct = tf.square(recons_item - target_emb)  # [B, k, L, d]
        loss_construct = tf.where(tf.tile(tf.expand_dims(tf.equal(positive_weight_idx, 0), axis=-1),
                                          [1, 1, 1, hidden_size]), tf.zeros_like(loss_construct), loss_construct)
        loss_construct = tf.where(tf.equal(tf.tile(tf.expand_dims(tf.expand_dims(self.mask, axis=1), axis=-1),
                                                   [1, num_interest, 1, hidden_size]), 0),
                                  tf.zeros_like(loss_construct), loss_construct)
        loss_construct = tf.reduce_mean(loss_construct)

        # watch_interests = K.tanh(tf.layers.dense(watch_interests, embedding_dim, activation=None))
        self.user_eb = watch_interests

        num_heads = num_interest
        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1))

        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]),
                            tf.argmax(input=atten, axis=1, output_type=tf.int32) + tf.range(
                                tf.shape(input=item_list_emb)[0]) * num_heads)

        self.build_sampled_softmax_loss(self.item_eb, readout, loss_attend, loss_contrast, loss_construct)


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


class Model_My(Model2):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len, n_layers=1,
                 n_head=1, relu_layer=False, hard_readout=True):
        super(Model_My, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len,
                                       flag="Set")

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

        self.mask_0 = tf.cast(tf.logical_and(tf.equal(S, 0), tf.cast(self.mask, tf.bool)), tf.float32)
        self.mask_1 = tf.cast(tf.logical_and(tf.equal(S, 1), tf.cast(self.mask, tf.bool)), tf.float32)
        self.mask_2 = tf.cast(tf.logical_and(tf.equal(S, 2), tf.cast(self.mask, tf.bool)), tf.float32)
        self.mask_3 = tf.cast(tf.logical_and(tf.equal(S, 3), tf.cast(self.mask, tf.bool)), tf.float32)

        # self.loss_clu = 0
        # X = tf.nn.l2_normalize(X, dim=2)
        # matrix = tf.exp(tf.matmul(X, tf.transpose(X, [0, 2, 1])))   # [B, len, len]
        # for i in range(num_interest):
        #     mask_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
        #     mask_pos_i = tf.tile(tf.expand_dims(mask_i, axis=-1), [1, 1, seq_len])
        #     mask_neg_i = tf.tile(tf.expand_dims(self.mask, axis=-1), [1, 1, seq_len])
        #     cont_pos = tf.multiply(matrix, mask_pos_i) + 1e-8
        #     cont_neg = tf.multiply(matrix, mask_neg_i) + 1e-8
        #     self.loss_clu += -tf.reduce_mean(tf.log(cont_pos / tf.tile(tf.expand_dims(tf.reduce_sum(cont_neg, axis=-1), axis=-1), [1, 1, seq_len])))

        # 簇的数目约束
        # self.loss_balance = 0
        # for i in range(num_interest):
        #     cluster_num_i = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32), axis=-1)
        #     self.loss_balance += tf.reduce_mean(tf.square(cluster_num_i - self.real_sequence_length / num_interest))

        A_item = tf.ones((get_shape(X)[0], seq_len, seq_len))
        readout_graph = []
        self.graph_gat = 0
        for i in range(num_interest):
            mask_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
            t = tf.expand_dims(tf.equal(S, i), axis=-1)
            A_i = tf.multiply(tf.cast(tf.logical_and(t, tf.transpose(t, perm=[0, 2, 1])), tf.float32), A_item)
            graph_gat_i = self.gat_item([X, A_i])
            # graph_gat_i = GCN(embedding_dim)([X, A_i])
            gat_mask_i = tf.tile(tf.expand_dims(mask_i, axis=-1), [1, 1, hidden_size])
            paddings_i = tf.cast(tf.zeros_like(gat_mask_i), tf.float32)
            graph_gat_i = tf.where(tf.equal(gat_mask_i, 0), paddings_i, graph_gat_i)
            readout_graph += [tf.sigmoid(tf.reduce_mean(graph_gat_i, axis=1))]
            self.graph_gat += graph_gat_i
        readout_graph = tf.stack(readout_graph, 1)
        # self.user_eb = readout_graph

        user_eb = []
        for i in range(num_interest):
            mask_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)     # [B, len]
            user_eb_i = sa(X, mask_i, i + 1, num_interest=1, hidden_size=hidden_size)        # [B, 1, d]
            user_eb += [user_eb_i]
        self.user_eb = tf.squeeze(tf.transpose(tf.stack(user_eb, 0), [1, 0, 2, 3]))

        num_heads = num_interest
        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_his_emb)[0], embedding_dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_his_emb)[0], num_heads]), 1))

        self.readout = tf.gather(tf.reshape(self.user_eb, [-1, embedding_dim]),
                                 tf.argmax(input=atten, axis=1, output_type=tf.int32) + tf.range(
                                     tf.shape(input=item_his_emb)[0]) * num_heads)

        # readout_graph = tf.nn.l2_normalize(readout_graph, dim=2)
        self.graph_gat = tf.nn.l2_normalize(self.graph_gat, dim=2)  # [B, len, dim]
        self.user_eb = tf.nn.l2_normalize(self.user_eb, dim=2)

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

        # self.loss_cont_item = 0
        # self.matrix1 = tf.exp(tf.matmul(gat_norm, tf.transpose(readout_graph, [0, 2, 1])))   # [B, len, k]
        # for i in range(num_interest):
        #     mask_pos_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
        #     matrix_exp = tf.expand_dims(self.matrix1[:, :, i], axis=-1)
        #     mask_neg_i = self.mask
        #     mask_pos_i = tf.expand_dims(mask_pos_i, axis=-1)
        #     mask_neg_i = tf.expand_dims(mask_neg_i, axis=-1)
        #     cont_pos = tf.squeeze(tf.multiply(matrix_exp, mask_pos_i)) + 1e-8   # [B, len]
        #     cont_neg = tf.squeeze(tf.multiply(matrix_exp, mask_neg_i)) + 1e-8   # [B, len]
        #     self.loss_cont_item += -tf.reduce_mean(tf.log(cont_pos / tf.tile(tf.expand_dims(tf.reduce_sum(cont_neg, axis=-1), axis=-1), [1, seq_len])))
        #
        # self.loss_cont_interest = 0
        # matrix = tf.exp(tf.matmul(self.user_eb, tf.transpose(readout_graph, [0, 2, 1])))     # [B, k, k]
        # for i in range(num_interest):
        #     self.loss_cont_interest += -tf.reduce_mean(tf.log(matrix[:, i, i] / tf.reduce_mean(matrix[:, i, i:], axis=-1)))

        # L_M
        # self.loss_LM = tf.norm(tf.multiply(A, tf.matmul(S, tf.transpose(S, [0, 2, 1]))))       #

        # 消融实验
        # dense    gat-4head
        # self.build_sampled_softmax_loss(self.item_eb, self.readout)

        # loss_balance
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_balance)

        # loss_lm
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_LM)

        # loss_cont
        self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_cont_item_interest)
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_cont, self.loss_cont_interest)

        # loss_LM loss_balance
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_LM, self.loss_balance)

        # loss_cont loss_balance
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_cont_item_interest, self.loss_balance)

        # loss_LM loss_cont
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_LM, self.loss_cont)

        # # loss_lm loss_cont loss_balance
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_LM, self.loss_cont, self.loss_balance)

# class Model_Set(Model2):
#     def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len, n_layers=1, n_head=1, relu_layer=False, hard_readout=True):
#         super(Model_Set, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len, flag="Set")
#         # item_his_emb = self.item_list_add_pos_time
#         item_his_emb = self.item_list_emb
#         self.num_interest = num_interest
#         self.dim = embedding_dim
#         self.relative_threshold = 0.2
#         self.metric_heads = 1
#         self.attention_heads = 1
#         self.pool_layers = 1
#         self.layer_shared = True
#         self.seq_len = seq_len
#         self.layer_num = num_interest
#         self.n_layers = n_layers
#         self.n_head = n_head
#         self.intent_num = 100
#
#         X = tf.tile(self.item_list_emb, [1, 1, 1])
#         X = tf.reshape(X, [-1, embedding_dim])         # [B*l, d]
#         self.float_mask = tf.tile(self.mask, [1, 1])
#         self.float_mask = tf.reshape(tf.cast(self.float_mask, tf.float32), [-1, 1])  # [B*l, 1]
#         self.real_sequence_length = tf.reduce_sum(self.mask, 1)
#
#         with tf.name_scope('interest_graph'):
#             ## Node similarity metric learning
#             S = []
#             for i in range(self.metric_heads):
#                 # weighted cosine similarity
#                 self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), get_shape(X)[-1], use_bias=False)
#                 X_fts = X * self.weighted_tensor                       # [B*l, d]
#                 X_fts = tf.nn.l2_normalize(X_fts, dim=1)
#                 S_one = tf.matmul(X_fts, tf.transpose(X_fts, (1, 0)))  # [B*l, B*l]
#                 # min-max normalization for mask
#                 S_min = tf.reduce_min(S_one, -1, keepdims=True)
#                 S_max = tf.reduce_max(S_one, -1, keepdims=True)
#                 # S_one = (S_one - S_min) / ((S_max - S_min) + 1)
#                 S_one = (S_one - S_min) / (S_max - S_min)
#                 S_one = tf.where(tf.math.is_nan(S_one), tf.zeros_like(S_one), S_one)
#                 S += [S_one]
#             S = tf.reduce_mean(tf.stack(S, 0), 0)
#             S = S * tf.tile(self.float_mask, [1, get_shape(S)[0]]) * tf.tile(tf.transpose(self.float_mask), [get_shape(S)[0], 1])    ####
#
#             ## Graph sparsification via seted sparseness
#             num_edges = tf.cast(tf.count_nonzero(S, [0, 1]), tf.float32)
#             threshold_score = num_edges * self.relative_threshold
#             A = tf.where(tf.greater(S, threshold_score * tf.ones_like(S)), tf.ones_like(S), tf.zeros_like(S))      ###
#
#         self.gat = GraphAttention(embedding_dim)
#         gat_X = self.gat([X, A])         #####
#         gat_X = tf.reshape(gat_X, [-1, embedding_dim])
#         # Kmeans
#         mu = tf.slice(gat_X, [0, 0], [self.intent_num, embedding_dim])
#         mu_iter = tf.stop_gradient(mu, name='mu_iter')
#         # mu_iter = mu
#         max_iter = 10
#         for _ in range(max_iter):
#             sim = tf.matmul(X, tf.transpose(mu_iter))   # [B*l, i_num]
#             S = tf.arg_max(sim, dimension=1)    # [B*l]
#             mu_temp = []
#             for i in range(self.intent_num):
#                 X_i = tf.multiply(X, tf.cast(tf.equal(tf.tile(tf.expand_dims(S, axis=-1), [1, embedding_dim]), i), tf.float32))
#                 mu_i = tf.reduce_sum(X_i, axis=-1)     # [B*l]
#                 mu_temp += [mu_i]
#             mu_iter = tf.stack(mu_temp, axis=0)
#
#         self.intent = mu_iter
#         # X-cluster-att
#         # item_cluster_att = multihead_attention(self.item_list_emb, mu_iter, mu_iter)
#         S = tf.nn.softmax(tf.layers.dense(self.item_list_emb, num_interest, activation=None), axis=-1)
#         S = tf.argmax(S, axis=-1)
#         user_eb = []
#         for i in range(num_interest):
#             mask_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
#             user_eb_i = sa(self.item_list_emb, mask_i, i + 1, num_interest=1, hidden_size=hidden_size)
#             user_eb += [user_eb_i]
#         self.user_eb = tf.squeeze(tf.transpose(tf.stack(user_eb, 0), [1, 0, 2, 3]))
#
#         num_heads = num_interest
#         atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_his_emb)[0], embedding_dim, 1]))
#         atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_his_emb)[0], num_heads]), 1))
#
#         self.readout = tf.gather(tf.reshape(self.user_eb, [-1, embedding_dim]),
#                             tf.argmax(input=atten, axis=1, output_type=tf.int32) + tf.range(
#                                 tf.shape(input=item_his_emb)[0]) * num_heads)
#
#         self.build_sampled_softmax_loss(self.item_eb, self.readout)

import os
from modules import *
from utils.gat import *
from dmon.dmon import *
from dmon.gcn import *
from dmon.utils import *


class Model2(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len, flag="DNN"):
    # def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len, re_cont, flag="DNN"):
        self.batch_size = batch_size
        self.num_items = n_mid
        self.neg_num = 10
        self.dim = embedding_dim
        self.reg_cont = 0.1
        self.reg_cont_readout_in = 0.1
        self.reg_cont_readout_cont = 0.1
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

            # self.time_matrix_emb = embedding(self.time_matrix, vocab_size=time_span + 1, num_units=hidden_size,
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

    def build_sampled_softmax_loss(self, item_emb, user_emb, loss_cont):
        self.loss = tf.reduce_mean(
            input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
                                                    tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
                                                    self.neg_num * self.batch_size, self.num_items))

        self.loss = self.loss + self.reg_cont * loss_cont
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # def build_sampled_softmax_loss(self, item_emb, user_emb, reg_cont_readout_in):
    #     self.loss = tf.reduce_mean(
    #         input_tensor=tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias,
    #                                                 tf.reshape(self.itemid_batch, [-1, 1]), user_emb,
    #                                                 self.neg_num * self.batch_size, self.num_items))
    #
    #     self.loss = self.loss + self.reg_cont_readout_in * reg_cont_readout_in
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
        X, cluster_res, user_eb, user_eb_before = sess.run([self.item_list_emb,
                            self.cluster_res, self.user_eb, self.user_eb_before],
                            feed_dict={self.his_itemid_batch: hist_item,
                                       self.mask: hist_mask,
                                       self.is_training: False})

        return X, cluster_res, user_eb, user_eb_before

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)


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
    with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
        item_hidden = tf.layers.dense(item_list_add_pos, hidden_size * 4, activation=tf.nn.tanh)
        item_att_w = tf.layers.dense(item_hidden, num_heads, activation=None)
        item_att_w = tf.transpose(a=item_att_w, perm=[0, 2, 1])

        atten_mask = tf.cast(tf.tile(tf.expand_dims(mask, axis=1), [1, num_heads, 1]), tf.float32)
        paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

        item_att_w_pad = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
        item_att_w_pad = tf.nn.softmax(item_att_w_pad)

        interest_emb = tf.matmul(item_att_w_pad, item_list_emb)

    return interest_emb


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


class Model(Model2):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len):
        super(Model, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len,
                                       flag="Set")
        item_his_emb = self.item_list_emb
        self.num_interest = num_interest
        self.relative_threshold = 0.5
        self.metric_heads_item = 1
        self.seq_len = seq_len
        self.gat_item = GraphAttention1(embedding_dim)

        X = self.item_list_emb
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
            to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.relative_threshold), tf.int32)
            threshold_score = tf.gather_nd(sorted_S_flatten,
                                           tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1),
                                           batch_dims=1)  # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            A_item = tf.cast(tf.greater(S_item, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)


        # Kmeans
        self.w = tf.layers.dense(tf.ones([1, 1]), get_shape(X)[-1], use_bias=False)
        X_clu = X * tf.expand_dims(self.w, 0)
        mu = tf.slice(X_clu, [0, 0, 0], [get_shape(X)[0], num_interest, get_shape(X)[2]])
        mu_iter = tf.stop_gradient(mu, name='mu_iter')
        max_iter = 3
        for _ in range(max_iter):
            sim = tf.matmul(X_clu, tf.transpose(mu_iter, [0, 2, 1]))  # [B, len, dim] * [B, k, dim] = [B, len, k]
            S = tf.arg_max(sim, dimension=2)  # [B, len]
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
        # self.graph_gat = self.gat_item([X, A_item])

        self.cluster_res = S
        self.graph_gat = 0
        readout_graph = []
        for i in range(num_interest):
            mask_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
            t = tf.expand_dims(tf.equal(S, i), axis=-1)
            A_i = tf.multiply(tf.cast(tf.logical_and(t, tf.transpose(t, perm=[0, 2, 1])), tf.float32), A_item)
            graph_gat_i = self.gat_item([X, A_i])
            # graph_gat_i = GCN(embedding_dim)([X, A_i])
            gat_mask_i = tf.tile(tf.expand_dims(mask_i, axis=-1), [1, 1, hidden_size])
            paddings_i = tf.cast(tf.zeros_like(gat_mask_i), tf.float32)
            graph_gat_i = tf.where(tf.equal(gat_mask_i, 0), paddings_i, graph_gat_i)
            # X_i = tf.multiply(X, gat_mask_i)
            readout_graph += [tf.sigmoid(tf.reduce_mean(graph_gat_i, axis=1))]
            self.graph_gat += graph_gat_i
        readout_graph = tf.stack(readout_graph, 1)
        # self.user_eb = sa(X, tf.cast(self.mask, tf.float32), 4, num_interest=num_interest, hidden_size=hidden_size)
        # print(self.user_eb)

        user_eb = []
        for i in range(num_interest):
            mask_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
            user_eb_i = sa(X, mask_i, i + 1, num_interest=1, hidden_size=hidden_size)  # [B, 1, d]
            user_eb += [user_eb_i]
        self.user_eb = tf.squeeze(tf.transpose(tf.stack(user_eb, 0), [1, 0, 2, 3]))
        # self.user_eb_before = self.user_eb

        readout_graph = tf.nn.l2_normalize(readout_graph, dim=2)
        self.graph_gat = tf.nn.l2_normalize(X, dim=2)  # [B, len, dim]
        self.user_eb = tf.nn.l2_normalize(self.user_eb, dim=2)

        num_heads = num_interest
        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_his_emb)[0], embedding_dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_his_emb)[0], num_heads]), 1))

        self.readout = tf.gather(tf.reshape(self.user_eb, [-1, embedding_dim]),
                                 tf.argmax(input=atten, axis=1, output_type=tf.int32) + tf.range(
                                     tf.shape(input=item_his_emb)[0]) * num_heads)

        # self.loss_cont_readout_in = 0
        # matrix_readout_in = tf.exp(tf.matmul(self.user_eb, tf.transpose(readout_graph, [0, 2, 1])))  # [B, k, k]
        # for i in range(num_interest):
        #     self.loss_cont_readout_in += -tf.reduce_mean(tf.log(matrix_readout_in[:, i, i] /
        #                                                         tf.reduce_mean(matrix_readout_in[:, i, i], axis=-1)))

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


        self.loss_cont = 0
        self.matrix = tf.exp(tf.matmul(self.graph_gat, tf.transpose(self.user_eb, [0, 2, 1])))  # [B, len, k]
        for i in range(num_interest):
            mask_pos_i = tf.cast(tf.logical_and(tf.equal(S, i), tf.cast(self.mask, tf.bool)), tf.float32)
            matrix_exp = tf.expand_dims(self.matrix[:, :, i], axis=-1)
            mask_neg_i = self.mask
            mask_pos_i = tf.expand_dims(mask_pos_i, axis=-1)
            mask_neg_i = tf.expand_dims(mask_neg_i, axis=-1)
            cont_pos = tf.squeeze(tf.multiply(matrix_exp, mask_pos_i)) + 1e-8  # [B, len]
            cont_neg = tf.squeeze(tf.multiply(matrix_exp, mask_neg_i)) + 1e-8  # [B, len]
            self.loss_cont += -tf.reduce_mean(
                tf.log(cont_pos / tf.tile(tf.expand_dims(tf.reduce_sum(cont_neg, axis=-1), axis=-1), [1, seq_len])))

        # self.build_sampled_softmax_loss(self.item_eb, self.readout)
        # loss_cont
        self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_cont)

        # loss_cont_readout_in
        # self.build_sampled_softmax_loss(self.item_eb, self.readout, self.loss_cont_readout_in)

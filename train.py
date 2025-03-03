# coding:utf-8
import argparse
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
import numpy as np
import faiss
import tensorflow as tf
from data_iterator import DataIterator
from tensorboardX import SummaryWriter
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from MGNM import *
# from model import *       # baseline
from MyModel import *       # GCL4MI
from matplotlib.colors import rgb2hex
from sklearn.preprocessing import MinMaxScaler
import umap


parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='book', help='book | taobao')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--model_type', type=str, default='My', help='DNN | GRU4REC | ..')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=140, help='(k)')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--coef', default=None)
parser.add_argument('--topN', type=int, default=50)
parser.add_argument('--time_span', type=int, default=64)
parser.add_argument('--ta', type=int, default=0)

best_metric = 0


def prepare_data(src, matrix, target):
    user_id, item_id = src
    adj_matrix, time_matrix = matrix
    hist_item, hist_mask = target
    return user_id, item_id, adj_matrix, time_matrix, hist_item, hist_mask


def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate


def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity


def evaluate_full(sess, test_data, model, model_path, batch_size, topN):
    item_embs = model.output_item(sess)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    try:
        gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
        # gpu_index = faiss.IndexFlatIP(args.embedding_dim)
        gpu_index.add(item_embs)
    except Exception as e:
        return {}

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    for src, matrix, tgt in test_data:
        nick_id, item_id, adj_matrix, time_matrix, hist_item, hist_mask = prepare_data(src, matrix, tgt)
        if args.model_type == "MIP":
            user_embs = model.output_user_mip(sess, hist_item, time_matrix, hist_mask)
        else:
            user_embs = model.output_user(sess, hist_item, hist_mask)
        if len(user_embs.shape) == 2:
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list = set(I[i])
                for no, iid in enumerate(iid_list):
                    if iid in item_list:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()

                item_list = list(zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                item_list.sort(key=lambda x: x[1], reverse=True)

                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break

                for no, iid in enumerate(iid_list):
                    if iid in item_list_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1

        total += len(item_id)

    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total

    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}


def get_model(dataset, model_type, item_count, batch_size, maxlen):
    if model_type == 'DNN':
        model = Model_DNN(item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen)
    elif model_type == 'GRU4REC': 
        model = Model_GRU4REC(item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen)
    elif model_type == 'MIND':
        relu_layer = True if dataset == 'book' else False
        model = Model_MIND(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen, relu_layer=relu_layer)
    elif model_type == 'ComiRec-DR':
        model = Model_ComiRec_DR(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    elif model_type == 'ComiRec-SA':
        model = Model_ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    elif model_type == 'SURGE':
        model = SURGE(item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen)
    elif model_type == 'Re4':
        model = Re4(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    elif model_type == 'My':
        model = Model(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    elif model_type == 'MIP':
        model = MIP(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, args.time_span, maxlen)
    elif model_type == 'MGNM':
        neg_num = 10
        model = modelTy(item_count, args.embedding_dim, batch_size, maxlen, neg_num, args.hidden_size, args.num_interest, num_layer=3, se_num=0)
    else:
        print("Invalid model_type : %s", model_type)
        return
    return model


# def get_exp_name(dataset, batch_size, time_span, embedding_dim, lr, maxlen, extr_name, save=True):
def get_exp_name(dataset, batch_size, time_span, embedding_dim, lr, maxlen, save=True):
    extr_name = '_'.join([str(args.model_type), str(args.num_interest)])
    extr_name = "GCL4MI_k=4"
    para_name = '_'.join([dataset, 'b' + str(batch_size), 'ts' + str(time_span), 'd' + str(embedding_dim), 'lr' + str(lr), 'len' + str(maxlen)])
    exp_name = para_name + '_' + extr_name

    return exp_name


def train(train_file, valid_file, test_file, item_count, batch_size=128, maxlen=20, test_iter=50, lr=0.001,
          max_iter=100, patience=20):
    exp_name = get_exp_name(args.dataset, batch_size, args.time_span, args.embedding_dim, lr, maxlen)

    best_model_path = "best_model/" + exp_name + '/'

    gpu_options = tf.GPUOptions(allow_growth=True)

    writer = SummaryWriter('runs/' + exp_name)

    with open('./res/' + exp_name + '.txt', 'w') as f:
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            train_data = DataIterator(train_file, batch_size, maxlen, args.time_span, train_flag=0)
            valid_data = DataIterator(valid_file, batch_size, maxlen, args.time_span, train_flag=1)
            test_data = DataIterator(test_file, batch_size, maxlen, args.time_span, train_flag=2)

            model = get_model(dataset, args.model_type, item_count, batch_size, maxlen)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            print('training begin')
            f.write('training begin\n')
            sys.stdout.flush()

            start_time = time.time()
            iter = 0
            try:
                loss_sum = 0.0
                trials = 0

                for src, matrix, tgt in train_data:
                    data_iter = prepare_data(src, matrix, tgt)
                    # if args.model_type == "My":
                    #     loss = model.train(sess, list(data_iter) + [lr])
                    # else:
                    loss = model.train(sess, list(data_iter[:2]) + list(data_iter[4:]) + [lr])
                    loss_sum += loss
                    iter += 1

                    if iter % test_iter == 0:
                        log_str = 'iter: %d, train loss: %.4f' % (iter, loss_sum / test_iter)

                        metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, 20)
                        if metrics != {}:
                            log_str += ', ' + ', '.join(
                                ['valid_20 ' + key + ': %.6f' % value for key, value in metrics.items()])

                        metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, 50)
                        if metrics != {}:
                            log_str += ', ' + ', '.join(
                                ['valid_50 ' + key + ': %.6f' % value for key, value in metrics.items()])

                        print(log_str)
                        print(exp_name)
                        f.write(log_str + '\n')
                        f.write(exp_name + '\n')

                        writer.add_scalar('train/loss', loss_sum / test_iter, iter)
                        if metrics != {}:
                            for key, value in metrics.items():
                                writer.add_scalar('eval/' + key, value, iter)

                        if 'recall' in metrics:
                            recall = metrics['recall']
                            global best_metric
                            if recall > best_metric:
                                best_metric = recall
                                model.save(sess, best_model_path)
                                trials = 0
                            else:
                                trials += 1
                                if trials > patience:
                                    break

                        loss_sum = 0.0
                        test_time = time.time()
                        print("time interval: %.4f min" % ((test_time - start_time) / 60.0))
                        f.write("time interval: %.4f min \n" % ((test_time - start_time) / 60.0))
                        sys.stdout.flush()

                    if iter >= max_iter * 1000:
                        break
            except KeyboardInterrupt:
                print('-' * 89)
                print('Exiting from training early')

            model.restore(sess, best_model_path)

            metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, 20)
            print(', '.join(['valid_20 ' + key + ': %.6f' % value for key, value in metrics.items()]))
            f.write(', '.join(['valid_20 ' + key + ': %.6f' % value for key, value in metrics.items()]))
            f.write('\n')
            metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, 50)
            print(', '.join(['valid_50 ' + key + ': %.6f' % value for key, value in metrics.items()]))
            f.write(', '.join(['valid_50 ' + key + ': %.6f' % value for key, value in metrics.items()]))
            f.write('\n')

            metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, 20)
            print(', '.join(['test_20 ' + key + ': %.6f' % value for key, value in metrics.items()]))
            f.write(', '.join(['test_20 ' + key + ': %.6f' % value for key, value in metrics.items()]))
            f.write('\n')
            metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, 50)
            print(', '.join(['test_50 ' + key + ': %.6f' % value for key, value in metrics.items()]))
            f.write(', '.join(['test_50 ' + key + ': %.6f' % value for key, value in metrics.items()]))
            f.write('\n')


def test(
        test_file,
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        lr=0.001,
):
    exp_name = get_exp_name(args.dataset, batch_size, args.time_span, args.embedding_dim, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, batch_size, maxlen)
    start_time = time.time()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        
        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, 20)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))
        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, 50)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

    end_time = time.time()
    print("time interval: %.4f min" % ((end_time - start_time) / 60.0))

def output(
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        lr=0.001
):
    exp_name = get_exp_name(args.dataset, batch_size, args.time_span, args.embedding_dim, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, batch_size, maxlen)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        item_embs = model.output_item(sess)
        np.save('output/' + exp_name + '_emb.npy', item_embs)


if __name__ == '__main__':
    print(sys.argv)
    args = parser.parse_args()
    SEED = args.random_seed

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'taobao':
        path = '/mnt/e/model_code/data/taobao_data/'
        item_count = 1708531
        batch_size = 256
        maxlen = 50
        test_iter = 500

    elif args.dataset == 'book':
        path = '/mnt/e/model_code/data/book_data/'
        item_count = 367983
        batch_size = 1024
        maxlen = 20
        test_iter = 1000

    elif args.dataset == 'movielens':
        path = '/mnt/e/model_code/data/movielens_data/'
        item_count = 3417
        batch_size = 128
        maxlen = 20
        test_iter = 1000

    elif args.dataset == 'ml-10m':
        path = '/mnt/e/model_code/data/ml-10m_data/'
        item_count = 10197
        batch_size = 128
        maxlen = 10
        test_iter = 1000

    elif args.dataset == 'gowalla_3w':
        path = '/mnt/e/model_code/data/gowalla_data_3w/'
        item_count = 40982
        batch_size = 1024
        maxlen = 20
        test_iter = 1000

    elif args.dataset == 'gowalla':
        path = '/mnt/e/model_code/data/gowalla_data/'
        item_count = 75382
        batch_size = 128
        maxlen = 10
        test_iter = 1000

    elif args.dataset == 'Electronics':
        path = '/mnt/e/model_code/data/Electronics_data/'
        item_count = 63002
        batch_size = 128
        maxlen = 10
        test_iter = 1000

    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset
    print("before train")
    if args.p == 'train':
        train(train_file=train_file, valid_file=valid_file, test_file=test_file,
              item_count=item_count, batch_size=batch_size,  maxlen=maxlen,
              test_iter=test_iter, lr=args.learning_rate, max_iter=args.max_iter, patience=args.patience)
    elif args.p == 'test':
        test(test_file=test_file, item_count=item_count, dataset=args.dataset,batch_size=batch_size, maxlen=maxlen,
             model_type=args.model_type, lr=args.learning_rate)
    elif args.p == 'output':
        output(item_count=item_count, dataset=args.dataset, batch_size=batch_size, maxlen=maxlen,
               model_type=args.model_type, lr=args.learning_rate)
    else:
        print('do nothing...')

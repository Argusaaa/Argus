#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import ast
import logging
import math
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import os

import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
from fitter import Fitter

from sklearn.cluster import DBSCAN


def label_cluster_attack_ctu(is_train, files, src_root, dst_root, name2cluster):
    if is_train:
        for file in files:
            path = os.path.join(src_root, file)
            cur_cluster = name2cluster[file]
            new_data = pd.read_pickle(path)
            new_data['cluster'] = cur_cluster
            new_data.to_pickle(os.path.join(dst_root, file))
    else:
        for file in files:
            file_path = os.path.join(src_root, file)
            data = pd.read_pickle(file_path)
            data['cluster'] = -2
            data.to_pickle(os.path.join(dst_root, file))


def label_cluster_attack_ids(is_train, files, src_root, dst_root, name2cluster):
    if is_train:
        for file in files:
            path = os.path.join(src_root, file)
            cur_cluster = name2cluster[file]
            new_data = pd.read_pickle(path)
            new_data['cluster'] = cur_cluster
            new_data.to_pickle(os.path.join(dst_root, file))
    else:
        for file in files:
            file_path = os.path.join(src_root, file)
            data = pd.read_pickle(file_path)
            data['cluster'] = -2
            data.to_pickle(os.path.join(dst_root, file))


def label_cluster_normal_for_test(files, src_root, dst_root):
    for file in files:
        file_path = os.path.join(src_root, file)
        data = pd.read_pickle(file_path)
        data['cluster'] = -2
        data.to_pickle(os.path.join(dst_root, file))


def label_cluster_attack_for_test(files, src_root, dst_root):
    for file in files:
        file_path = os.path.join(src_root, file)
        data = pd.read_pickle(file_path)
        data['cluster'] = -2
        data.to_pickle(os.path.join(dst_root, file))


def label_cluster_attack_for_train(cluster_alg, files, src_root, dst_root, cluster_args=[]):
    # 对所有的攻击数据也进行聚类
    if cluster_alg == 'None':
        print('[INFO] Label train attack data, no cluster')
        logging.info('[INFO] Label train attack data, no cluster')
        for file in files:
            file_path = os.path.join(src_root, file)
            data = pd.read_pickle(file_path)
            data['cluster'] = 0
            data.to_pickle(os.path.join(dst_root, file))

    if cluster_alg == 'kmeans':
        print(f'[INFO] Label train attack data, use kmeans, k: {cluster_args[0]}')
        logging.info(f'[INFO] Label train attack data, use kmeans, k: {cluster_args[0]}')

        from sklearn.cluster import KMeans
        k = cluster_args[0]
        drop_minor = cluster_args[1]
        data = []
        for file in files:
            file_path = os.path.join(src_root, file)
            cur_data = pd.read_pickle(file_path)
            data.append(cur_data)
        data = pd.concat(data, axis=0)
        data['ngram'] = data['ngram'].apply(ast.literal_eval)

        feature = np.array(data['ngram'].values)
        feature_a = [np.array(i) for i in feature]
        feature = np.array(feature_a)

        model = KMeans(n_clusters=k, random_state=42)
        model.fit(feature)
        labels = model.labels_
        labels_pd = pd.DataFrame(labels, columns=['cluster'])
        tmp_res = pd.concat([data.reset_index(drop=True), labels_pd], axis=1)
        all_length = len(tmp_res)

        # 打印聚类结果
        print('== Cluster info: ')
        logging.info('== Cluster info:')
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique_labels, counts))
        sum_count = 0
        for label, count in cluster_counts.items():
            print(f"类别 {label}: {count}个")
            logging.info(f"类别 {label}: {count}个")
            sum_count += count

        # drop的时候考虑类别的数量，要不类别太多的时候很多类都会被drop掉，6.13增加
        all_length_no_noise_eachclass = len(tmp_res) / k

        # drop一些数据
        if drop_minor != 0:
            for i in np.unique(tmp_res['cluster']):
                data_i = tmp_res[tmp_res['cluster'] == i]
                if len(data_i) < all_length_no_noise_eachclass * drop_minor:
                    tmp_res = tmp_res[tmp_res['cluster'] != i]
                    print(f'drop class {i}, num: {len(data_i)}')
                    logging.info(f'drop class {i}, num: {len(data_i)}')
            print(f'[INFO] Before drop minor: {all_length}, after drop: {len(tmp_res)}')

        dst_name = f'attack_kmeans_k{k}_dm{drop_minor}.pickle'

        tmp_res.to_pickle(os.path.join(dst_root, dst_name))


def label_cluster_normal_for_train(cluster_alg, files, src_root, dst_root, cluster_args=[]):
    if cluster_alg == 'None':
        print('[INFO] Label train normal data, no cluster')
        logging.info('[INFO] Label train normal data, no cluster')
        for file in files:
            file_path = os.path.join(src_root, file)
            data = pd.read_pickle(file_path)
            data['cluster'] = 0
            data.to_pickle(os.path.join(dst_root, file))
    if cluster_alg == 'dbscan':
        print(f'[INFO] Label train normal data, use dbscan, eps: {cluster_args[0]}, minpts: {cluster_args[1]}')
        logging.info(f'[INFO] Label train normal data, use dbscan, eps: {cluster_args[0]}, minpts: {cluster_args[1]}')
        from sklearn.cluster import DBSCAN
        from sklearn.ensemble import RandomForestClassifier
        eps = cluster_args[0]
        minpts = cluster_args[1]
        drop_minor = cluster_args[2]
        data = []
        for file in files:
            file_path = os.path.join(src_root, file)
            cur_data = pd.read_pickle(file_path)
            data.append(cur_data)
        data = pd.concat(data, axis=0)
        data['ngram'] = data['ngram'].apply(ast.literal_eval)

        feature = np.array(data['ngram'].values)
        feature_a = [np.array(i) for i in feature]
        feature = np.array(feature_a)

        if len(data) > 50000:
            part_data = data.sample(n=50000, random_state=0)
            part_feature = np.array(part_data['ngram'].values)
            part_feature_a = [np.array(i) for i in part_feature]
            part_feature = np.array(part_feature_a)
            model = DBSCAN(eps=eps, min_samples=minpts)
            model.fit(part_feature)
            part_labels = model.labels_

            df_model = RandomForestClassifier()
            df_model.fit(part_feature, part_labels)

            labels = df_model.predict(feature)
        else:
            model = DBSCAN(eps=eps, min_samples=minpts)
            model.fit(feature)
            labels = model.labels_

        labels_pd = pd.DataFrame(labels, columns=['cluster'])
        tmp_res = pd.concat([data.reset_index(drop=True), labels_pd], axis=1)
        all_length = len(tmp_res)

        # 打印聚类结果
        print('== Cluster info: ')
        logging.info('== Cluster info:')
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique_labels, counts))
        sum_count = 0
        for label, count in cluster_counts.items():
            print(f"类别 {label}: {count}个")
            logging.info(f"类别 {label}: {count}个")
            sum_count += count

        tmp_res = tmp_res[tmp_res['cluster'] != -1]  # 去掉噪声数据

        all_length_no_noise = len(tmp_res)

        # drop一些数据
        if drop_minor != 0:
            for i in np.unique(tmp_res['cluster']):
                data_i = tmp_res[tmp_res['cluster'] == i]
                if len(data_i) < all_length_no_noise * drop_minor:
                    tmp_res = tmp_res[tmp_res['cluster'] != i]
                    print(f'drop class {i}, num: {len(data_i)}')
                    logging.info(f'drop class {i}, num: {len(data_i)}')
            print(f'[INFO] Before drop minor: {all_length}, after drop: {len(tmp_res)}')

        dst_name = f'normal_dbscan_e{eps}_mp{minpts}_dm{drop_minor}.pickle'

        tmp_res.to_pickle(os.path.join(dst_root, dst_name))

    if cluster_alg == 'kmeans':
        print(f'[INFO] Label train normal data, use kmeans, k: {cluster_args[0]}')
        logging.info(f'[INFO] Label train normal data, use kmeans, k: {cluster_args[0]}')

        from sklearn.cluster import KMeans
        k = cluster_args[0]
        drop_minor = cluster_args[1]
        data = []
        for file in files:
            file_path = os.path.join(src_root, file)
            cur_data = pd.read_pickle(file_path)
            data.append(cur_data)
        data = pd.concat(data, axis=0)
        data['ngram'] = data['ngram'].apply(ast.literal_eval)

        feature = np.array(data['ngram'].values)
        feature_a = [np.array(i) for i in feature]
        feature = np.array(feature_a)

        model = KMeans(n_clusters=k, random_state=42)
        model.fit(feature)
        labels = model.labels_
        labels_pd = pd.DataFrame(labels, columns=['cluster'])
        tmp_res = pd.concat([data.reset_index(drop=True), labels_pd], axis=1)
        all_length = len(tmp_res)

        # 打印聚类结果
        print('== Cluster info: ')
        logging.info('== Cluster info:')
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique_labels, counts))
        sum_count = 0
        for label, count in cluster_counts.items():
            print(f"类别 {label}: {count}个")
            logging.info(f"类别 {label}: {count}个")
            sum_count += count

        # drop的时候考虑类别的数量，要不类别太多的时候很多类都会被drop掉，6.13增加
        all_length_no_noise_eachclass = len(tmp_res) / k

        # drop一些数据
        if drop_minor != 0:
            for i in np.unique(tmp_res['cluster']):
                data_i = tmp_res[tmp_res['cluster'] == i]
                if len(data_i) < all_length_no_noise_eachclass * drop_minor:
                    tmp_res = tmp_res[tmp_res['cluster'] != i]
                    print(f'drop class {i}, num: {len(data_i)}')
                    logging.info(f'drop class {i}, num: {len(data_i)}')
            print(f'[INFO] Before drop minor: {all_length}, after drop: {len(tmp_res)}')

        dst_name = f'normal_kmeans_k{k}_dm{drop_minor}.pickle'

        tmp_res.to_pickle(os.path.join(dst_root, dst_name))


def normalize_row_new(row):
    # 将过长的数据处理掉
    return [(i - 54) / 2000 if i > 2054 else 1 for i in row]


class StaSeqTrafficNormalizedDataset(Dataset):
    def __init__(self, pickle_paths, label_dict, detail_label_dict, myconfig, create_scaler, scaler_path):
        # 只有在第一次训练的时候才创建scaler，后面的都是用的之前的
        # 序列信息
        self.data = []
        self.labels = []
        self.msks = []
        self.detail_labels = []
        self.clusters = []
        # 统计信息
        self.sta_data = []

        for pickle_path in pickle_paths:
            # print(pickle_path)
            df = pd.read_pickle(pickle_path)

            # 序列信息
            df['lengths'] = df['lengths'].apply(ast.literal_eval)
            df['directions'] = df['directions'].apply(ast.literal_eval)
            df['intervals'] = df['intervals'].apply(ast.literal_eval)
            # df['ngram'] = df['ngram'].apply(ast.literal_eval)   # ngram 在加载数据的时候就处理过了，这里就不用了

            df = df[df['cluster'] != -1]  # 噪声不考虑
            directions = df['lengths'].values
            # 对包长序列归一化
            df['new_directions'] = df['directions'].apply(normalize_row_new)
            lengths = df['new_directions'].values
            flow_num = len(lengths)
            seq, cur_msk = self.process_single(directions, lengths, myconfig.sequence_length)
            self.data.extend(seq)
            self.labels.extend([label_dict[os.path.basename(pickle_path)]] * flow_num)
            self.msks.extend(cur_msk)
            self.detail_labels.extend([detail_label_dict[os.path.basename(pickle_path)]] * flow_num)
            self.clusters.extend(df['cluster'].values)
            # tmp = np.unique(df["cluster"].values)

            # print(f'--- cluster {pickle_path}, {tmp}')

            # 统计信息
            sta = df[['bidirectional_duration_ms', 'bidirectional_packets', 'bidirectional_bytes', 'src2dst_packets',
                      'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes', 'bidirectional_min_ps',
                      'bidirectional_mean_ps', 'bidirectional_stddev_ps', 'bidirectional_max_ps',
                      'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms',
                      'bidirectional_max_piat_ms', 'bidirectional_syn_packets', 'bidirectional_cwr_packets',
                      'bidirectional_ece_packets', 'bidirectional_urg_packets', 'bidirectional_ack_packets',
                      'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_fin_packets']]
            self.sta_data.append(sta)

        self.sta_data = pd.concat(self.sta_data, axis=0)
        if create_scaler:
            print(f'[INFO] Create new MinMaxScaler with {pickle_paths[0]} ....')
            scaler = MinMaxScaler()
            self.sta_scalered = scaler.fit_transform(self.sta_data)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        else:
            if os.path.exists(scaler_path):
                print(f'[INFO] Use existing scaler, {scaler_path}')
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                print(f'[INFO] Claim scaler exists, but not, Create new MinMaxScaler with {pickle_paths[0]} ....')
                scaler = MinMaxScaler()
                self.sta_scalered = scaler.fit_transform(self.sta_data)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            self.sta_scalered = scaler.transform(self.sta_data)
        self.sta_scalered = torch.tensor(self.sta_scalered, dtype=torch.float32)

    def process_single(self, directions, lengths, target_len):
        new_seq = []
        new_masklen = []
        for (direction, length) in zip(directions, lengths):
            assert len(direction) == len(length)
            if len(direction) < target_len:
                new_masklen.append(len(direction))
            else:
                new_masklen.append(target_len)
                direction = direction[:target_len]
                length = length[:target_len]

            vector = torch.zeros(2, target_len)
            for i, (d, l) in enumerate(zip(direction, length)):
                if d == 1:
                    vector[0, i] = l
                else:
                    vector[1, i] = l
            new_seq.append(vector)

        return new_seq, new_masklen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.msks[idx], self.detail_labels[idx], self.clusters[idx], \
            self.sta_scalered[idx]


def get_sample_weights_for_multiple_dataset(dataset):
    labels = []
    clusters = []
    for ds in dataset:
        labels.extend(ds.labels)
        clusters.extend(ds.clusters)

    # 由于label和cluster不一定是连续的，后面取值会出现问题，这里就换成了max来计算
    mat = [[0] * int(max(labels) + 1) for _ in range(int(max(clusters) + 1))]
    # mat = [[0] * len(np.  unique(labels)) for _ in range(len(np.unique(clusters)))]
    for l in np.unique(labels):
        for c in np.unique(clusters):
            cur_count = np.sum((np.array(labels) == l) & (np.array(clusters) == c))
            if cur_count != 0:
                mat[c][l] = 1. / cur_count
            print(f'l {l}, c {c}, {cur_count}')
    sample_weights = [0] * len(labels)
    for i in range(len(labels)):
        sample_weights[i] = mat[clusters[i]][labels[i]]
    print('weight matrix: ')
    print(np.array(mat))
    return np.array(sample_weights)


def get_sample_weights(dataset):
    labels = dataset.labels
    clusters = dataset.clusters
    # 由于label和cluster不一定是连续的，后面取值会出现问题，这里就换成了max来计算
    mat = [[0] * int(max(labels) + 1) for _ in range(int(max(clusters) + 1))]
    # mat = [[0] * len(np.  unique(labels)) for _ in range(len(np.unique(clusters)))]
    for l in np.unique(labels):
        for c in np.unique(clusters):
            cur_count = np.sum((np.array(labels) == l) & (np.array(clusters) == c))
            if cur_count != 0:
                mat[c][l] = 1. / cur_count
            print(f'l {l}, c {c}, {cur_count}')
    sample_weights = [0] * len(labels)
    for i in range(len(labels)):
        sample_weights[i] = mat[clusters[i]][labels[i]]
    print('weight matrix: ')
    print(np.array(mat))
    return np.array(sample_weights)


class MyModelStaAE(nn.Module):
    def __init__(self):
        super(MyModelStaAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(23, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 23)
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return latent, output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MyModelTransformer(nn.Module):
    def __init__(self, seq_len, feature_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 dropout):
        super(MyModelTransformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                       batch_first=True),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                       batch_first=True),
            num_layers=num_decoder_layers
        )
        self.input_embedding = nn.Linear(feature_size, d_model)
        self.output_projection = nn.Linear(d_model, feature_size)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.fc1 = nn.Linear(seq_len * d_model, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, src, return_memory=False):
        batch_size = src.shape[0]
        src = self.input_embedding(src)  # Embed input features to d_model dimensions
        src = self.pos_encoder(src)
        memory = self.encoder(src)  # Encoder output (memory)

        flat_memory = memory.view(batch_size, self.seq_len * self.d_model)
        latent = F.relu(self.fc1(flat_memory))
        latent = self.fc2(latent)

        output = self.decoder(memory, memory)  # Decoder output
        output = self.output_projection(output)
        return output, latent


# 相比于6/myutil.py的区别是，传递了margin作为参数，而不是直接从myconfig中获取
class ContraLossEucNewM2(nn.Module):
    def __init__(self, myconfig, margin):
        super(ContraLossEucNewM2, self).__init__()
        self.margin = margin

    def forward(self, inputs, labels, clusters):
        # print('len inputs: ', len(inputs))
        # 扩展X到[n, 1, m]和[1, n, m]，然后广播相减得到[n, n, m]的差值tensor
        diff = inputs.unsqueeze(1) - inputs.unsqueeze(0)
        # 计算欧式距离：对差值平方后沿特征维度求和，然后开方
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-08)

        # 确保labels和clusters是一维且具有相同的长度n
        assert labels.dim() == 1 and clusters.dim() == 1
        assert labels.size(0) == clusters.size(0)
        # 将labels和clusters合并为一个二维tensor，形状为[n, 2]
        combined = torch.stack([labels, clusters], dim=1)
        # 扩展combined以进行广播，分别为[n, 1, 2]和[1, n, 2]
        # 广播比较每对组合，得到一个[n, n, 2]的比较结果
        comparison = combined.unsqueeze(1) == combined.unsqueeze(0)
        # 仅当label和cluster同时相等时，两个组合才相等
        # 因此，我们需要沿着最后一个维度检查所有True（dim=-1），得到一个[n, n]的相等矩阵
        equal_matrix = comparison.all(dim=-1)
        # 将相等的位置设为0，不相等的位置设为1
        pair_matrix = (~equal_matrix).float()
        # print('pari: ', pair_matrix)
        upper_triangular = torch.triu(dist_matrix, diagonal=1)

        pos_loss = upper_triangular * (1 - pair_matrix)
        pos_loss = pos_loss ** 2

        neg_loss = upper_triangular * pair_matrix
        neg_loss = torch.clamp(neg_loss, max=self.margin)
        neg_loss = (self.margin - neg_loss) ** 2
        neg_loss = torch.triu(neg_loss, diagonal=1)

        # print('pos num: ', torch.triu(1-pair_matrix, diagonal=1).sum())
        # print('neg num: ', torch.triu(pair_matrix, diagonal=1).sum())

        neg_loss = neg_loss * pair_matrix
        # print('pos: ', pos_loss.sum(), 'neg: ', neg_loss.sum())

        loss = (pos_loss.sum() + neg_loss.sum()) / (len(inputs) * (len(inputs) - 1) / 2)

        return loss


def train_model(myconfig, train_loader, model_root, alpha_seq_contra, alpha_seq_recon, alpha_sta_contra,
                alpha_sta_recon):
    # ===================================================
    # 1. 训练模型
    # ===================================================
    print('[INFO] Begin train model')
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward,
                                   dropout=myconfig.dropout).to(myconfig.device)

    seq_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.seq_margin)
    seq_reconst_criterion = nn.MSELoss(reduction='sum')
    seq_optimizer = optim.AdamW(seq_model.parameters(), lr=myconfig.seq_lr)
    seq_scheduler = CosineAnnealingLR(seq_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.seq_eta_min)

    sta_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.sta_margin)
    sta_reconst_criterion = nn.MSELoss(reduction='sum')
    sta_optimizer = optim.AdamW(sta_model.parameters(), lr=myconfig.sta_lr)
    sta_scheduler = CosineAnnealingLR(sta_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.sta_eta_min)

    for epoch in range(myconfig.train_epoch):
        sta_model.train()
        seq_model.train()

        sta_contra_loss_epoch = 0
        sta_recon_loss_epoch = 0
        seq_contra_loss_epoch = 0
        seq_recon_loss_epoch = 0

        for inputs, labels, masks, detaillabels, clusters, stas in train_loader:
            # 使用序列信息进行训练
            inputs = inputs.transpose(1, 2)
            inputs = inputs.to(myconfig.device)
            labels = labels.to(myconfig.device)
            clusters = clusters.to(myconfig.device)

            seq_optimizer.zero_grad()

            seq_recon, seq_latent = seq_model(inputs)
            seq_contra_loss = seq_contra_criterion(seq_latent, labels, clusters)
            seq_recon_loss = seq_reconst_criterion(seq_recon, inputs)
            # seq_loss = seq_contra_loss + seq_recon_loss / 20
            seq_loss = seq_contra_loss * alpha_seq_contra + seq_recon_loss * alpha_seq_recon

            seq_loss.backward()
            seq_optimizer.step()

            seq_contra_loss_epoch += seq_contra_loss
            seq_recon_loss_epoch += seq_recon_loss

            # 使用统计信息进行训练
            stas = stas.to(myconfig.device)

            sta_optimizer.zero_grad()

            sta_latent, sta_recon = sta_model(stas)
            sta_contra_loss = sta_contra_criterion(sta_latent, labels, clusters)
            sta_recon_loss = sta_reconst_criterion(sta_recon, stas)
            # sta_loss = sta_contra_loss + sta_recon_loss
            sta_loss = sta_contra_loss * alpha_sta_contra + sta_recon_loss * alpha_sta_recon
            sta_loss.backward()
            sta_optimizer.step()

            sta_contra_loss_epoch += sta_contra_loss
            sta_recon_loss_epoch += sta_recon_loss

        print(f'Epoch {epoch}')
        print(
            f'Seq Train Loss: {seq_recon_loss_epoch + seq_contra_loss_epoch}, Recon Loss: {seq_recon_loss_epoch}, Contra Loss: {seq_contra_loss_epoch}')
        print(
            f'Sta Train Loss: {sta_recon_loss_epoch + sta_contra_loss_epoch}, Recon Loss: {sta_recon_loss_epoch}, Contra Loss: {sta_contra_loss_epoch}')
        logging.info(f'Epoch {epoch}')
        logging.info(
            f'Seq Train Loss: {seq_recon_loss_epoch + seq_contra_loss_epoch}, Recon Loss: {seq_recon_loss_epoch}, Contra Loss: {seq_contra_loss_epoch}')
        logging.info(
            f'Sta Train Loss: {sta_recon_loss_epoch + sta_contra_loss_epoch}, Recon Loss: {sta_recon_loss_epoch}, Contra Loss: {sta_contra_loss_epoch}')

        torch.save(seq_model.state_dict(), os.path.join(model_root, 'seq_model_' + str(epoch) + '.pth'))
        torch.save(sta_model.state_dict(), os.path.join(model_root, 'sta_model_' + str(epoch)) + '.pth')
        seq_scheduler.step()
        sta_scheduler.step()


def get_seqmid_info_of_train_transformer_sta(model, train_loader, myconfig, save_data_root):
    singleloss_criterion = nn.MSELoss(reduction='none')

    all_recon_loss, all_feature, all_label, all_cluster, all_detail_label = [], [], [], [], []

    for inputs, labels, masks, detaillabels, clusters, stas in train_loader:
        inputs = inputs.transpose(1, 2)
        inputs = inputs.to(myconfig.device)
        masks = masks.to(myconfig.device)
        labels = labels.to(myconfig.device)
        clusters = clusters.to(myconfig.device)
        detaillabels = detaillabels.to(myconfig.device)
        # inputs = inputs.unsqueeze(1)

        recon, latent = model(inputs)

        # 计算重构损失
        # batch_size, width, height = inputs.size()
        recon_masks = torch.zeros_like(recon)
        for i, mask_len in enumerate(masks):
            recon_masks[i, :mask_len, :] = 1
        masked_recon = recon * recon_masks
        masked_inputs = inputs * recon_masks

        # 获得每个样本的重构损失
        recon_loss = singleloss_criterion(masked_recon, masked_inputs)

        per_recon_loss = recon_loss.sum(dim=[1, 2]) / (masks * 2)

        all_recon_loss.extend(per_recon_loss.detach().cpu().numpy())
        all_label.extend(labels.detach().cpu().numpy())
        all_cluster.extend(clusters.detach().cpu().numpy())
        all_detail_label.extend(detaillabels.detach().cpu().numpy())
        all_feature.extend(latent.detach().cpu().numpy())

    new_data = [all_feature, all_recon_loss, all_label, all_detail_label, all_cluster]
    new_data = pd.DataFrame(new_data).transpose()
    new_data.columns = ['feature', 'recon_loss', 'label', 'detail_label', 'cluster']

    if not os.path.exists(save_data_root):
        print(f'[Attention] Root not exists, regenerate: {save_data_root}')
        os.makedirs(save_data_root)

    new_data.to_pickle(os.path.join(save_data_root, 'train_mid_info.pickle'))


def get_stamid_info_of_train_ae_sta(sta_model, train_loader, myconfig, save_data_root):
    singleloss_criterion = nn.MSELoss(reduction='none')

    all_recon_loss, all_feature, all_label, all_cluster, all_detail_label = [], [], [], [], []
    for inputs, labels, masks, detaillabels, clusters, stas in train_loader:
        stas = stas.to(myconfig.device)
        sta_latent, sta_recon = sta_model(stas)
        recon_loss = singleloss_criterion(sta_recon, stas)
        per_recon_loss = recon_loss.sum(dim=[1])

        all_recon_loss.extend(per_recon_loss.detach().cpu().numpy())
        all_label.extend(labels.detach().cpu().numpy())
        all_cluster.extend(clusters.detach().cpu().numpy())
        all_detail_label.extend(detaillabels.detach().cpu().numpy())
        all_feature.extend(sta_latent.detach().cpu().numpy())

    new_data = [all_feature, all_recon_loss, all_label, all_detail_label, all_cluster]
    new_data = pd.DataFrame(new_data).transpose()
    new_data.columns = ['feature', 'recon_loss', 'label', 'detail_label', 'cluster']

    if not os.path.exists(save_data_root):
        print(f'[Attention] save data root not exists, regenerate, {save_data_root}')
        os.makedirs(save_data_root)

    new_data.to_pickle(os.path.join(save_data_root, 'train_stamid_info.pickle'))


def fit_norm(data):
    f = Fitter(data, distributions='norm', timeout=3600)
    f.fit()
    # print(f.get_best())
    paras = f.get_best()['norm']
    return paras, f.df_errors


def get_sta_info_of_train_95_stadata(save_data_root, re_generate_sta_info=True):
    # 获取流量的统计特征的统计信息，基于get_sta_info_of_train_95实现
    if (
            not os.path.exists(
                os.path.join(save_data_root, 'train_sta_info_stadata.pickle'))) or re_generate_sta_info:
        if not os.path.exists(os.path.join(save_data_root, 'train_sta_info_stadata.pickle')):
            print('[Attention] train_sta_info.pickle does not exists, regenerate.')
        elif re_generate_sta_info:
            print('[Attention] train_sta_info.pickle is required to regenerate.')

        new_data = pd.read_pickle(os.path.join(save_data_root, 'train_stamid_info.pickle'))
        label_unique = np.unique(new_data['label'])
        cluster_unique = np.unique(new_data['cluster'])

        print(f'label unique: {label_unique}')
        print(f'cluster unique: {cluster_unique}')

        # 保存每一个细化的类的中心、类内样本到中心的距离、重构损失
        center_dict = {}
        distance_dict = {}
        reconloss_dict = {}

        for l in label_unique:
            for c in cluster_unique:
                tmp_data = new_data[(new_data['label'] == l) & (new_data['cluster'] == c)]
                print(f'({l}, {c}): {len(tmp_data)}')
                if len(tmp_data) == 0:
                    continue
                features = np.stack(tmp_data['feature'].values)
                mean_feature = np.mean(features, axis=0)
                center_dict[(l, c)] = mean_feature

                distances = np.linalg.norm(features - mean_feature, axis=1)
                distance_dict[(l, c)] = distances

                reconl = tmp_data['recon_loss'].values
                reconloss_dict[(l, c)] = reconl

        train_sta_info = {'center_dict': center_dict, 'distance_dict': distance_dict, 'reconloss_dict': reconloss_dict}
        with open(os.path.join(save_data_root, 'train_sta_info_stadata.pickle'), 'wb') as f:
            pickle.dump(train_sta_info, f)

    print('[INFO] Fit distance to a distribution use 95 data (sta).')
    with open(os.path.join(save_data_root, 'train_sta_info_stadata.pickle'), 'rb') as f:
        train_sta_info = pickle.load(f)

    # 保存拟合后的分布的参数信息
    train_distri_para_info = {}

    distance_dict = train_sta_info['distance_dict']
    # 将正态分布的参数存储进去,每个类的样本到其类心的距离
    dist_norm_para_dict = {}
    print('fit distances')
    for key in distance_dict.keys():
        print(key)
        cur_dis = distance_dict[key]
        cur_dis = sorted(cur_dis)[:int(len(cur_dis) * 0.95)]
        if len(cur_dis) < 1:
            continue
        paras, errors = fit_norm(cur_dis)
        dist_norm_para_dict[key] = paras
        print(errors)

    train_distri_para_info['dist_distri'] = dist_norm_para_dict

    print('fit recon errors')
    reconloss_dict = train_sta_info['reconloss_dict']
    relos_norm_para_dict = {}
    all_loss = []
    for key in reconloss_dict.keys():
        print(key)
        cur_loss = reconloss_dict[key].astype(np.float64)
        cur_loss = sorted(cur_loss)[:int(len(cur_loss) * 0.95)]
        if len(cur_loss) < 1:
            continue
        paras, errors = fit_norm(cur_loss)
        relos_norm_para_dict[key] = paras
        print(errors)
        all_loss.extend(cur_loss)
    # 也拟合一个所有数据的重构损失
    paras, errors = fit_norm(all_loss)
    relos_norm_para_dict[(-1, -1)] = paras
    train_distri_para_info['reclos_distri'] = relos_norm_para_dict

    with open(os.path.join(save_data_root, 'train_distri_para_info_95_stadata.pickle'), 'wb') as f:
        pickle.dump(train_distri_para_info, f)


def get_sta_info_of_train_95(save_data_root, re_generate_sta_info=True):
    # 如果统计特征（类的中心、类内样本到中心的距离、重构损失）不存在，或者需要重新生成
    if (not os.path.exists(os.path.join(save_data_root, 'train_sta_info.pickle'))) or re_generate_sta_info:
        if not os.path.exists(os.path.join(save_data_root, 'train_sta_info.pickle')):
            print('[Attention] train_sta_info.pickle does not exists, regenerate.')
        elif re_generate_sta_info:
            print('[Attention] train_sta_info.pickle is required to regenerate.')

        new_data = pd.read_pickle(os.path.join(save_data_root, 'train_mid_info.pickle'))

        label_unique = np.unique(new_data['label'])
        cluster_unique = np.unique(new_data['cluster'])

        print(f'label unique: {label_unique}')
        print(f'cluster unique: {cluster_unique}')

        # 保存每一个细化的类的中心、类内样本到中心的距离、重构损失
        center_dict = {}
        distance_dict = {}
        reconloss_dict = {}

        for l in label_unique:
            for c in cluster_unique:
                tmp_data = new_data[(new_data['label'] == l) & (new_data['cluster'] == c)]
                print(f'({l}, {c}): {len(tmp_data)}')
                if len(tmp_data) == 0:
                    continue
                features = np.stack(tmp_data['feature'].values)
                mean_feature = np.mean(features, axis=0)
                center_dict[(l, c)] = mean_feature

                distances = np.linalg.norm(features - mean_feature, axis=1)
                distance_dict[(l, c)] = distances

                reconl = tmp_data['recon_loss'].values
                reconloss_dict[(l, c)] = reconl

        train_sta_info = {'center_dict': center_dict, 'distance_dict': distance_dict, 'reconloss_dict': reconloss_dict}
        with open(os.path.join(save_data_root, 'train_sta_info.pickle'), 'wb') as f:
            pickle.dump(train_sta_info, f)

    print('[INFO] Fit distance to a distribution use 95 data.')
    with open(os.path.join(save_data_root, 'train_sta_info.pickle'), 'rb') as f:
        train_sta_info = pickle.load(f)

    # 保存拟合后的分布的参数信息
    train_distri_para_info = {}

    distance_dict = train_sta_info['distance_dict']
    # 将正态分布的参数存储进去,每个类的样本到其类心的距离
    dist_norm_para_dict = {}
    print('fit distances')
    for key in distance_dict.keys():
        print(key)
        cur_dis = distance_dict[key]
        cur_dis = sorted(cur_dis)[:int(len(cur_dis) * 0.95)]
        if len(cur_dis) < 1:
            continue
        paras, errors = fit_norm(cur_dis)
        dist_norm_para_dict[key] = paras
        print(errors)

    train_distri_para_info['dist_distri'] = dist_norm_para_dict

    print('fit recon errors')
    reconloss_dict = train_sta_info['reconloss_dict']
    relos_norm_para_dict = {}
    all_loss = []
    for key in reconloss_dict.keys():
        print(key)
        cur_loss = reconloss_dict[key].astype(np.float64)
        cur_loss = sorted(cur_loss)[:int(len(cur_loss) * 0.95)]
        if len(cur_loss) < 1:
            continue
        paras, errors = fit_norm(cur_loss)
        relos_norm_para_dict[key] = paras
        print(errors)
        all_loss.extend(cur_loss)
    # 也拟合一个所有数据的重构损失
    paras, errors = fit_norm(all_loss)
    relos_norm_para_dict[(-1, -1)] = paras
    train_distri_para_info['reclos_distri'] = relos_norm_para_dict

    with open(os.path.join(save_data_root, 'train_distri_para_info_95.pickle'), 'wb') as f:
        pickle.dump(train_distri_para_info, f)


def get_seqmid_info_of_test_transformer_sta(model, test_loader, myconfig, save_data_root):
    singleloss_criterion = nn.MSELoss(reduction='none')

    all_recon_loss, all_feature, all_label, all_detail_label = [], [], [], []

    for inputs, labels, masks, detaillabels, _, stas in test_loader:
        inputs = inputs.transpose(1, 2)
        inputs = inputs.to(myconfig.device)
        masks = masks.to(myconfig.device)
        labels = labels.to(myconfig.device)

        recon, latent = model(inputs)

        # 计算重构损失
        recon_masks = torch.zeros_like(recon)
        for i, mask_len in enumerate(masks):
            recon_masks[i, :mask_len, :] = 1
        masked_recon = recon * recon_masks
        masked_inputs = inputs * recon_masks

        # 获得每个样本的重构损失
        recon_loss = singleloss_criterion(masked_recon, masked_inputs)

        per_recon_loss = recon_loss.sum(dim=[1, 2]) / (masks * 2)

        all_recon_loss.extend(per_recon_loss.detach().cpu().numpy())
        all_label.extend(labels.detach().cpu().numpy())
        all_detail_label.extend(detaillabels.detach().cpu().numpy())
        all_feature.extend(latent.detach().cpu().numpy())

    new_test_data = [all_feature, all_recon_loss, all_label, all_detail_label]
    new_test_data = pd.DataFrame(new_test_data).transpose()
    new_test_data.columns = ['feature', 'recon_loss', 'label', 'detail_label']

    new_test_data.to_pickle(os.path.join(save_data_root, 'test_seqmid_info.pickle'))


def get_stamid_info_of_test_ae_sta(sta_model, test_loader, myconfig, save_data_root):
    singleloss_criterion = nn.MSELoss(reduction='none')

    all_recon_loss, all_feature, all_label, all_detail_label = [], [], [], []

    for inputs, labels, masks, detaillabels, _, stas in test_loader:
        labels = labels.to(myconfig.device)
        stas = stas.to(myconfig.device)

        sta_latent, sta_recon = sta_model(stas)
        recon_loss = singleloss_criterion(sta_recon, stas)
        per_recon_loss = recon_loss.sum(dim=[1])

        all_recon_loss.extend(per_recon_loss.detach().cpu().numpy())
        all_label.extend(labels.detach().cpu().numpy())
        all_detail_label.extend(detaillabels.detach().cpu().numpy())
        all_feature.extend(sta_latent.detach().cpu().numpy())

    new_test_data = [all_feature, all_recon_loss, all_label, all_detail_label]
    new_test_data = pd.DataFrame(new_test_data).transpose()
    new_test_data.columns = ['feature', 'recon_loss', 'label', 'detail_label']

    new_test_data.to_pickle(os.path.join(save_data_root, 'test_stamid_info.pickle'))


def cal_distance(row, tar_feat):
    feat = row['feature']
    dist = np.sqrt(np.sum((feat - tar_feat) ** 2))
    return dist


def get_distance_to_center_test_seq(train_save_data_root, test_save_data_root):
    # # 测试样本的隐含特征
    new_test_data = pd.read_pickle(os.path.join(test_save_data_root, 'test_seqmid_info.pickle'))
    # 训练样本的簇心
    with open(os.path.join(train_save_data_root, 'train_sta_info.pickle'), 'rb') as f:
        train_sta_info = pickle.load(f)

    cluster_centers = train_sta_info['center_dict']

    for key, value in cluster_centers.items():
        new_row = 'dist_' + str(key)
        new_test_data[new_row] = new_test_data.apply(cal_distance, args=(value,), axis=1)

    new_test_data.to_pickle(os.path.join(test_save_data_root, 'test_recloss_distancetocenter_seq.pickle'))


def get_distance_to_center_test_sta(train_save_data_root, test_save_data_root):
    # # 测试样本的隐含特征
    new_test_data = pd.read_pickle(os.path.join(test_save_data_root, 'test_stamid_info.pickle'))
    # 训练样本的簇心
    with open(os.path.join(train_save_data_root, 'train_sta_info_stadata.pickle'), 'rb') as f:
        train_sta_info = pickle.load(f)

    cluster_centers = train_sta_info['center_dict']

    for key, value in cluster_centers.items():
        new_row = 'dist_' + str(key)
        new_test_data[new_row] = new_test_data.apply(cal_distance, args=(value,), axis=1)

    new_test_data.to_pickle(os.path.join(test_save_data_root, 'test_recloss_distancetocenter_sta.pickle'))


def get_norm_par(value):
    if isinstance(value, dict):
        return value['loc'], value['scale']
    else:
        return value[0], value[1]


def get_ksigma(x, loc, scale):
    return round(abs(x - loc) / scale, 4)


def get_test_result_ksigma_95_seq(train_save_data_root, test_save_data_root, res_root):
    with open(os.path.join(train_save_data_root, 'train_distri_para_info_95.pickle'), 'rb') as f:
        train_distri_para_info = pickle.load(f)
    with open(os.path.join(test_save_data_root, 'test_recloss_distancetocenter_seq.pickle'), 'rb') as f:
        new_test_data = pickle.load(f)

    dist_distri = train_distri_para_info['dist_distri']
    for key, value in dist_distri.items():
        ori_row = 'dist_' + str(key)
        dst_row = 'ksigma_dist_' + str(key)
        v1, v2 = get_norm_par(value)
        new_test_data[dst_row] = new_test_data[ori_row].apply(lambda x: get_ksigma(x, v1, v2))

    reclos_distri = train_distri_para_info['reclos_distri']
    for key, value in reclos_distri.items():
        ori_row = 'recon_loss'
        dst_row = 'ksigma_reclos_' + str(key)
        v1, v2 = get_norm_par(value)
        new_test_data[dst_row] = new_test_data[ori_row].apply(lambda x: get_ksigma(x, v1, v2))

    final_res_data = new_test_data.drop(['feature'], axis=1)

    final_res_data.to_csv(os.path.join(res_root, 'test_result_ksigma_95_seq.csv'), index=False)


def get_test_result_ksigma_95_sta(train_save_data_root, test_save_data_root, res_root):
    with open(os.path.join(train_save_data_root, 'train_distri_para_info_95_stadata.pickle'), 'rb') as f:
        train_distri_para_info = pickle.load(f)
    with open(os.path.join(test_save_data_root, 'test_recloss_distancetocenter_sta.pickle'), 'rb') as f:
        new_test_data = pickle.load(f)

    dist_distri = train_distri_para_info['dist_distri']
    for key, value in dist_distri.items():
        ori_row = 'dist_' + str(key)
        dst_row = 'ksigma_dist_' + str(key)
        v1, v2 = get_norm_par(value)
        new_test_data[dst_row] = new_test_data[ori_row].apply(lambda x: get_ksigma(x, v1, v2))
    print(dist_distri)

    reclos_distri = train_distri_para_info['reclos_distri']
    for key, value in reclos_distri.items():
        ori_row = 'recon_loss'
        dst_row = 'ksigma_reclos_' + str(key)
        v1, v2 = get_norm_par(value)
        new_test_data[dst_row] = new_test_data[ori_row].apply(lambda x: get_ksigma(x, v1, v2))
    print(reclos_distri)

    final_res_data = new_test_data.drop(['feature'], axis=1)

    final_res_data.to_csv(os.path.join(res_root, 'test_result_ksigma_95_sta.csv'), index=False)


def split_normal_attack(all_path, normal_path, attack_path):
    all_data = pd.read_csv(all_path)
    normal_data = all_data[all_data['label'] == 0]
    normal_data.to_csv(normal_path, index=False)
    attack_data = all_data[all_data['label'] == 1]
    attack_data.to_csv(attack_path, index=False)
    print('split normal and attack')


def get_min_dst_correspond_k(row, src_col):
    # print(f'row: {row}')
    tmp = row[src_col]
    # print(f'tmp: {tmp}')
    new_col = 'ksigma_' + tmp
    # print(f'rowwwwwwwwwww: {row[new_col]}')
    return row[new_col]


def get_dist_kdist_equal(row):
    # 最近的距离和最小的ksigdist所对应的类别是否相等，相等返回1，不相等0
    tmp1 = row['min_dist_col'].split('_')[-1]
    tmp2 = row['min_ksigmadist_col'].split('_')[-1]
    if tmp1 == tmp2:
        return 1
    else:
        return 0


def get_k(row, src_col):
    tmp = row[src_col]
    new_col = 'ksigma_reclos_' + tmp.split('_')[-1]
    return row[new_col]


def get_class(row):
    cl = row.split('_')[-1].strip('()').split(',')[0]
    return cl


def get_uncertain(row, sigma):
    """如果在n sigma之外，那说明结果是不准确的，返回1"""
    if row > sigma:
        return 1
    else:
        return 0


def generate_new_data_new(ori_path, dst_path):
    ori_data = pd.read_csv(ori_path)

    print(f'len ori data: {len(ori_data)}')

    cols_all = ori_data.columns

    dist_cols = []
    ksigmadist_cols = []
    for item in cols_all:
        if item.startswith('dist_('):
            dist_cols.append(item)
        elif item.startswith('ksigma_dist_('):
            ksigmadist_cols.append(item)

    # 距离的方差
    ori_data['dist_var'] = ori_data[dist_cols].var(axis=1)

    # 最近的距离所在的列
    ori_data['min_dist_col'] = ori_data[dist_cols].idxmin(axis=1)

    # 最小的距离ksigma所在的列
    ori_data['min_ksigmadist_col'] = ori_data[ksigmadist_cols].idxmin(axis=1)

    # 最近的距离
    ori_data['min_dist'] = ori_data[dist_cols].min(axis=1)
    # 最小的距离ksigma
    ori_data['min_dist_ksigma'] = ori_data[ksigmadist_cols].min(axis=1)
    # 距离最小的类对应距离的ksigma
    ori_data['min_dist_corr_distksigma'] = ori_data.apply(lambda x: get_min_dst_correspond_k(x, 'min_dist_col'), axis=1)
    # ori_data['min_dist_corr_distksigma'] = ori_data.apply(lambda x: print(x), axis=1)

    # 最近的距离和最小的ksigdist所对应的类别（小簇）是否相等
    ori_data['dist_equal_ksigdist'] = ori_data.apply(lambda x: get_dist_kdist_equal(x), axis=1)

    # 距离最小的类的重构损失是多少
    ori_data['min_dist_recon_k'] = ori_data.apply(lambda x: get_k(x, 'min_dist_col'), axis=1)
    # 最小的距离ksigma对应的重构损失是多少k
    ori_data['min_ksigmadist_recon_k'] = ori_data.apply(lambda x: get_k(x, 'min_ksigmadist_col'), axis=1)

    # 根据距离（sigma）判断样本属于哪个类
    ori_data['dist_class'] = ori_data['min_dist_col'].apply(get_class)
    ori_data['distksigma_class'] = ori_data['min_ksigmadist_col'].apply(get_class)

    # 根据最小的距离对应的ksigma判断为不确定
    ori_data['dist_uncertain_2'] = ori_data['min_dist_corr_distksigma'].apply(get_uncertain, sigma=2)
    ori_data['dist_uncertain_3'] = ori_data['min_dist_corr_distksigma'].apply(get_uncertain, sigma=3)

    # 根据最小的sigma判断是否为不确定
    ori_data['ksigmadist_uncertain_2'] = ori_data['min_dist_ksigma'].apply(get_uncertain, sigma=2)
    ori_data['ksigmadist_uncertain_3'] = ori_data['min_dist_ksigma'].apply(get_uncertain, sigma=3)

    # 测试样本的重构损失是否在距离最近的类的重构损失的n-sigma里
    ori_data['cla_md_recon_uncertain_2'] = ori_data['min_dist_recon_k'].apply(get_uncertain, sigma=2)
    ori_data['cla_md_recon_uncertain_3'] = ori_data['min_dist_recon_k'].apply(get_uncertain, sigma=3)

    # 测试样本的重构损失是否在最距离k-sigma最近的那个类的重构损失的n-sigma里
    ori_data['cla_mkd_recon_uncertain_2'] = ori_data['min_ksigmadist_recon_k'].apply(get_uncertain, sigma=2)
    ori_data['cla_mkd_recon_uncertain_3'] = ori_data['min_ksigmadist_recon_k'].apply(get_uncertain, sigma=3)

    # 测试样本的重构损失是否在整体的重构损失的n sigma里
    ori_data['all_recon_uncertain_2'] = ori_data['ksigma_reclos_(-1, -1)'].apply(get_uncertain, sigma=2)
    ori_data['all_recon_uncertain_3'] = ori_data['ksigma_reclos_(-1, -1)'].apply(get_uncertain, sigma=3)

    ori_data.to_csv(dst_path, index=False)


def generate_eval_pre_95_data_seq(res_root):
    """ 使用95%的数据，生成最终决策前的中间结果，例如是不是certain这些，把myutil中的 eval_test_result拆开了，评估另外放一个函数"""
    test_result_path = os.path.join(res_root, 'test_result_ksigma_95_seq.csv')
    normal_path = os.path.join(res_root, 'normal_test_result_ksigma_95_seq.csv')
    attack_path = os.path.join(res_root, 'attack_test_result_ksigma_95_seq.csv')
    normal_deep_new_path = os.path.join(res_root, 'normal_test_result_ksigma_95deep_new_seq.csv')
    attack_deep_new_path = os.path.join(res_root, 'attack_test_result_ksigma_95deep_new_seq.csv')
    split_normal_attack(test_result_path, normal_path, attack_path)
    generate_new_data_new(normal_path, normal_deep_new_path)
    generate_new_data_new(attack_path, attack_deep_new_path)


def generate_eval_pre_95_data_sta(res_root):
    """ 使用95%的数据，生成最终决策前的中间结果，例如是不是certain这些，把myutil中的 eval_test_result拆开了，评估另外放一个函数"""
    test_result_path = os.path.join(res_root, 'test_result_ksigma_95_sta.csv')
    normal_path = os.path.join(res_root, 'normal_test_result_ksigma_95_sta.csv')
    attack_path = os.path.join(res_root, 'attack_test_result_ksigma_95_sta.csv')
    normal_deep_new_path = os.path.join(res_root, 'normal_test_result_ksigma_95deep_new_sta.csv')
    attack_deep_new_path = os.path.join(res_root, 'attack_test_result_ksigma_95deep_new_sta.csv')
    split_normal_attack(test_result_path, normal_path, attack_path)
    generate_new_data_new(normal_path, normal_deep_new_path)
    generate_new_data_new(attack_path, attack_deep_new_path)


def eval_new_data_new(file_path, write_to_log=True, use_recon=False):
    data = pd.read_csv(file_path)

    detail_labels = np.unique(data['detail_label'])

    detail_data = data

    # 通过最小的距离判定确定性
    dist_certain_2 = detail_data[detail_data['dist_uncertain_2'] == 0]
    dist_certain_3 = detail_data[detail_data['dist_uncertain_3'] == 0]

    dist_certain_2_true_dist = dist_certain_2[dist_certain_2['label'] == dist_certain_2['dist_class']]
    dist_certain_2_true_ksigma = dist_certain_2[dist_certain_2['label'] == dist_certain_2['distksigma_class']]

    dist_certain_3_true_dist = dist_certain_3[dist_certain_3['label'] == dist_certain_3['dist_class']]
    dist_certain_3_true_ksigma = dist_certain_3[dist_certain_3['label'] == dist_certain_3['distksigma_class']]

    if use_recon:
        # 如果使用重构损失，就在前面的基础上再进行过滤
        # 使用最近的簇的重构损失
        dist_certain_2_recon_md = dist_certain_2[dist_certain_2['cla_md_recon_uncertain_2'] == 0]
        dist_certain_3_recon_md = dist_certain_3[dist_certain_3['cla_md_recon_uncertain_3'] == 0]
        dist_certain_2_true_dist_recon_md = dist_certain_2_recon_md[
            dist_certain_2_recon_md['label'] == dist_certain_2_recon_md['dist_class']]
        dist_certain_3_true_dist_recon_md = dist_certain_3_recon_md[
            dist_certain_3_recon_md['label'] == dist_certain_3_recon_md['dist_class']]
        dist_certain_2_true_ksigma_recon_md = dist_certain_2_recon_md[
            dist_certain_2_recon_md['label'] == dist_certain_2_recon_md['distksigma_class']]
        dist_certain_3_true_ksigma_recon_md = dist_certain_3_recon_md[
            dist_certain_3_recon_md['label'] == dist_certain_3_recon_md['distksigma_class']]

        # 使用全部的重构损失
        dist_certain_2_recon_all = dist_certain_2[dist_certain_2['all_recon_uncertain_2'] == 0]
        dist_certain_3_recon_all = dist_certain_3[dist_certain_3['all_recon_uncertain_3'] == 0]
        dist_certain_2_true_dist_recon_all = dist_certain_2_recon_all[
            dist_certain_2_recon_all['label'] == dist_certain_2_recon_all['dist_class']]
        dist_certain_3_true_dist_recon_all = dist_certain_3_recon_all[
            dist_certain_3_recon_all['label'] == dist_certain_3_recon_all['dist_class']]
        dist_certain_2_true_ksigma_recon_all = dist_certain_2_recon_all[
            dist_certain_2_recon_all['label'] == dist_certain_2_recon_all['distksigma_class']]
        dist_certain_3_true_ksigma_recon_all = dist_certain_3_recon_all[
            dist_certain_3_recon_all['label'] == dist_certain_3_recon_all['distksigma_class']]

    # 通过最小的ksigma判断确定性
    ksigma_certain_2 = detail_data[detail_data['ksigmadist_uncertain_2'] == 0]
    ksigma_certain_3 = detail_data[detail_data['ksigmadist_uncertain_3'] == 0]

    ksigma_certain_2_true_dist = ksigma_certain_2[ksigma_certain_2['label'] == ksigma_certain_2['dist_class']]
    ksigma_certain_2_true_ksigma = ksigma_certain_2[ksigma_certain_2['label'] == ksigma_certain_2['distksigma_class']]

    ksigma_certain_3_true_dist = ksigma_certain_3[ksigma_certain_3['label'] == ksigma_certain_3['dist_class']]
    ksigma_certain_3_true_ksigma = ksigma_certain_3[ksigma_certain_3['label'] == ksigma_certain_3['distksigma_class']]

    if use_recon:
        # 如果使用重构损失，就在前面的基础上再进行过滤
        # 使用最近的簇的重构损失
        ksigma_certain_2_recon_mkd = ksigma_certain_2[ksigma_certain_2['cla_mkd_recon_uncertain_2'] == 0]
        ksigma_certain_3_recon_mkd = ksigma_certain_3[ksigma_certain_3['cla_mkd_recon_uncertain_3'] == 0]
        ksigma_certain_2_true_dist_recon_mkd = ksigma_certain_2_recon_mkd[
            ksigma_certain_2_recon_mkd['label'] == ksigma_certain_2_recon_mkd['dist_class']]
        ksigma_certain_3_true_dist_recon_mkd = ksigma_certain_3_recon_mkd[
            ksigma_certain_3_recon_mkd['label'] == ksigma_certain_3_recon_mkd['dist_class']]
        ksigma_certain_2_true_ksigma_recon_mkd = ksigma_certain_2_recon_mkd[
            ksigma_certain_2_recon_mkd['label'] == ksigma_certain_2_recon_mkd['distksigma_class']]
        ksigma_certain_3_true_ksigma_recon_mkd = ksigma_certain_3_recon_mkd[
            ksigma_certain_3_recon_mkd['label'] == ksigma_certain_3_recon_mkd['distksigma_class']]

        # 使用全部的重构损失
        ksigma_certain_2_recon_all = ksigma_certain_2[ksigma_certain_2['all_recon_uncertain_2'] == 0]
        ksigma_certain_3_recon_all = ksigma_certain_3[ksigma_certain_3['all_recon_uncertain_3'] == 0]
        ksigma_certain_2_true_dist_recon_all = ksigma_certain_2_recon_all[
            ksigma_certain_2_recon_all['label'] == ksigma_certain_2_recon_all['dist_class']]
        ksigma_certain_3_true_dist_recon_all = ksigma_certain_3_recon_all[
            ksigma_certain_3_recon_all['label'] == ksigma_certain_3_recon_all['dist_class']]
        ksigma_certain_2_true_ksigma_recon_all = ksigma_certain_2_recon_all[
            ksigma_certain_2_recon_all['label'] == ksigma_certain_2_recon_all['distksigma_class']]
        ksigma_certain_3_true_ksigma_recon_all = ksigma_certain_3_recon_all[
            ksigma_certain_3_recon_all['label'] == ksigma_certain_3_recon_all['distksigma_class']]

    print(f'== Overall performance of {os.path.basename(file_path)} ==')
    print(' -- certain by dist')
    print(
        f'     2-sigma: certain per: {format(len(dist_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2)}')
    print(f'        dist acc: {len(dist_certain_2_true_dist) / (len(dist_certain_2) + 1e-9)}')
    print(f'        ksig acc: {len(dist_certain_2_true_ksigma) / (len(dist_certain_2) + 1e-9)}')
    print(
        f'     3-sigma: certain per: {format(len(dist_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3)}')
    print(f'        dist acc: {len(dist_certain_3_true_dist) / (len(dist_certain_3) + 1e-9)}')
    print(f'        ksig acc: {len(dist_certain_3_true_ksigma) / (len(dist_certain_3) + 1e-9)}')

    if use_recon:
        print(
            f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
        print(f'        dist acc: {len(dist_certain_2_true_dist_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
        print(f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
        print(
            f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
        print(f'        dist acc: {len(dist_certain_2_true_dist_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
        print(f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
        print(
            f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
        print(f'        dist acc: {len(dist_certain_3_true_dist_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
        print(f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
        print(
            f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
        print(f'        dist acc: {len(dist_certain_3_true_dist_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')
        print(f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

    print(' -- certain by ksigma')
    print(
        f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
    print(f'        dist acc: {len(ksigma_certain_2_true_dist) / (len(ksigma_certain_2) + 1e-9)}')
    print(f'        ksig acc: {len(ksigma_certain_2_true_ksigma) / (len(ksigma_certain_2) + 1e-9)}')
    print(
        f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
    print(f'        dist acc: {len(ksigma_certain_3_true_dist) / (len(ksigma_certain_3) + 1e-9)}')
    print(f'        ksig acc: {len(ksigma_certain_3_true_ksigma) / (len(ksigma_certain_3) + 1e-9)}')

    if use_recon:
        print(
            f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
        print(
            f'        dist acc: {len(ksigma_certain_2_true_dist_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
        print(
            f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
        print(
            f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
        print(
            f'        dist acc: {len(ksigma_certain_2_true_dist_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')
        print(
            f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

        print(
            f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
        print(
            f'        dist acc: {len(ksigma_certain_3_true_dist_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
        print(
            f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
        print(
            f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
        print(
            f'        dist acc: {len(ksigma_certain_3_true_dist_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')
        print(
            f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')

    if write_to_log:
        logging.info(f'== Overall performance of {os.path.basename(file_path)} ==')
        logging.info(' -- certain by dist')
        logging.info(
            f'     2-sigma: certain per: {format(len(dist_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2)}')
        logging.info(f'        dist acc: {len(dist_certain_2_true_dist) / (len(dist_certain_2) + 1e-9)}')
        logging.info(f'        ksig acc: {len(dist_certain_2_true_ksigma) / (len(dist_certain_2) + 1e-9)}')
        logging.info(
            f'     3-sigma: certain per: {format(len(dist_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3)}')
        logging.info(f'        dist acc: {len(dist_certain_3_true_dist) / (len(dist_certain_3) + 1e-9)}')
        logging.info(f'        ksig acc: {len(dist_certain_3_true_ksigma) / (len(dist_certain_3) + 1e-9)}')

        if use_recon:
            logging.info(
                f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
            logging.info(
                f'        dist acc: {len(dist_certain_2_true_dist_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
            logging.info(
                f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
            logging.info(
                f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
            logging.info(
                f'        dist acc: {len(dist_certain_2_true_dist_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
            logging.info(
                f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
            logging.info(
                f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
            logging.info(
                f'        dist acc: {len(dist_certain_3_true_dist_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
            logging.info(
                f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
            logging.info(
                f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
            logging.info(
                f'        dist acc: {len(dist_certain_3_true_dist_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')
            logging.info(
                f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

        logging.info(' -- certain by ksigma')
        logging.info(
            f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
        logging.info(f'        dist acc: {len(ksigma_certain_2_true_dist) / (len(ksigma_certain_2) + 1e-9)}')
        logging.info(f'        ksig acc: {len(ksigma_certain_2_true_ksigma) / (len(ksigma_certain_2) + 1e-9)}')
        logging.info(
            f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
        logging.info(f'        dist acc: {len(ksigma_certain_3_true_dist) / (len(ksigma_certain_3) + 1e-9)}')
        logging.info(f'        ksig acc: {len(ksigma_certain_3_true_ksigma) / (len(ksigma_certain_3) + 1e-9)}')

        if use_recon:
            logging.info(
                f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
            logging.info(
                f'        dist acc: {len(ksigma_certain_2_true_dist_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
            logging.info(
                f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
            logging.info(
                f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
            logging.info(
                f'        dist acc: {len(ksigma_certain_2_true_dist_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')
            logging.info(
                f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

            logging.info(
                f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
            logging.info(
                f'        dist acc: {len(ksigma_certain_3_true_dist_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
            logging.info(
                f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
            logging.info(
                f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
            logging.info(
                f'        dist acc: {len(ksigma_certain_3_true_dist_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')
            logging.info(
                f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')

    print(f'== Subclass performance of {os.path.basename(file_path)} ==')
    if write_to_log:
        logging.info(f'== Subclass performance of {os.path.basename(file_path)} ==')
    for i in detail_labels:
        print(f'== {i} ==')
        detail_data = data[data['detail_label'] == i]
        # 通过最小的距离判定确定性
        dist_certain_2 = detail_data[detail_data['dist_uncertain_2'] == 0]
        dist_certain_3 = detail_data[detail_data['dist_uncertain_3'] == 0]

        dist_certain_2_true_dist = dist_certain_2[dist_certain_2['label'] == dist_certain_2['dist_class']]
        dist_certain_2_true_ksigma = dist_certain_2[dist_certain_2['label'] == dist_certain_2['distksigma_class']]

        dist_certain_3_true_dist = dist_certain_3[dist_certain_3['label'] == dist_certain_3['dist_class']]
        dist_certain_3_true_ksigma = dist_certain_3[dist_certain_3['label'] == dist_certain_3['distksigma_class']]

        if use_recon:
            # 如果使用重构损失，就在前面的基础上再进行过滤
            # 使用最近的簇的重构损失
            dist_certain_2_recon_md = dist_certain_2[dist_certain_2['cla_md_recon_uncertain_2'] == 0]
            dist_certain_3_recon_md = dist_certain_3[dist_certain_3['cla_md_recon_uncertain_3'] == 0]
            dist_certain_2_true_dist_recon_md = dist_certain_2_recon_md[
                dist_certain_2_recon_md['label'] == dist_certain_2_recon_md['dist_class']]
            dist_certain_3_true_dist_recon_md = dist_certain_3_recon_md[
                dist_certain_3_recon_md['label'] == dist_certain_3_recon_md['dist_class']]
            dist_certain_2_true_ksigma_recon_md = dist_certain_2_recon_md[
                dist_certain_2_recon_md['label'] == dist_certain_2_recon_md['distksigma_class']]
            dist_certain_3_true_ksigma_recon_md = dist_certain_3_recon_md[
                dist_certain_3_recon_md['label'] == dist_certain_3_recon_md['distksigma_class']]

            # 使用全部的重构损失
            dist_certain_2_recon_all = dist_certain_2[dist_certain_2['all_recon_uncertain_2'] == 0]
            dist_certain_3_recon_all = dist_certain_3[dist_certain_3['all_recon_uncertain_3'] == 0]
            dist_certain_2_true_dist_recon_all = dist_certain_2_recon_all[
                dist_certain_2_recon_all['label'] == dist_certain_2_recon_all['dist_class']]
            dist_certain_3_true_dist_recon_all = dist_certain_3_recon_all[
                dist_certain_3_recon_all['label'] == dist_certain_3_recon_all['dist_class']]
            dist_certain_2_true_ksigma_recon_all = dist_certain_2_recon_all[
                dist_certain_2_recon_all['label'] == dist_certain_2_recon_all['distksigma_class']]
            dist_certain_3_true_ksigma_recon_all = dist_certain_3_recon_all[
                dist_certain_3_recon_all['label'] == dist_certain_3_recon_all['distksigma_class']]

        # 通过最小的ksigma判断确定性
        ksigma_certain_2 = detail_data[detail_data['ksigmadist_uncertain_2'] == 0]
        ksigma_certain_3 = detail_data[detail_data['ksigmadist_uncertain_3'] == 0]

        ksigma_certain_2_true_dist = ksigma_certain_2[ksigma_certain_2['label'] == ksigma_certain_2['dist_class']]
        ksigma_certain_2_true_ksigma = ksigma_certain_2[
            ksigma_certain_2['label'] == ksigma_certain_2['distksigma_class']]

        ksigma_certain_3_true_dist = ksigma_certain_3[ksigma_certain_3['label'] == ksigma_certain_3['dist_class']]
        ksigma_certain_3_true_ksigma = ksigma_certain_3[
            ksigma_certain_3['label'] == ksigma_certain_3['distksigma_class']]

        if use_recon:
            # 如果使用重构损失，就在前面的基础上再进行过滤
            # 使用最近的簇的重构损失
            ksigma_certain_2_recon_mkd = ksigma_certain_2[ksigma_certain_2['cla_mkd_recon_uncertain_2'] == 0]
            ksigma_certain_3_recon_mkd = ksigma_certain_3[ksigma_certain_3['cla_mkd_recon_uncertain_3'] == 0]
            ksigma_certain_2_true_dist_recon_mkd = ksigma_certain_2_recon_mkd[
                ksigma_certain_2_recon_mkd['label'] == ksigma_certain_2_recon_mkd['dist_class']]
            ksigma_certain_3_true_dist_recon_mkd = ksigma_certain_3_recon_mkd[
                ksigma_certain_3_recon_mkd['label'] == ksigma_certain_3_recon_mkd['dist_class']]
            ksigma_certain_2_true_ksigma_recon_mkd = ksigma_certain_2_recon_mkd[
                ksigma_certain_2_recon_mkd['label'] == ksigma_certain_2_recon_mkd['distksigma_class']]
            ksigma_certain_3_true_ksigma_recon_mkd = ksigma_certain_3_recon_mkd[
                ksigma_certain_3_recon_mkd['label'] == ksigma_certain_3_recon_mkd['distksigma_class']]

            # 使用全部的重构损失
            ksigma_certain_2_recon_all = ksigma_certain_2[ksigma_certain_2['all_recon_uncertain_2'] == 0]
            ksigma_certain_3_recon_all = ksigma_certain_3[ksigma_certain_3['all_recon_uncertain_3'] == 0]
            ksigma_certain_2_true_dist_recon_all = ksigma_certain_2_recon_all[
                ksigma_certain_2_recon_all['label'] == ksigma_certain_2_recon_all['dist_class']]
            ksigma_certain_3_true_dist_recon_all = ksigma_certain_3_recon_all[
                ksigma_certain_3_recon_all['label'] == ksigma_certain_3_recon_all['dist_class']]
            ksigma_certain_2_true_ksigma_recon_all = ksigma_certain_2_recon_all[
                ksigma_certain_2_recon_all['label'] == ksigma_certain_2_recon_all['distksigma_class']]
            ksigma_certain_3_true_ksigma_recon_all = ksigma_certain_3_recon_all[
                ksigma_certain_3_recon_all['label'] == ksigma_certain_3_recon_all['distksigma_class']]

        print(f'== {i} ==')
        print(' -- certain by dist')
        print(
            f'     2-sigma: certain per: {format(len(dist_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2)}')
        print(f'        dist acc: {len(dist_certain_2_true_dist) / (len(dist_certain_2) + 1e-9)}')
        print(f'        ksig acc: {len(dist_certain_2_true_ksigma) / (len(dist_certain_2) + 1e-9)}')
        print(
            f'     3-sigma: certain per: {format(len(dist_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3)}')
        print(f'        dist acc: {len(dist_certain_3_true_dist) / (len(dist_certain_3) + 1e-9)}')
        print(f'        ksig acc: {len(dist_certain_3_true_ksigma) / (len(dist_certain_3) + 1e-9)}')

        if use_recon:
            print(
                f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
            print(f'        dist acc: {len(dist_certain_2_true_dist_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
            print(
                f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
            print(
                f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
            print(
                f'        dist acc: {len(dist_certain_2_true_dist_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
            print(
                f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
            print(
                f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
            print(f'        dist acc: {len(dist_certain_3_true_dist_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
            print(
                f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
            print(
                f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
            print(
                f'        dist acc: {len(dist_certain_3_true_dist_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')
            print(
                f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

        print(' -- certain by ksigma')
        print(
            f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
        print(f'        dist acc: {len(ksigma_certain_2_true_dist) / (len(ksigma_certain_2) + 1e-9)}')
        print(f'        ksig acc: {len(ksigma_certain_2_true_ksigma) / (len(ksigma_certain_2) + 1e-9)}')
        print(
            f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
        print(f'        dist acc: {len(ksigma_certain_3_true_dist) / (len(ksigma_certain_3) + 1e-9)}')
        print(f'        ksig acc: {len(ksigma_certain_3_true_ksigma) / (len(ksigma_certain_3) + 1e-9)}')

        if use_recon:
            print(
                f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
            print(
                f'        dist acc: {len(ksigma_certain_2_true_dist_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
            print(
                f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
            print(
                f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
            print(
                f'        dist acc: {len(ksigma_certain_2_true_dist_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')
            print(
                f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

            print(
                f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
            print(
                f'        dist acc: {len(ksigma_certain_3_true_dist_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
            print(
                f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
            print(
                f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
            print(
                f'        dist acc: {len(ksigma_certain_3_true_dist_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')
            print(
                f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')

        if write_to_log:
            logging.info(f'== {i} ==')
            logging.info(' -- certain by dist')
            logging.info(
                f'     2-sigma: certain per: {format(len(dist_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2)}')
            logging.info(f'        dist acc: {len(dist_certain_2_true_dist) / (len(dist_certain_2) + 1e-9)}')
            logging.info(f'        ksig acc: {len(dist_certain_2_true_ksigma) / (len(dist_certain_2) + 1e-9)}')
            logging.info(
                f'     3-sigma: certain per: {format(len(dist_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3)}')
            logging.info(f'        dist acc: {len(dist_certain_3_true_dist) / (len(dist_certain_3) + 1e-9)}')
            logging.info(f'        ksig acc: {len(dist_certain_3_true_ksigma) / (len(dist_certain_3) + 1e-9)}')

            if use_recon:
                logging.info(
                    f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
                logging.info(
                    f'        dist acc: {len(dist_certain_2_true_dist_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
                logging.info(
                    f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
                logging.info(
                    f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
                logging.info(
                    f'        dist acc: {len(dist_certain_2_true_dist_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
                logging.info(
                    f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
                logging.info(
                    f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
                logging.info(
                    f'        dist acc: {len(dist_certain_3_true_dist_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
                logging.info(
                    f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
                logging.info(
                    f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
                logging.info(
                    f'        dist acc: {len(dist_certain_3_true_dist_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')
                logging.info(
                    f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

            logging.info(' -- certain by ksigma')
            logging.info(
                f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
            logging.info(f'        dist acc: {len(ksigma_certain_2_true_dist) / (len(ksigma_certain_2) + 1e-9)}')
            logging.info(f'        ksig acc: {len(ksigma_certain_2_true_ksigma) / (len(ksigma_certain_2) + 1e-9)}')
            logging.info(
                f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
            logging.info(f'        dist acc: {len(ksigma_certain_3_true_dist) / (len(ksigma_certain_3) + 1e-9)}')
            logging.info(f'        ksig acc: {len(ksigma_certain_3_true_ksigma) / (len(ksigma_certain_3) + 1e-9)}')

            if use_recon:
                logging.info(
                    f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
                logging.info(
                    f'        dist acc: {len(ksigma_certain_2_true_dist_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
                logging.info(
                    f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
                logging.info(
                    f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
                logging.info(
                    f'        dist acc: {len(ksigma_certain_2_true_dist_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')
                logging.info(
                    f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

                logging.info(
                    f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
                logging.info(
                    f'        dist acc: {len(ksigma_certain_3_true_dist_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
                logging.info(
                    f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
                logging.info(
                    f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
                logging.info(
                    f'        dist acc: {len(ksigma_certain_3_true_dist_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')
                logging.info(
                    f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')


def process_single_ocassion_distco(data, all_data_num, write_to_log, use_recon):
    data['dist_class'] = data['seq_dist_class']
    detail_labels = np.unique(data['detail_label'])

    detail_data = data

    # 通过最小的距离判定确定性
    dist_certain_2 = detail_data[detail_data['dist_uncertain_2'] == 0]
    dist_certain_3 = detail_data[detail_data['dist_uncertain_3'] == 0]

    dist_certain_2_true_dist = dist_certain_2[dist_certain_2['label'] == dist_certain_2['dist_class']]
    dist_certain_3_true_dist = dist_certain_3[dist_certain_3['label'] == dist_certain_3['dist_class']]
    if use_recon:
        # 如果使用重构损失，就在前面的基础上再进行过滤
        # 使用最近的簇的重构损失
        dist_certain_2_recon_md = dist_certain_2[dist_certain_2['cla_md_recon_uncertain_2'] == 0]
        dist_certain_3_recon_md = dist_certain_3[dist_certain_3['cla_md_recon_uncertain_3'] == 0]
        dist_certain_2_true_dist_recon_md = dist_certain_2_recon_md[
            dist_certain_2_recon_md['label'] == dist_certain_2_recon_md['dist_class']]
        dist_certain_3_true_dist_recon_md = dist_certain_3_recon_md[
            dist_certain_3_recon_md['label'] == dist_certain_3_recon_md['dist_class']]

        # 使用全部的重构损失
        dist_certain_2_recon_all = dist_certain_2[dist_certain_2['all_recon_uncertain_2'] == 0]
        dist_certain_3_recon_all = dist_certain_3[dist_certain_3['all_recon_uncertain_3'] == 0]
        dist_certain_2_true_dist_recon_all = dist_certain_2_recon_all[
            dist_certain_2_recon_all['label'] == dist_certain_2_recon_all['dist_class']]
        dist_certain_3_true_dist_recon_all = dist_certain_3_recon_all[
            dist_certain_3_recon_all['label'] == dist_certain_3_recon_all['dist_class']]

    # 通过最小的ksigma判断确定性
    ksigma_certain_2 = detail_data[detail_data['ksigmadist_uncertain_2'] == 0]
    ksigma_certain_3 = detail_data[detail_data['ksigmadist_uncertain_3'] == 0]
    ksigma_certain_2_true_dist = ksigma_certain_2[ksigma_certain_2['label'] == ksigma_certain_2['dist_class']]
    ksigma_certain_3_true_dist = ksigma_certain_3[ksigma_certain_3['label'] == ksigma_certain_3['dist_class']]

    if use_recon:
        # 如果使用重构损失，就在前面的基础上再进行过滤
        # 使用最近的簇的重构损失
        ksigma_certain_2_recon_mkd = ksigma_certain_2[ksigma_certain_2['cla_mkd_recon_uncertain_2'] == 0]
        ksigma_certain_3_recon_mkd = ksigma_certain_3[ksigma_certain_3['cla_mkd_recon_uncertain_3'] == 0]
        ksigma_certain_2_true_dist_recon_mkd = ksigma_certain_2_recon_mkd[
            ksigma_certain_2_recon_mkd['label'] == ksigma_certain_2_recon_mkd['dist_class']]
        ksigma_certain_3_true_dist_recon_mkd = ksigma_certain_3_recon_mkd[
            ksigma_certain_3_recon_mkd['label'] == ksigma_certain_3_recon_mkd['dist_class']]

        # 使用全部的重构损失
        ksigma_certain_2_recon_all = ksigma_certain_2[ksigma_certain_2['all_recon_uncertain_2'] == 0]
        ksigma_certain_3_recon_all = ksigma_certain_3[ksigma_certain_3['all_recon_uncertain_3'] == 0]
        ksigma_certain_2_true_dist_recon_all = ksigma_certain_2_recon_all[
            ksigma_certain_2_recon_all['label'] == ksigma_certain_2_recon_all['dist_class']]
        ksigma_certain_3_true_dist_recon_all = ksigma_certain_3_recon_all[
            ksigma_certain_3_recon_all['label'] == ksigma_certain_3_recon_all['dist_class']]

    print('== Overall performance ==')
    print(' -- certain by dist')
    print(
        f'     2-sigma: certain per: {format(len(dist_certain_2) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2)}')
    print(f'        dist acc: {len(dist_certain_2_true_dist) / (len(dist_certain_2) + 1e-9)}')
    print(
        f'     3-sigma: certain per: {format(len(dist_certain_3) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3)}')
    print(f'        dist acc: {len(dist_certain_3_true_dist) / (len(dist_certain_3) + 1e-9)}')

    if use_recon:
        print(
            f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
        print(f'        dist acc: {len(dist_certain_2_true_dist_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
        print(
            f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
        print(f'        dist acc: {len(dist_certain_2_true_dist_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
        print(
            f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
        print(f'        dist acc: {len(dist_certain_3_true_dist_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
        print(
            f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
        print(f'        dist acc: {len(dist_certain_3_true_dist_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

    print(' -- certain by ksigma')
    print(
        f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
    print(f'        dist acc: {len(ksigma_certain_2_true_dist) / (len(ksigma_certain_2) + 1e-9)}')
    print(
        f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
    print(f'        dist acc: {len(ksigma_certain_3_true_dist) / (len(ksigma_certain_3) + 1e-9)}')

    if use_recon:
        print(
            f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
        print(
            f'        dist acc: {len(ksigma_certain_2_true_dist_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
        print(
            f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
        print(
            f'        dist acc: {len(ksigma_certain_2_true_dist_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

        print(
            f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
        print(
            f'        dist acc: {len(ksigma_certain_3_true_dist_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
        print(
            f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
        print(
            f'        dist acc: {len(ksigma_certain_3_true_dist_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')

    if write_to_log:
        logging.info(f'== Overall performance ==')
        logging.info(' -- certain by dist')
        logging.info(
            f'     2-sigma: certain per: {format(len(dist_certain_2) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2)}')
        logging.info(f'        dist acc: {len(dist_certain_2_true_dist) / (len(dist_certain_2) + 1e-9)}')
        logging.info(
            f'     3-sigma: certain per: {format(len(dist_certain_3) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3)}')
        logging.info(f'        dist acc: {len(dist_certain_3_true_dist) / (len(dist_certain_3) + 1e-9)}')

        if use_recon:
            logging.info(
                f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
            logging.info(
                f'        dist acc: {len(dist_certain_2_true_dist_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
            logging.info(
                f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
            logging.info(
                f'        dist acc: {len(dist_certain_2_true_dist_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
            logging.info(
                f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
            logging.info(
                f'        dist acc: {len(dist_certain_3_true_dist_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
            logging.info(
                f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
            logging.info(
                f'        dist acc: {len(dist_certain_3_true_dist_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

        logging.info(' -- certain by ksigma')
        logging.info(
            f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
        logging.info(f'        dist acc: {len(ksigma_certain_2_true_dist) / (len(ksigma_certain_2) + 1e-9)}')
        logging.info(
            f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
        logging.info(f'        dist acc: {len(ksigma_certain_3_true_dist) / (len(ksigma_certain_3) + 1e-9)}')

        if use_recon:
            logging.info(
                f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
            logging.info(
                f'        dist acc: {len(ksigma_certain_2_true_dist_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
            logging.info(
                f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
            logging.info(
                f'        dist acc: {len(ksigma_certain_2_true_dist_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

            logging.info(
                f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
            logging.info(
                f'        dist acc: {len(ksigma_certain_3_true_dist_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
            logging.info(
                f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
            logging.info(
                f'        dist acc: {len(ksigma_certain_3_true_dist_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')

    print(f'== Subclass performance ==')
    if write_to_log:
        logging.info(f'== Subclass performance ==')
    for i in detail_labels:
        print(f'== {i} ==')
        detail_data = data[data['detail_label'] == i]
        # 通过最小的距离判定确定性
        dist_certain_2 = detail_data[detail_data['dist_uncertain_2'] == 0]
        dist_certain_3 = detail_data[detail_data['dist_uncertain_3'] == 0]
        dist_certain_2_true_dist = dist_certain_2[dist_certain_2['label'] == dist_certain_2['dist_class']]
        dist_certain_3_true_dist = dist_certain_3[dist_certain_3['label'] == dist_certain_3['dist_class']]

        if use_recon:
            # 如果使用重构损失，就在前面的基础上再进行过滤
            # 使用最近的簇的重构损失
            dist_certain_2_recon_md = dist_certain_2[dist_certain_2['cla_md_recon_uncertain_2'] == 0]
            dist_certain_3_recon_md = dist_certain_3[dist_certain_3['cla_md_recon_uncertain_3'] == 0]
            dist_certain_2_true_dist_recon_md = dist_certain_2_recon_md[
                dist_certain_2_recon_md['label'] == dist_certain_2_recon_md['dist_class']]
            dist_certain_3_true_dist_recon_md = dist_certain_3_recon_md[
                dist_certain_3_recon_md['label'] == dist_certain_3_recon_md['dist_class']]

            # 使用全部的重构损失
            dist_certain_2_recon_all = dist_certain_2[dist_certain_2['all_recon_uncertain_2'] == 0]
            dist_certain_3_recon_all = dist_certain_3[dist_certain_3['all_recon_uncertain_3'] == 0]
            dist_certain_2_true_dist_recon_all = dist_certain_2_recon_all[
                dist_certain_2_recon_all['label'] == dist_certain_2_recon_all['dist_class']]
            dist_certain_3_true_dist_recon_all = dist_certain_3_recon_all[
                dist_certain_3_recon_all['label'] == dist_certain_3_recon_all['dist_class']]

        # 通过最小的ksigma判断确定性
        ksigma_certain_2 = detail_data[detail_data['ksigmadist_uncertain_2'] == 0]
        ksigma_certain_3 = detail_data[detail_data['ksigmadist_uncertain_3'] == 0]
        ksigma_certain_2_true_dist = ksigma_certain_2[ksigma_certain_2['label'] == ksigma_certain_2['dist_class']]
        ksigma_certain_3_true_dist = ksigma_certain_3[ksigma_certain_3['label'] == ksigma_certain_3['dist_class']]

        if use_recon:
            # 如果使用重构损失，就在前面的基础上再进行过滤
            # 使用最近的簇的重构损失
            ksigma_certain_2_recon_mkd = ksigma_certain_2[ksigma_certain_2['cla_mkd_recon_uncertain_2'] == 0]
            ksigma_certain_3_recon_mkd = ksigma_certain_3[ksigma_certain_3['cla_mkd_recon_uncertain_3'] == 0]
            ksigma_certain_2_true_dist_recon_mkd = ksigma_certain_2_recon_mkd[
                ksigma_certain_2_recon_mkd['label'] == ksigma_certain_2_recon_mkd['dist_class']]
            ksigma_certain_3_true_dist_recon_mkd = ksigma_certain_3_recon_mkd[
                ksigma_certain_3_recon_mkd['label'] == ksigma_certain_3_recon_mkd['dist_class']]

            # 使用全部的重构损失
            ksigma_certain_2_recon_all = ksigma_certain_2[ksigma_certain_2['all_recon_uncertain_2'] == 0]
            ksigma_certain_3_recon_all = ksigma_certain_3[ksigma_certain_3['all_recon_uncertain_3'] == 0]
            ksigma_certain_2_true_dist_recon_all = ksigma_certain_2_recon_all[
                ksigma_certain_2_recon_all['label'] == ksigma_certain_2_recon_all['dist_class']]
            ksigma_certain_3_true_dist_recon_all = ksigma_certain_3_recon_all[
                ksigma_certain_3_recon_all['label'] == ksigma_certain_3_recon_all['dist_class']]

        print(f'== {i} ==')
        print(' -- certain by dist')
        print(
            f'     2-sigma: certain per: {format(len(dist_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2)}')
        print(f'        dist acc: {len(dist_certain_2_true_dist) / (len(dist_certain_2) + 1e-9)}')
        print(
            f'     3-sigma: certain per: {format(len(dist_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3)}')
        print(f'        dist acc: {len(dist_certain_3_true_dist) / (len(dist_certain_3) + 1e-9)}')

        if use_recon:
            print(
                f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
            print(f'        dist acc: {len(dist_certain_2_true_dist_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
            print(
                f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
            print(
                f'        dist acc: {len(dist_certain_2_true_dist_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
            print(
                f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
            print(f'        dist acc: {len(dist_certain_3_true_dist_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
            print(
                f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
            print(
                f'        dist acc: {len(dist_certain_3_true_dist_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

        print(' -- certain by ksigma')
        print(
            f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
        print(f'        dist acc: {len(ksigma_certain_2_true_dist) / (len(ksigma_certain_2) + 1e-9)}')
        print(
            f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
        print(f'        dist acc: {len(ksigma_certain_3_true_dist) / (len(ksigma_certain_3) + 1e-9)}')

        if use_recon:
            print(
                f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
            print(
                f'        dist acc: {len(ksigma_certain_2_true_dist_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
            print(
                f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
            print(
                f'        dist acc: {len(ksigma_certain_2_true_dist_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

            print(
                f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
            print(
                f'        dist acc: {len(ksigma_certain_3_true_dist_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
            print(
                f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
            print(
                f'        dist acc: {len(ksigma_certain_3_true_dist_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')

        if write_to_log:
            logging.info(f'== {i} ==')
            logging.info(' -- certain by dist')
            logging.info(
                f'     2-sigma: certain per: {format(len(dist_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2)}')
            logging.info(f'        dist acc: {len(dist_certain_2_true_dist) / (len(dist_certain_2) + 1e-9)}')
            logging.info(
                f'     3-sigma: certain per: {format(len(dist_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3)}')
            logging.info(f'        dist acc: {len(dist_certain_3_true_dist) / (len(dist_certain_3) + 1e-9)}')

            if use_recon:
                logging.info(
                    f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
                logging.info(
                    f'        dist acc: {len(dist_certain_2_true_dist_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
                logging.info(
                    f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
                logging.info(
                    f'        dist acc: {len(dist_certain_2_true_dist_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
                logging.info(
                    f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
                logging.info(
                    f'        dist acc: {len(dist_certain_3_true_dist_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
                logging.info(
                    f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
                logging.info(
                    f'        dist acc: {len(dist_certain_3_true_dist_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

            logging.info(' -- certain by ksigma')
            logging.info(
                f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
            logging.info(f'        dist acc: {len(ksigma_certain_2_true_dist) / (len(ksigma_certain_2) + 1e-9)}')
            logging.info(
                f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
            logging.info(f'        dist acc: {len(ksigma_certain_3_true_dist) / (len(ksigma_certain_3) + 1e-9)}')

            if use_recon:
                logging.info(
                    f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
                logging.info(
                    f'        dist acc: {len(ksigma_certain_2_true_dist_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
                logging.info(
                    f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
                logging.info(
                    f'        dist acc: {len(ksigma_certain_2_true_dist_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

                logging.info(
                    f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
                logging.info(
                    f'        dist acc: {len(ksigma_certain_3_true_dist_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
                logging.info(
                    f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
                logging.info(
                    f'        dist acc: {len(ksigma_certain_3_true_dist_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')


def process_single_ocassion_kdistco(data, all_data_num, write_to_log, use_recon):
    data['distksigma_class'] = data['seq_distksigma_class']
    detail_labels = np.unique(data['detail_label'])
    detail_data = data

    # 通过最小的距离判定确定性
    dist_certain_2 = detail_data[detail_data['dist_uncertain_2'] == 0]
    dist_certain_3 = detail_data[detail_data['dist_uncertain_3'] == 0]
    dist_certain_2_true_ksigma = dist_certain_2[dist_certain_2['label'] == dist_certain_2['distksigma_class']]
    dist_certain_3_true_ksigma = dist_certain_3[dist_certain_3['label'] == dist_certain_3['distksigma_class']]

    if use_recon:
        # 如果使用重构损失，就在前面的基础上再进行过滤
        # 使用最近的簇的重构损失
        dist_certain_2_recon_md = dist_certain_2[dist_certain_2['cla_md_recon_uncertain_2'] == 0]
        dist_certain_3_recon_md = dist_certain_3[dist_certain_3['cla_md_recon_uncertain_3'] == 0]
        dist_certain_2_true_ksigma_recon_md = dist_certain_2_recon_md[
            dist_certain_2_recon_md['label'] == dist_certain_2_recon_md['distksigma_class']]
        dist_certain_3_true_ksigma_recon_md = dist_certain_3_recon_md[
            dist_certain_3_recon_md['label'] == dist_certain_3_recon_md['distksigma_class']]

        # 使用全部的重构损失
        dist_certain_2_recon_all = dist_certain_2[dist_certain_2['all_recon_uncertain_2'] == 0]
        dist_certain_3_recon_all = dist_certain_3[dist_certain_3['all_recon_uncertain_3'] == 0]
        dist_certain_2_true_ksigma_recon_all = dist_certain_2_recon_all[
            dist_certain_2_recon_all['label'] == dist_certain_2_recon_all['distksigma_class']]
        dist_certain_3_true_ksigma_recon_all = dist_certain_3_recon_all[
            dist_certain_3_recon_all['label'] == dist_certain_3_recon_all['distksigma_class']]

    # 通过最小的ksigma判断确定性
    ksigma_certain_2 = detail_data[detail_data['ksigmadist_uncertain_2'] == 0]
    ksigma_certain_3 = detail_data[detail_data['ksigmadist_uncertain_3'] == 0]
    ksigma_certain_2_true_ksigma = ksigma_certain_2[ksigma_certain_2['label'] == ksigma_certain_2['distksigma_class']]
    ksigma_certain_3_true_ksigma = ksigma_certain_3[ksigma_certain_3['label'] == ksigma_certain_3['distksigma_class']]

    if use_recon:
        # 如果使用重构损失，就在前面的基础上再进行过滤
        # 使用最近的簇的重构损失
        ksigma_certain_2_recon_mkd = ksigma_certain_2[ksigma_certain_2['cla_mkd_recon_uncertain_2'] == 0]
        ksigma_certain_3_recon_mkd = ksigma_certain_3[ksigma_certain_3['cla_mkd_recon_uncertain_3'] == 0]
        ksigma_certain_2_true_ksigma_recon_mkd = ksigma_certain_2_recon_mkd[
            ksigma_certain_2_recon_mkd['label'] == ksigma_certain_2_recon_mkd['distksigma_class']]
        ksigma_certain_3_true_ksigma_recon_mkd = ksigma_certain_3_recon_mkd[
            ksigma_certain_3_recon_mkd['label'] == ksigma_certain_3_recon_mkd['distksigma_class']]

        # 使用全部的重构损失
        ksigma_certain_2_recon_all = ksigma_certain_2[ksigma_certain_2['all_recon_uncertain_2'] == 0]
        ksigma_certain_3_recon_all = ksigma_certain_3[ksigma_certain_3['all_recon_uncertain_3'] == 0]
        ksigma_certain_2_true_ksigma_recon_all = ksigma_certain_2_recon_all[
            ksigma_certain_2_recon_all['label'] == ksigma_certain_2_recon_all['distksigma_class']]
        ksigma_certain_3_true_ksigma_recon_all = ksigma_certain_3_recon_all[
            ksigma_certain_3_recon_all['label'] == ksigma_certain_3_recon_all['distksigma_class']]

    print('== Overall performance ==')
    print(' -- certain by dist')
    print(
        f'     2-sigma: certain per: {format(len(dist_certain_2) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2)}')
    print(f'        ksig acc: {len(dist_certain_2_true_ksigma) / (len(dist_certain_2) + 1e-9)}')
    print(
        f'     3-sigma: certain per: {format(len(dist_certain_3) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3)}')
    print(f'        ksig acc: {len(dist_certain_3_true_ksigma) / (len(dist_certain_3) + 1e-9)}')

    if use_recon:
        print(
            f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
        print(f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
        print(
            f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
        print(f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
        print(
            f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
        print(f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
        print(
            f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
        print(f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

    print(' -- certain by ksigma')
    print(
        f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
    print(f'        ksig acc: {len(ksigma_certain_2_true_ksigma) / (len(ksigma_certain_2) + 1e-9)}')
    print(
        f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
    print(f'        ksig acc: {len(ksigma_certain_3_true_ksigma) / (len(ksigma_certain_3) + 1e-9)}')

    if use_recon:
        print(
            f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
        print(
            f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
        print(
            f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
        print(
            f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

        print(
            f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
        print(
            f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
        print(
            f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
        print(
            f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')

    if write_to_log:
        logging.info(f'== Overall performance ==')
        logging.info(' -- certain by dist')
        logging.info(
            f'     2-sigma: certain per: {format(len(dist_certain_2) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2)}')
        logging.info(f'        ksig acc: {len(dist_certain_2_true_ksigma) / (len(dist_certain_2) + 1e-9)}')
        logging.info(
            f'     3-sigma: certain per: {format(len(dist_certain_3) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3)}')
        logging.info(f'        ksig acc: {len(dist_certain_3_true_ksigma) / (len(dist_certain_3) + 1e-9)}')

        if use_recon:
            logging.info(
                f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
            logging.info(
                f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
            logging.info(
                f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
            logging.info(
                f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
            logging.info(
                f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
            logging.info(
                f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
            logging.info(
                f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
            logging.info(
                f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

        logging.info(' -- certain by ksigma')
        logging.info(
            f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
        logging.info(f'        ksig acc: {len(ksigma_certain_2_true_ksigma) / (len(ksigma_certain_2) + 1e-9)}')
        logging.info(
            f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
        logging.info(f'        ksig acc: {len(ksigma_certain_3_true_ksigma) / (len(ksigma_certain_3) + 1e-9)}')

        if use_recon:
            logging.info(
                f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
            logging.info(
                f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
            logging.info(
                f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
            logging.info(
                f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

            logging.info(
                f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
            logging.info(
                f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
            logging.info(
                f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (all_data_num + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
            logging.info(
                f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')

    print(f'== Subclass performance ==')
    if write_to_log:
        logging.info(f'== Subclass performance ==')
    for i in detail_labels:
        print(f'== {i} ==')
        detail_data = data[data['detail_label'] == i]
        # 通过最小的距离判定确定性
        dist_certain_2 = detail_data[detail_data['dist_uncertain_2'] == 0]
        dist_certain_3 = detail_data[detail_data['dist_uncertain_3'] == 0]
        dist_certain_2_true_ksigma = dist_certain_2[dist_certain_2['label'] == dist_certain_2['distksigma_class']]
        dist_certain_3_true_ksigma = dist_certain_3[dist_certain_3['label'] == dist_certain_3['distksigma_class']]

        if use_recon:
            # 如果使用重构损失，就在前面的基础上再进行过滤
            # 使用最近的簇的重构损失
            dist_certain_2_recon_md = dist_certain_2[dist_certain_2['cla_md_recon_uncertain_2'] == 0]
            dist_certain_3_recon_md = dist_certain_3[dist_certain_3['cla_md_recon_uncertain_3'] == 0]
            dist_certain_2_true_ksigma_recon_md = dist_certain_2_recon_md[
                dist_certain_2_recon_md['label'] == dist_certain_2_recon_md['distksigma_class']]
            dist_certain_3_true_ksigma_recon_md = dist_certain_3_recon_md[
                dist_certain_3_recon_md['label'] == dist_certain_3_recon_md['distksigma_class']]

            # 使用全部的重构损失
            dist_certain_2_recon_all = dist_certain_2[dist_certain_2['all_recon_uncertain_2'] == 0]
            dist_certain_3_recon_all = dist_certain_3[dist_certain_3['all_recon_uncertain_3'] == 0]
            dist_certain_2_true_ksigma_recon_all = dist_certain_2_recon_all[
                dist_certain_2_recon_all['label'] == dist_certain_2_recon_all['distksigma_class']]
            dist_certain_3_true_ksigma_recon_all = dist_certain_3_recon_all[
                dist_certain_3_recon_all['label'] == dist_certain_3_recon_all['distksigma_class']]

        # 通过最小的ksigma判断确定性
        ksigma_certain_2 = detail_data[detail_data['ksigmadist_uncertain_2'] == 0]
        ksigma_certain_3 = detail_data[detail_data['ksigmadist_uncertain_3'] == 0]
        ksigma_certain_2_true_ksigma = ksigma_certain_2[
            ksigma_certain_2['label'] == ksigma_certain_2['distksigma_class']]
        ksigma_certain_3_true_ksigma = ksigma_certain_3[
            ksigma_certain_3['label'] == ksigma_certain_3['distksigma_class']]

        if use_recon:
            # 如果使用重构损失，就在前面的基础上再进行过滤
            # 使用最近的簇的重构损失
            ksigma_certain_2_recon_mkd = ksigma_certain_2[ksigma_certain_2['cla_mkd_recon_uncertain_2'] == 0]
            ksigma_certain_3_recon_mkd = ksigma_certain_3[ksigma_certain_3['cla_mkd_recon_uncertain_3'] == 0]
            ksigma_certain_2_true_ksigma_recon_mkd = ksigma_certain_2_recon_mkd[
                ksigma_certain_2_recon_mkd['label'] == ksigma_certain_2_recon_mkd['distksigma_class']]
            ksigma_certain_3_true_ksigma_recon_mkd = ksigma_certain_3_recon_mkd[
                ksigma_certain_3_recon_mkd['label'] == ksigma_certain_3_recon_mkd['distksigma_class']]

            # 使用全部的重构损失
            ksigma_certain_2_recon_all = ksigma_certain_2[ksigma_certain_2['all_recon_uncertain_2'] == 0]
            ksigma_certain_3_recon_all = ksigma_certain_3[ksigma_certain_3['all_recon_uncertain_3'] == 0]
            ksigma_certain_2_true_ksigma_recon_all = ksigma_certain_2_recon_all[
                ksigma_certain_2_recon_all['label'] == ksigma_certain_2_recon_all['distksigma_class']]
            ksigma_certain_3_true_ksigma_recon_all = ksigma_certain_3_recon_all[
                ksigma_certain_3_recon_all['label'] == ksigma_certain_3_recon_all['distksigma_class']]

        print(f'== {i} ==')
        print(' -- certain by dist')
        print(
            f'     2-sigma: certain per: {format(len(dist_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2)}')
        print(f'        ksig acc: {len(dist_certain_2_true_ksigma) / (len(dist_certain_2) + 1e-9)}')
        print(
            f'     3-sigma: certain per: {format(len(dist_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3)}')
        print(f'        ksig acc: {len(dist_certain_3_true_ksigma) / (len(dist_certain_3) + 1e-9)}')

        if use_recon:
            print(
                f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
            print(
                f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
            print(
                f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
            print(
                f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
            print(
                f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
            print(
                f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
            print(
                f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
            print(
                f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

        print(' -- certain by ksigma')
        print(
            f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
        print(f'        ksig acc: {len(ksigma_certain_2_true_ksigma) / (len(ksigma_certain_2) + 1e-9)}')
        print(
            f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
        print(f'        ksig acc: {len(ksigma_certain_3_true_ksigma) / (len(ksigma_certain_3) + 1e-9)}')

        if use_recon:
            print(
                f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
            print(
                f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
            print(
                f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
            print(
                f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

            print(
                f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
            print(
                f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
            print(
                f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
            print(
                f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')

        if write_to_log:
            logging.info(f'== {i} ==')
            logging.info(' -- certain by dist')
            logging.info(
                f'     2-sigma: certain per: {format(len(dist_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2)}')
            logging.info(f'        ksig acc: {len(dist_certain_2_true_ksigma) / (len(dist_certain_2) + 1e-9)}')
            logging.info(
                f'     3-sigma: certain per: {format(len(dist_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3)}')
            logging.info(f'        ksig acc: {len(dist_certain_3_true_ksigma) / (len(dist_certain_3) + 1e-9)}')

            if use_recon:
                logging.info(
                    f'     2-sigma & recon md: certain per: {format(len(dist_certain_2_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_md)}')
                logging.info(
                    f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_md) / (len(dist_certain_2_recon_md) + 1e-9)}')
                logging.info(
                    f'     2-sigma & recon all: certain per: {format(len(dist_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_2_recon_all)}')
                logging.info(
                    f'        ksig acc: {len(dist_certain_2_true_ksigma_recon_all) / (len(dist_certain_2_recon_all) + 1e-9)}')
                logging.info(
                    f'     3-sigma & recon md: certain per: {format(len(dist_certain_3_recon_md) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_md)}')
                logging.info(
                    f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_md) / (len(dist_certain_3_recon_md) + 1e-9)}')
                logging.info(
                    f'     3-sigma & recon all: certain per: {format(len(dist_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(dist_certain_3_recon_all)}')
                logging.info(
                    f'        ksig acc: {len(dist_certain_3_true_ksigma_recon_all) / (len(dist_certain_3_recon_all) + 1e-9)}')

            logging.info(' -- certain by ksigma')
            logging.info(
                f'     2-sigma: certain per: {format(len(ksigma_certain_2) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2)}')
            logging.info(f'        ksig acc: {len(ksigma_certain_2_true_ksigma) / (len(ksigma_certain_2) + 1e-9)}')
            logging.info(
                f'     3-sigma: certain per: {format(len(ksigma_certain_3) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3)}')
            logging.info(f'        ksig acc: {len(ksigma_certain_3_true_ksigma) / (len(ksigma_certain_3) + 1e-9)}')

            if use_recon:
                logging.info(
                    f'     2-sigma & recon md: certain per: {format(len(ksigma_certain_2_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_mkd)}')
                logging.info(
                    f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_mkd) / (len(ksigma_certain_2_recon_mkd) + 1e-9)}')
                logging.info(
                    f'     2-sigma & recon all: certain per: {format(len(ksigma_certain_2_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_2_recon_all)}')
                logging.info(
                    f'        ksig acc: {len(ksigma_certain_2_true_ksigma_recon_all) / (len(ksigma_certain_2_recon_all) + 1e-9)}')

                logging.info(
                    f'     3-sigma & recon md: certain per: {format(len(ksigma_certain_3_recon_mkd) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_mkd)}')
                logging.info(
                    f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_mkd) / (len(ksigma_certain_3_recon_mkd) + 1e-9)}')
                logging.info(
                    f'     3-sigma & recon all: certain per: {format(len(ksigma_certain_3_recon_all) / (len(detail_data) + 1e-9), ".5f")}, {len(ksigma_certain_3_recon_all)}')
                logging.info(
                    f'        ksig acc: {len(ksigma_certain_3_true_ksigma_recon_all) / (len(ksigma_certain_3_recon_all) + 1e-9)}')


def get_dist_co_certain(row):
    if row['seq_dist_class'] == row['sta_dist_class']:
        return 1
    else:
        return 0


def get_kdist_co_certain(row):
    if row['seq_distksigma_class'] == row['sta_distksigma_class']:
        return 1
    else:
        return 0


def eval_new_data_seqsta_singlefile(seq_path, sta_path):
    seq_data = pd.read_csv(seq_path)
    sta_data = pd.read_csv(sta_path)

    # 通过最小距离判断确定性
    columns = ['label', 'detail_label', 'min_dist_col', 'min_ksigmadist_col', 'min_dist', 'min_dist_ksigma',
               'min_dist_corr_distksigma', 'min_dist_recon_k', 'min_ksigmadist_recon_k', 'dist_class',
               'distksigma_class',
               'dist_uncertain_2', 'dist_uncertain_3', 'ksigmadist_uncertain_2', 'ksigmadist_uncertain_3',
               'cla_md_recon_uncertain_2', 'cla_md_recon_uncertain_3', 'cla_mkd_recon_uncertain_2',
               'cla_mkd_recon_uncertain_3', 'all_recon_uncertain_2', 'all_recon_uncertain_3']
    seq_data = seq_data[columns]
    sta_data = sta_data[columns]

    new_seq_col = ['seq_' + col for col in seq_data.columns]
    new_seq_col[0] = 'label'  # rename一直失败，用这样的方法调整列名
    new_seq_col[1] = 'detail_label'
    new_sta_col = ['sta_' + col for col in sta_data.columns]

    seq_data.columns = new_seq_col
    sta_data.columns = new_sta_col

    new_data = pd.concat([seq_data.reset_index(drop=True), sta_data.reset_index(drop=True)], axis=1)

    # 使用sta和seq得到的类别结果是否一致
    new_data['dist_co_certain'] = new_data.apply(get_dist_co_certain, axis=1)
    new_data['kdist_co_certain'] = new_data.apply(get_kdist_co_certain, axis=1)

    # 之前把or都写成and了，但是and的结果更好
    new_data['dist_uncertain_2'] = new_data.apply(lambda x: x['seq_dist_uncertain_2'] or x['sta_dist_uncertain_2'],
                                                  axis=1)
    new_data['dist_uncertain_3'] = new_data.apply(lambda x: x['seq_dist_uncertain_3'] or x['sta_dist_uncertain_3'],
                                                  axis=1)

    new_data['ksigmadist_uncertain_2'] = new_data.apply(
        lambda x: x['seq_ksigmadist_uncertain_2'] or x['sta_ksigmadist_uncertain_2'], axis=1)
    new_data['ksigmadist_uncertain_3'] = new_data.apply(
        lambda x: x['seq_ksigmadist_uncertain_3'] or x['sta_ksigmadist_uncertain_3'], axis=1)

    new_data['cla_md_recon_uncertain_2'] = new_data.apply(
        lambda x: x['seq_cla_md_recon_uncertain_2'] or x['sta_cla_md_recon_uncertain_2'], axis=1)
    new_data['cla_md_recon_uncertain_3'] = new_data.apply(
        lambda x: x['seq_cla_md_recon_uncertain_3'] or x['sta_cla_md_recon_uncertain_3'], axis=1)
    new_data['cla_mkd_recon_uncertain_2'] = new_data.apply(
        lambda x: x['seq_cla_mkd_recon_uncertain_2'] or x['sta_cla_mkd_recon_uncertain_2'], axis=1)
    new_data['cla_mkd_recon_uncertain_3'] = new_data.apply(
        lambda x: x['seq_cla_mkd_recon_uncertain_3'] or x['sta_cla_mkd_recon_uncertain_3'], axis=1)

    new_data['all_recon_uncertain_2'] = new_data.apply(
        lambda x: x['seq_all_recon_uncertain_2'] or x['sta_all_recon_uncertain_2'], axis=1)
    new_data['all_recon_uncertain_3'] = new_data.apply(
        lambda x: x['seq_all_recon_uncertain_3'] or x['sta_all_recon_uncertain_3'], axis=1)

    dist_co_certain = new_data[new_data['dist_co_certain'] == 1]
    kdist_co_certain = new_data[new_data['kdist_co_certain'] == 1]

    all_data_num = len(new_data)

    print('################# Performance of dist co certain ################# ')
    logging.info('################# Performance of dist co certain ################# ')
    process_single_ocassion_distco(dist_co_certain, all_data_num, True, True)
    print('################# Performance of ksigma dist co certain ################# ')
    logging.info('################# Performance of ksigma dist co certain ################# ')
    process_single_ocassion_kdistco(kdist_co_certain, all_data_num, True, True)


def eval_test_95_data(res_root):
    print('==============================================')
    print('[INFO] Evaluate test data, only with seq feature, using 95% data.')
    logging.info('==============================================')
    logging.info('[INFO] Evaluate test data, only with seq feature, using 95% data.')
    normal_deep_new_path_seq = os.path.join(res_root, 'normal_test_result_ksigma_95deep_new_seq.csv')
    attack_deep_new_path_seq = os.path.join(res_root, 'attack_test_result_ksigma_95deep_new_seq.csv')
    eval_new_data_new(normal_deep_new_path_seq, write_to_log=True, use_recon=True)
    eval_new_data_new(attack_deep_new_path_seq, write_to_log=True, use_recon=True)

    print('==============================================')
    print('[INFO] Evaluate test data, only with sta feature, using 95% data.')
    logging.info('==============================================')
    logging.info('[INFO] Evaluate test data, only with sta feature, using 95% data.')
    normal_deep_new_path_sta = os.path.join(res_root, 'normal_test_result_ksigma_95deep_new_sta.csv')
    attack_deep_new_path_sta = os.path.join(res_root, 'attack_test_result_ksigma_95deep_new_sta.csv')
    eval_new_data_new(normal_deep_new_path_sta, write_to_log=True, use_recon=True)
    eval_new_data_new(attack_deep_new_path_sta, write_to_log=True, use_recon=True)

    print('==============================================')
    print('[INFO] Evaluate test data, with sta and seq feature, using 95% data.')
    logging.info('==============================================')
    logging.info('[INFO] Evaluate test data, with sta and seq feature, using 95% data.')
    print('****** Normal ******')
    logging.info('****** Normal ******')
    eval_new_data_seqsta_singlefile(normal_deep_new_path_seq, normal_deep_new_path_sta)
    print('****** Attack ******')
    logging.info('****** Attack ******')
    eval_new_data_seqsta_singlefile(attack_deep_new_path_seq, attack_deep_new_path_sta)


def get_sta_info_of_train(save_data_root, re_generate_sta_info=True):
    """
    使用所有数据，计算每个簇的簇心、类内样本到簇心的距离、重构损失，并将距离和重构损失拟合成正态分布，将参数存储起来
    """

    if (not os.path.exists(os.path.join(save_data_root, 'train_sta_info.pickle'))) or re_generate_sta_info:
        if not os.path.exists(os.path.join(save_data_root, 'train_sta_info.pickle')):
            print('[Attention] train_sta_info.pickle does not exists, regenerate.')
        elif re_generate_sta_info:
            print('[Attention] train_sta_info.pickle is required to regenerate.')

        new_data = pd.read_pickle(os.path.join(save_data_root, 'train_mid_info.pickle'))

        label_unique = np.unique(new_data['label'])
        cluster_unique = np.unique(new_data['cluster'])

        print(f'label unique: {label_unique}')
        print(f'cluster unique: {cluster_unique}')

        # 保存每一个细化的类的中心、类内样本到中心的距离、重构损失
        center_dict = {}
        distance_dict = {}
        reconloss_dict = {}

        for l in label_unique:
            for c in cluster_unique:
                tmp_data = new_data[(new_data['label'] == l) & (new_data['cluster'] == c)]
                print(f'({l}, {c}): {len(tmp_data)}')
                if len(tmp_data) == 0:
                    continue
                features = np.stack(tmp_data['feature'].values)
                mean_feature = np.mean(features, axis=0)
                center_dict[(l, c)] = mean_feature

                distances = np.linalg.norm(features - mean_feature, axis=1)
                distance_dict[(l, c)] = distances

                reconl = tmp_data['recon_loss'].values
                reconloss_dict[(l, c)] = reconl

        train_sta_info = {'center_dict': center_dict, 'distance_dict': distance_dict, 'reconloss_dict': reconloss_dict}
        with open(os.path.join(save_data_root, 'train_sta_info.pickle'), 'wb') as f:
            pickle.dump(train_sta_info, f)

    # 将距离、重构损失等等，拟合到分布上
    print('[INFO] Fit distance to a distribution.')

    with open(os.path.join(save_data_root, 'train_sta_info.pickle'), 'rb') as f:
        train_sta_info = pickle.load(f)

    # 保存拟合后的分布的参数信息
    train_distri_para_info = {}

    distance_dict = train_sta_info['distance_dict']
    # 将正态分布的参数存储进去,每个类的样本到其类心的距离
    dist_norm_para_dict = {}
    print('fit distances')
    for key in distance_dict.keys():
        print(key)
        cur_dis = distance_dict[key]
        if len(cur_dis) < 1:
            continue
        paras, errors = fit_norm(cur_dis)
        dist_norm_para_dict[key] = paras
        print(errors)

    train_distri_para_info['dist_distri'] = dist_norm_para_dict

    print('fit recon errors')
    reconloss_dict = train_sta_info['reconloss_dict']
    relos_norm_para_dict = {}
    all_loss = []
    for key in reconloss_dict.keys():
        print(key)
        cur_loss = reconloss_dict[key].astype(np.float64)
        if len(cur_loss) < 1:
            continue
        paras, errors = fit_norm(cur_loss)
        relos_norm_para_dict[key] = paras
        print(errors)
        all_loss.extend(cur_loss)
    # 也拟合一个所有数据的重构损失
    paras, errors = fit_norm(all_loss)
    relos_norm_para_dict[(-1, -1)] = paras
    train_distri_para_info['reclos_distri'] = relos_norm_para_dict

    with open(os.path.join(save_data_root, 'train_distri_para_info.pickle'), 'wb') as f:
        pickle.dump(train_distri_para_info, f)


def get_sta_info_of_train_stadata(save_data_root, re_generate_sta_info=True):
    # 获取流量的统计特征的统计信息，基于get_sta_info_of_train_95_stadata实现
    if (
            not os.path.exists(
                os.path.join(save_data_root, 'train_sta_info_stadata.pickle'))) or re_generate_sta_info:
        if not os.path.exists(os.path.join(save_data_root, 'train_sta_info_stadata.pickle')):
            print('[Attention] train_sta_info.pickle does not exists, regenerate.')
        elif re_generate_sta_info:
            print('[Attention] train_sta_info.pickle is required to regenerate.')

        new_data = pd.read_pickle(os.path.join(save_data_root, 'train_stamid_info.pickle'))
        label_unique = np.unique(new_data['label'])
        cluster_unique = np.unique(new_data['cluster'])

        print(f'label unique: {label_unique}')
        print(f'cluster unique: {cluster_unique}')

        # 保存每一个细化的类的中心、类内样本到中心的距离、重构损失
        center_dict = {}
        distance_dict = {}
        reconloss_dict = {}

        for l in label_unique:
            for c in cluster_unique:
                tmp_data = new_data[(new_data['label'] == l) & (new_data['cluster'] == c)]
                print(f'({l}, {c}): {len(tmp_data)}')
                if len(tmp_data) == 0:
                    continue
                features = np.stack(tmp_data['feature'].values)
                mean_feature = np.mean(features, axis=0)
                center_dict[(l, c)] = mean_feature

                distances = np.linalg.norm(features - mean_feature, axis=1)
                distance_dict[(l, c)] = distances

                reconl = tmp_data['recon_loss'].values
                reconloss_dict[(l, c)] = reconl

        train_sta_info = {'center_dict': center_dict, 'distance_dict': distance_dict, 'reconloss_dict': reconloss_dict}
        with open(os.path.join(save_data_root, 'train_sta_info_stadata.pickle'), 'wb') as f:
            pickle.dump(train_sta_info, f)

    print('[INFO] Fit distance to a distribution use all data (sta).')
    with open(os.path.join(save_data_root, 'train_sta_info_stadata.pickle'), 'rb') as f:
        train_sta_info = pickle.load(f)

    # 保存拟合后的分布的参数信息
    train_distri_para_info = {}

    distance_dict = train_sta_info['distance_dict']
    # 将正态分布的参数存储进去,每个类的样本到其类心的距离
    dist_norm_para_dict = {}
    print('fit distances')
    for key in distance_dict.keys():
        print(key)
        cur_dis = distance_dict[key]
        if len(cur_dis) < 1:
            continue
        paras, errors = fit_norm(cur_dis)
        dist_norm_para_dict[key] = paras
        print(errors)

    train_distri_para_info['dist_distri'] = dist_norm_para_dict

    print('fit recon errors')
    reconloss_dict = train_sta_info['reconloss_dict']
    relos_norm_para_dict = {}
    all_loss = []
    for key in reconloss_dict.keys():
        print(key)
        cur_loss = reconloss_dict[key].astype(np.float64)
        if len(cur_loss) < 1:
            continue
        paras, errors = fit_norm(cur_loss)
        relos_norm_para_dict[key] = paras
        print(errors)
        all_loss.extend(cur_loss)
    # 也拟合一个所有数据的重构损失
    paras, errors = fit_norm(all_loss)
    relos_norm_para_dict[(-1, -1)] = paras
    train_distri_para_info['reclos_distri'] = relos_norm_para_dict

    with open(os.path.join(save_data_root, 'train_distri_para_info_stadata.pickle'), 'wb') as f:
        pickle.dump(train_distri_para_info, f)


def get_test_result_ksigma_seq(train_save_data_root, test_save_data_root, res_root):
    with open(os.path.join(train_save_data_root, 'train_distri_para_info.pickle'), 'rb') as f:
        train_distri_para_info = pickle.load(f)
    with open(os.path.join(test_save_data_root, 'test_recloss_distancetocenter_seq.pickle'), 'rb') as f:
        new_test_data = pickle.load(f)

    dist_distri = train_distri_para_info['dist_distri']
    for key, value in dist_distri.items():
        ori_row = 'dist_' + str(key)
        dst_row = 'ksigma_dist_' + str(key)
        v1, v2 = get_norm_par(value)
        new_test_data[dst_row] = new_test_data[ori_row].apply(lambda x: get_ksigma(x, v1, v2))

    reclos_distri = train_distri_para_info['reclos_distri']
    for key, value in reclos_distri.items():
        ori_row = 'recon_loss'
        dst_row = 'ksigma_reclos_' + str(key)
        v1, v2 = get_norm_par(value)
        new_test_data[dst_row] = new_test_data[ori_row].apply(lambda x: get_ksigma(x, v1, v2))

    final_res_data = new_test_data.drop(['feature'], axis=1)

    final_res_data.to_csv(os.path.join(res_root, 'test_result_ksigma_seq.csv'), index=False)


def get_test_result_ksigma_sta(train_save_data_root, test_save_data_root, res_root):
    with open(os.path.join(train_save_data_root, 'train_distri_para_info_stadata.pickle'), 'rb') as f:
        train_distri_para_info = pickle.load(f)
    with open(os.path.join(test_save_data_root, 'test_recloss_distancetocenter_sta.pickle'), 'rb') as f:
        new_test_data = pickle.load(f)

    dist_distri = train_distri_para_info['dist_distri']
    for key, value in dist_distri.items():
        ori_row = 'dist_' + str(key)
        dst_row = 'ksigma_dist_' + str(key)
        v1, v2 = get_norm_par(value)
        new_test_data[dst_row] = new_test_data[ori_row].apply(lambda x: get_ksigma(x, v1, v2))

    reclos_distri = train_distri_para_info['reclos_distri']
    for key, value in reclos_distri.items():
        ori_row = 'recon_loss'
        dst_row = 'ksigma_reclos_' + str(key)
        v1, v2 = get_norm_par(value)
        new_test_data[dst_row] = new_test_data[ori_row].apply(lambda x: get_ksigma(x, v1, v2))

    final_res_data = new_test_data.drop(['feature'], axis=1)

    final_res_data.to_csv(os.path.join(res_root, 'test_result_ksigma_sta.csv'), index=False)


def generate_eval_pre_data_seq(res_root):
    """ 使用95%的数据，生成最终决策前的中间结果，例如是不是certain这些，把myutil中的 eval_test_result拆开了，评估另外放一个函数"""
    test_result_path = os.path.join(res_root, 'test_result_ksigma_seq.csv')
    normal_path = os.path.join(res_root, 'normal_test_result_ksigma_seq.csv')
    attack_path = os.path.join(res_root, 'attack_test_result_ksigma_seq.csv')
    normal_deep_new_path = os.path.join(res_root, 'normal_test_result_ksigma_deep_new_seq.csv')
    attack_deep_new_path = os.path.join(res_root, 'attack_test_result_ksigma_deep_new_seq.csv')
    split_normal_attack(test_result_path, normal_path, attack_path)
    generate_new_data_new(normal_path, normal_deep_new_path)
    generate_new_data_new(attack_path, attack_deep_new_path)


def generate_eval_pre_data_sta(res_root):
    """ 使用95%的数据，生成最终决策前的中间结果，例如是不是certain这些，把myutil中的 eval_test_result拆开了，评估另外放一个函数"""
    test_result_path = os.path.join(res_root, 'test_result_ksigma_sta.csv')
    normal_path = os.path.join(res_root, 'normal_test_result_ksigma_sta.csv')
    attack_path = os.path.join(res_root, 'attack_test_result_ksigma_sta.csv')
    normal_deep_new_path = os.path.join(res_root, 'normal_test_result_ksigma_deep_new_sta.csv')
    attack_deep_new_path = os.path.join(res_root, 'attack_test_result_ksigma_deep_new_sta.csv')
    split_normal_attack(test_result_path, normal_path, attack_path)
    generate_new_data_new(normal_path, normal_deep_new_path)
    generate_new_data_new(attack_path, attack_deep_new_path)


def eval_test_data(res_root):
    print('==============================================')
    print('[INFO] Evaluate test data, only with seq feature, using all data.')
    logging.info('==============================================')
    logging.info('[INFO] Evaluate test data, only with seq feature, using all data.')
    normal_deep_new_path_seq = os.path.join(res_root, 'normal_test_result_ksigma_deep_new_seq.csv')
    attack_deep_new_path_seq = os.path.join(res_root, 'attack_test_result_ksigma_deep_new_seq.csv')
    eval_new_data_new(normal_deep_new_path_seq, write_to_log=True, use_recon=True)
    eval_new_data_new(attack_deep_new_path_seq, write_to_log=True, use_recon=True)

    print('==============================================')
    print('[INFO] Evaluate test data, only with sta feature, using all data.')
    logging.info('==============================================')
    logging.info('[INFO] Evaluate test data, only with sta feature, using all data.')
    normal_deep_new_path_sta = os.path.join(res_root, 'normal_test_result_ksigma_deep_new_sta.csv')
    attack_deep_new_path_sta = os.path.join(res_root, 'attack_test_result_ksigma_deep_new_sta.csv')
    eval_new_data_new(normal_deep_new_path_sta, write_to_log=True, use_recon=True)
    eval_new_data_new(attack_deep_new_path_sta, write_to_log=True, use_recon=True)

    print('==============================================')
    print('[INFO] Evaluate test data, with sta and seq feature, using all data.')
    logging.info('==============================================')
    logging.info('[INFO] Evaluate test data, with sta and seq feature, using all data.')
    print('****** Normal ******')
    logging.info('****** Normal ******')
    eval_new_data_seqsta_singlefile(normal_deep_new_path_seq, normal_deep_new_path_sta)
    print('****** Attack ******')
    logging.info('****** Attack ******')
    eval_new_data_seqsta_singlefile(attack_deep_new_path_seq, attack_deep_new_path_sta)


def eval_model_stage1(eval_part, myconfig, train_loader, test_loader, model_root, res_root, train_save_data_root,
                      test_save_data_root, used_seq_model, used_sta_model, recalculate_train_related=True):
    # recalculate_train_related 是否重新计算训练相关的
    print('+++++++++++++++++++++++++++++++++++++++++++++')
    print(f'++++++++++++++++ {eval_part} ++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++')
    logging.info('+++++++++++++++++++++++++++++++++++++++++++++')
    logging.info(f'++++++++++++++++ {eval_part} ++++++++++++++++')
    logging.info('+++++++++++++++++++++++++++++++++++++++++++++')
    """评估第一轮的模型"""

    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward,
                                   dropout=myconfig.dropout).to(myconfig.device)
    seq_model.load_state_dict(torch.load(os.path.join(model_root, used_seq_model)))
    seq_model.eval()
    logging.info('use seq model: ' + used_seq_model)

    sta_model = MyModelStaAE().to(myconfig.device)
    sta_model.load_state_dict(torch.load(os.path.join(model_root, used_sta_model)))
    sta_model.eval()
    logging.info('use sta model: ' + used_sta_model)

    if recalculate_train_related:
        # ===================================================
        # 2. 加载新模型，计算训练样本的重构损失，以及隐藏层特征
        # ===================================================
        print('[INFO] Get recon loss and distance to center of train dataset.')
        get_seqmid_info_of_train_transformer_sta(seq_model, train_loader, myconfig, train_save_data_root)
        print('[INFO] Save mid seq info of train data.')
        get_stamid_info_of_train_ae_sta(sta_model, train_loader, myconfig, train_save_data_root)
        print('[INFO] Save mid sta info of train data.')

        # 2024.8.19 将适用95%的相关的代码注释掉了，节省时间
        # # # ===================================================
        # # # 3. 加载距离,重构损失等序列,根据细化的类分类保存,并将它们拟合到分布上,并保存分布的参数，并验证分布的相似性
        # # #    拟合的时候使用95%的数据做拟合，（与normalize_adamwcos不同）
        # # # ===================================================
        # print('[INFO] fit distribution of distance to center and recon loss of 95% train data.')
        # get_sta_info_of_train_95(train_save_data_root)
        # print('[INFO] Save sta info of 95% train data (seq).')
        # get_sta_info_of_train_95_stadata(train_save_data_root)
        # print('[INFO] Save sta info of 95% train data (sta).')

    if recalculate_train_related:
        # 2024.10.30 train_sta_info.pickle 在评估测试样本的过程中也会用到，放到后面会导致出现这个文件不存在的情况，所以放到前面了
        # # # ===================================================
        # # # 11. 使用所有的数据去拟合分布
        # # # ===================================================
        print('[INFO] Get sta info of train data, including cluster center, distance to center, reconloss.')
        get_sta_info_of_train(train_save_data_root)
        print('[INFO] Save sta info of all train data (seq).')
        get_sta_info_of_train_stadata(train_save_data_root)
        print('[INFO] Save sta info of all train data (sta).')

    # # # # ===================================================
    # # # # 4. 加载新模型，计算测试样本的重构损失和隐藏层特征
    # # # # ===================================================
    # 计算训练样本和测试样本的隐含特征用的是相同的模型，所以这里旧不用重新加载模型
    get_seqmid_info_of_test_transformer_sta(seq_model, test_loader, myconfig, test_save_data_root)
    print('[INFO] Save seq mid info of test data. (seq)')
    get_stamid_info_of_test_ae_sta(sta_model, test_loader, myconfig, test_save_data_root)
    print('[INFO] Save sta mid info of test data. (sta)')

    # # # # ===================================================
    # # # # 5. 计算测试样本到每个簇的中心的距离
    # # # # ===================================================
    print('[INFO] Get the distance to cluster center of test samples. (seq)')
    get_distance_to_center_test_seq(train_save_data_root, test_save_data_root)
    print('[INFO] Get the distance to cluster center of test samples. (sta)')
    get_distance_to_center_test_sta(train_save_data_root, test_save_data_root)

    # # # ===================================================
    # # # 9. 计算样本属于每个类别的概率
    # # # ===================================================
    # print('[INFO] Get k sigma of test samples using 95% train data. (seq)')
    # get_test_result_ksigma_95_seq(train_save_data_root, test_save_data_root, res_root)
    # print('[INFO] Get k sigma of test samples using 95% train data. (sta)')
    # get_test_result_ksigma_95_sta(train_save_data_root, test_save_data_root, res_root)

    # # # ===================================================
    # # # 10. 使用距离评估测试数据 (使用95%的数据)
    # # # ===================================================
    # print('[INFO] Evaluate the final test results... by n sigma using 95% train data.')
    # print('[INFO] Generate pre-decision data. (seq)')
    # generate_eval_pre_95_data_seq(res_root)
    # print('[INFO] Generate pre-decision data. (sta)')
    # generate_eval_pre_95_data_sta(res_root)
    # print('[INFO] Evaluate data.')  # 单独使用一种特征，以及两种特征同时使用的代码都包含在这里
    # eval_test_95_data(res_root)

    # # # =======================================================
    # # # 12. 计算样本属于每个类别的概率，使用全部数据
    # # （不需要再重复计算测试样本的隐藏特征以及重构损失，已经计算过；簇心统一使用的是所有样本的簇心，没有区分95%）
    # # # =======================================================
    print('[INFO] Get k sigma of test samples using all train data. (seq)')
    get_test_result_ksigma_seq(train_save_data_root, test_save_data_root, res_root)
    print('[INFO] Get k sigma of test samples using all train data. (sta)')
    get_test_result_ksigma_sta(train_save_data_root, test_save_data_root, res_root)

    # # # ===================================================
    # # # 13. 使用距离评估测试数据 (使用全部数据)
    # # # ===================================================
    print('[INFO] Evaluate the final test results... by n sigma using all train data.')
    print('[INFO] Generate pre-decision data. (seq)')
    generate_eval_pre_data_seq(res_root)
    print('[INFO] Generate pre-decision data. (sta)')
    generate_eval_pre_data_sta(res_root)
    print('[INFO] Evaluate data.')  # 单独使用一种特征，以及两种特征同时使用的代码都包含在这里
    eval_test_data(res_root)


# 生成综合考虑seq和sta的文件
def generate_co_certain_file(seq_path, sta_path, dst_path):
    # print('--------- regenerate cocertain')

    seq_data = pd.read_csv(seq_path)
    sta_data = pd.read_csv(sta_path)

    # 通过最小距离判断确定性
    columns = ['label', 'detail_label', 'min_dist_col', 'min_ksigmadist_col', 'min_dist', 'min_dist_ksigma',
               'min_dist_corr_distksigma', 'min_dist_recon_k', 'min_ksigmadist_recon_k', 'dist_class',
               'distksigma_class',
               'dist_uncertain_2', 'dist_uncertain_3', 'ksigmadist_uncertain_2', 'ksigmadist_uncertain_3',
               'cla_md_recon_uncertain_2', 'cla_md_recon_uncertain_3', 'cla_mkd_recon_uncertain_2',
               'cla_mkd_recon_uncertain_3', 'all_recon_uncertain_2', 'all_recon_uncertain_3']
    seq_data = seq_data[columns]
    sta_data = sta_data[columns]

    new_seq_col = ['seq_' + col for col in seq_data.columns]
    new_seq_col[0] = 'label'  # rename一直失败，用这样的方法调整列名
    new_seq_col[1] = 'detail_label'
    new_sta_col = ['sta_' + col for col in sta_data.columns]

    seq_data.columns = new_seq_col
    sta_data.columns = new_sta_col

    new_data = pd.concat([seq_data.reset_index(drop=True), sta_data.reset_index(drop=True)], axis=1)

    # print(new_data.columns)

    # 使用sta和seq得到的类别结果是否一致
    new_data['dist_co_certain'] = new_data.apply(get_dist_co_certain, axis=1)
    new_data['kdist_co_certain'] = new_data.apply(get_kdist_co_certain, axis=1)

    # 之前把or都写成and了，但是and的结果更好
    new_data['dist_uncertain_2'] = new_data.apply(lambda x: x['seq_dist_uncertain_2'] or x['sta_dist_uncertain_2'],
                                                  axis=1)
    new_data['dist_uncertain_3'] = new_data.apply(lambda x: x['seq_dist_uncertain_3'] or x['sta_dist_uncertain_3'],
                                                  axis=1)

    new_data['ksigmadist_uncertain_2'] = new_data.apply(
        lambda x: x['seq_ksigmadist_uncertain_2'] or x['sta_ksigmadist_uncertain_2'], axis=1)
    new_data['ksigmadist_uncertain_3'] = new_data.apply(
        lambda x: x['seq_ksigmadist_uncertain_3'] or x['sta_ksigmadist_uncertain_3'], axis=1)

    new_data['cla_md_recon_uncertain_2'] = new_data.apply(
        lambda x: x['seq_cla_md_recon_uncertain_2'] or x['sta_cla_md_recon_uncertain_2'], axis=1)
    new_data['cla_md_recon_uncertain_3'] = new_data.apply(
        lambda x: x['seq_cla_md_recon_uncertain_3'] or x['sta_cla_md_recon_uncertain_3'], axis=1)
    new_data['cla_mkd_recon_uncertain_2'] = new_data.apply(
        lambda x: x['seq_cla_mkd_recon_uncertain_2'] or x['sta_cla_mkd_recon_uncertain_2'], axis=1)
    new_data['cla_mkd_recon_uncertain_3'] = new_data.apply(
        lambda x: x['seq_cla_mkd_recon_uncertain_3'] or x['sta_cla_mkd_recon_uncertain_3'], axis=1)

    new_data['all_recon_uncertain_2'] = new_data.apply(
        lambda x: x['seq_all_recon_uncertain_2'] or x['sta_all_recon_uncertain_2'], axis=1)
    new_data['all_recon_uncertain_3'] = new_data.apply(
        lambda x: x['seq_all_recon_uncertain_3'] or x['sta_all_recon_uncertain_3'], axis=1)

    # 8.21新增，之前没有写dist和recon共同决定的结果，这里补充上
    new_data['seq_combine_uncertain_3'] = new_data.apply(
        lambda x: x['seq_cla_md_recon_uncertain_3'] or x['seq_dist_uncertain_3'], axis=1)
    new_data['sta_combine_uncertain_3'] = new_data.apply(
        lambda x: x['sta_cla_md_recon_uncertain_3'] or x['sta_dist_uncertain_3'], axis=1)
    new_data['combine_uncertain_3'] = new_data.apply(
        lambda x: x['seq_combine_uncertain_3'] or x['sta_combine_uncertain_3'], axis=1)

    new_data.to_csv(dst_path, index=False)


def concat_ori_files(ori_file_root, files):
    ori_data = []
    for file in files:
        data = pd.read_pickle(os.path.join(ori_file_root, file))
        ori_data.append(data)
    ori_data = pd.concat(ori_data, axis=0)
    return ori_data


def generate_uncertain_data(regenerate_cocertain, consider_unmatch, cocertain_row, used_uncertain_row, seq_path,
                            sta_path, cocertain_path, classtype, files, ori_root, distseq_path,
                            diststa_path, latent_seq_path, latent_sta_path, dst_save_res_root, certain_data_path,
                            uncertain_data_path):
    """ certain data和uncertain data都存储起来了，而且包含了各种要使用的特征 """
    print(f'[INFO] Get uncertain of {classtype}')
    logging.info(f'[INFO] Get uncertain of {classtype}')
    if regenerate_cocertain or (not os.path.exists(cocertain_path)):
        generate_co_certain_file(seq_path, sta_path, cocertain_path)
        print('[INFO] Regenerate new cocertain file.')
    else:
        print('[INFO] Use existing cocertain file.')

    # 获得置信度文件
    cocertain_data = pd.read_csv(cocertain_path)
    # print('cocertain columns: ')
    # print(cocertain_data.columns)
    # 获得原始特征文件；将包含原始特征的多个文件合并成一个文件
    ori_feature_data = concat_ori_files(ori_root, files)
    # 获取隐含特征
    latent_seq = pd.read_pickle(latent_seq_path)[['feature', 'recon_loss']]
    latent_seq.columns = ['seqla_feature', 'seqla_recon']
    latent_sta = pd.read_pickle(latent_sta_path)[['feature', 'recon_loss']]
    latent_sta.columns = ['stala_feature', 'stala_recon']
    # 获得到各个簇的距离
    distseq = pd.read_csv(distseq_path)
    diststa = pd.read_csv(diststa_path)
    seq_columns = [i for i in distseq.columns if i.startswith('dist_(')]
    sta_columns = [i for i in diststa.columns if i.startswith('dist_(')]
    distseq = distseq[seq_columns]
    diststa = diststa[sta_columns]
    distseq.columns = ['seq_' + i for i in seq_columns]
    diststa.columns = ['sta_' + i for i in sta_columns]
    try:
        assert len(cocertain_data) == len(ori_feature_data)
        assert len(distseq) == len(diststa)
        assert len(cocertain_data) == len(distseq)
        assert len(latent_sta) == len(latent_seq)
        assert len(latent_sta) == len(ori_feature_data)
    except:
        print('[ERROR] lengths of cocertain data and ori feature data do not match.')
        print(f'        cocertain_data: {len(cocertain_data)}')
        print(f'        ori_feature_data: {len(ori_feature_data)}')
        print(f'        distseq: {len(distseq)}')
        print(f'        diststa: {len(diststa)}')
        print(f'        latent_sta: {len(latent_sta)}')
        print(f'        latent_seq: {len(latent_seq)}')

        logging.info('[ERROR] lengths of cocertain data and ori feature data do not match.')
        logging.info(f'        cocertain_data: {len(cocertain_data)}')
        logging.info(f'        ori_feature_data: {len(ori_feature_data)}')
        logging.info(f'        distseq: {len(distseq)}')
        logging.info(f'        diststa: {len(diststa)}')
        logging.info(f'        latent_sta: {len(latent_sta)}')
        logging.info(f'        latent_seq: {len(latent_seq)}')

    new_data = pd.concat([ori_feature_data.reset_index(drop=True), cocertain_data.reset_index(drop=True),
                          distseq.reset_index(drop=True), diststa.reset_index(drop=True),
                          latent_seq.reset_index(drop=True), latent_sta.reset_index(drop=True)], axis=1)

    if not consider_unmatch:
        uncertain_data = new_data[new_data[cocertain_row] == 1]
        uncertain_data = uncertain_data[uncertain_data[used_uncertain_row] == 1]
        # 将路径的生成挪到了外面，其他函数也可以一起用，但是由于是否考虑unmatch会影响数据的生成方式，所以依旧传递了config
        uncertain_data.to_csv(uncertain_data_path, index=False)
        certain_data = new_data[~new_data.index.isin(uncertain_data.index)]
        certain_data.to_csv(certain_data_path, index=False)
        print(
            f'ori data: {len(ori_feature_data)}, uncertain data: {len(uncertain_data)}, certain data: {len(certain_data)}')
        logging.info(
            f'ori data: {len(ori_feature_data)}, uncertain data: {len(uncertain_data)}, certain data: {len(certain_data)}')

    else:
        uncertain_data_0 = new_data[new_data[cocertain_row] == 0]
        uncertain_data_1 = new_data[new_data[cocertain_row] == 1]
        uncertain_data_1 = uncertain_data_1[uncertain_data_1[used_uncertain_row] == 1]
        uncertain_data = pd.concat([uncertain_data_0, uncertain_data_1], axis=0)
        # uncertain_data.to_csv(
        #     os.path.join(dst_save_res_root,
        #                  f'{classtype}_uncertain-{config.cocertain_row}-{config.used_uncertain_row}_counmatch.csv'),
        #     index=False)
        uncertain_data.to_csv(uncertain_data_path, index=False)
        certain_data = new_data[~new_data.index.isin(uncertain_data.index)]
        # 只保存结果相关的列，防止占用太多的存储空间
        # certain_data = certain_data[['filename', 'src_ip', 'sport', 'dst_ip', 'dport', 'protocol', 'label',
        #                              'detail_label', 'seq_dist_class', 'seq_distksigma_class', 'sta_dist_class',
        #                              'sta_distksigma_class']]
        certain_data.to_csv(certain_data_path, index=False)
        # certain_data.to_csv(
        #     os.path.join(dst_save_res_root,
        #                  f'{classtype}_certain-{config.cocertain_row}-{config.used_uncertain_row}_counmatch.csv'),
        #     index=False
        # )
        print(
            f'ori data: {len(ori_feature_data)}, uncertain data: {len(uncertain_data)}, certain data: {len(certain_data)}')
        logging.info(
            f'ori data: {len(ori_feature_data)}, uncertain data: {len(uncertain_data)}, certain data: {len(certain_data)}')


def split_latent(ori_path, attack_path, normal_path, resplit=True):
    # 2024.8.16 之前resplit的默认参数是False，可能这会导致数据用的一直是之前的数据，现在把它修改成True
    print('[INFO] Split latent feature')
    logging.info('[INFO] Split latent feature')
    """将测试数据的隐含特征分为正常和攻击"""
    if os.path.exists(attack_path) and os.path.exists(normal_path):
        if resplit:
            data = pd.read_pickle(ori_path)
            attack_data = data[data['label'] == 1]
            normal_data = data[data['label'] == 0]

            attack_data.to_pickle(attack_path)
            normal_data.to_pickle(normal_path)
            print(f'normal: {len(normal_data)}, attack: {len(attack_data)}')
            logging.info(f'normal: {len(normal_data)}, attack: {len(attack_data)}')
        else:
            print('[INFO] Latent splitted, use existing file')
            logging.info('[INFO] Latent splitted, use existing file')

    else:
        print('[INFO] Splitted latent not exist, split now')
        data = pd.read_pickle(ori_path)
        attack_data = data[data['label'] == 1]
        normal_data = data[data['label'] == 0]

        attack_data.to_pickle(attack_path)
        normal_data.to_pickle(normal_path)
        print(f'normal: {len(normal_data)}, attack: {len(attack_data)}')
        logging.info(f'normal: {len(normal_data)}, attack: {len(attack_data)}')


def cluster_uncertain_data_latent(data, eps, minpts):
    """使用隐藏层特征进行聚类"""
    seqla_cols = [f'seqla_feature{i}' for i in range(1, 9)]
    stala_cols = [f'stala_feature{i}' for i in range(1, 9)]
    data[seqla_cols] = data['seqla_feature'].str.strip('[]').str.split(expand=True)
    data[stala_cols] = data['stala_feature'].str.strip('[]').str.split(expand=True)

    if 'cluster_uncertain_la' in data.columns:
        # 如果之前聚类过一次生成过一次结果，这里就要把之前的那一列删掉
        data.drop(['cluster_uncertain_la'], axis=1, inplace=True)
        print('drop cluster_uncertain_la')

    lacluster_data = data[seqla_cols + stala_cols]
    dbscan = DBSCAN(eps=eps, min_samples=minpts)
    dbscan.fit(lacluster_data)
    labels = dbscan.labels_
    labels_pd = pd.DataFrame(labels, columns=['cluster_uncertain_la'])

    new_data = pd.concat([data.reset_index(drop=True), labels_pd.reset_index(drop=True)], axis=1)
    return new_data


def process_single_cluster(data):
    """获取每个簇的统计特征"""
    normal_data = data[data['label'] == 0]
    attack_data = data[data['label'] == 1]
    # print(f'normal: {len(normal_data)}, attack: {len(attack_data)}, {np.unique(attack_data["detail_label"])}')
    G = nx.MultiGraph()
    for index, row in data.iterrows():
        src_ip = row['src_ip']
        sport = row['sport']
        dst_ip = row['dst_ip']
        dport = row['dport']
        ts = row['bidirectional_first_seen_ms']
        label = row['sta_label']
        detail_label = row['sta_detail_label']

        G.add_edge(src_ip, dst_ip, attribute=[sport, dport, ts], dlabel=[label, detail_label])

    node_num = G.number_of_nodes()
    edge_num = G.number_of_edges()
    # 边数除以节点数
    edge_div_node = edge_num / node_num
    edgenums_bnodes = []
    node_pairs = G.edges()
    for u, v in node_pairs:
        edgenums_bnodes.append(G.number_of_edges(u, v))
    edgenums_bnodes = sorted(edgenums_bnodes)
    # 节点之间的边数量的统计信息
    ebn_max = max(edgenums_bnodes)
    ebn_avg = sum(edgenums_bnodes) / len(edgenums_bnodes)
    ebn_mid = edgenums_bnodes[int(len(edgenums_bnodes) * 0.5)]
    ebn_75 = edgenums_bnodes[int(len(edgenums_bnodes) * 0.75)]
    # 连通分量数量
    compoment_num = nx.number_connected_components(G)
    edge_div_com = edge_num / compoment_num
    node_div_com = node_num / compoment_num
    # 分别统计连通分量节点数和边数
    c_edge_nums = []
    c_node_nums = []
    c_ed_nums = []
    for component in list(nx.connected_components(G)):
        subgraph = G.subgraph(component).copy()
        c_edge_nums.append(subgraph.number_of_edges())
        c_node_nums.append(subgraph.number_of_nodes())
        c_ed_nums.append(subgraph.number_of_edges() / subgraph.number_of_nodes())
    c_edge_num_max = max(c_edge_nums)
    c_edge_num_avg = sum(c_edge_nums) / len(c_edge_nums)
    c_node_num_max = max(c_node_nums)
    c_node_num_avg = sum(c_node_nums) / len(c_node_nums)
    c_ed_nums_max = max(c_ed_nums)
    c_ed_nums_avg = sum(c_ed_nums) / len(c_ed_nums)
    # print(
    #     f'{edge_num}, {node_num}, {edge_div_node}, {ebn_max}, {ebn_avg}, {ebn_mid}, {ebn_75}, {compoment_num}, {edge_div_com}, {node_div_com} '
    #     f'{c_edge_num_max}, {c_edge_num_avg}, {c_node_num_max}, {c_node_num_avg}, {c_ed_nums_max}, {c_ed_nums_avg}')
    return [len(normal_data), len(attack_data), len(np.unique(attack_data['detail_label'])),
            edge_num, node_num, edge_div_node, ebn_max, ebn_avg, ebn_mid, ebn_75, compoment_num, edge_div_com,
            node_div_com, c_edge_num_max, c_edge_num_avg, c_node_num_max, c_node_num_avg, c_ed_nums_max, c_ed_nums_avg]


def get_each_cluster_info(clustered_data):
    """获得每个簇的统计特征"""
    res = []
    res_col = ['uncertain_cluster', 'len_normal', 'len_attack', 'attack_type_num', 'edge_num', 'node_num',
               'edge_div_node',
               'ebn_max', 'ebn_avg', 'ebn_mid', 'ebn_75', 'compoment_num', 'edge_div_com', 'node_div_com',
               'c_edge_num_max',
               'c_edge_num_avg',
               'c_node_num_max', 'c_node_num_avg', 'c_ed_nums_max', 'c_ed_nums_avg']

    for cl in np.unique(clustered_data['cluster_uncertain_la']):
        # print(f'== {cl} ==')
        cur_data = clustered_data[clustered_data['cluster_uncertain_la'] == cl]
        cur_res = process_single_cluster(cur_data)
        cur_res2 = [cl] + cur_res
        res.append(cur_res2)
    res = pd.DataFrame(res, columns=res_col)
    return res


def get_final_class(row, edge_div_node_threshold=5, edge_div_com_threshold=10):
    # edge_div_node_threshold = 5   # 边数除以节点数
    # edge_div_com_threshold = 10   # 边数除以连通分量数
    # 如果是噪声，说明数据是比较分散的，那就是正常
    if row['uncertain_cluster'] == -1:
        return 0
    if row['edge_div_node'] > edge_div_node_threshold:
        return 1
    if row['edge_div_com'] > edge_div_com_threshold:
        return 1
    return 0


def get_final_class_map(data, edge_div_node_threshold, edge_div_com_threshold):
    """判定最终的结果，生成簇和最终类别的映射"""
    data['final_class'] = data.apply(get_final_class, axis=1, args=(edge_div_node_threshold, edge_div_com_threshold))
    final_class_map = {}
    for i in np.unique(data['uncertain_cluster']):
        final_class_map[i] = data[data['uncertain_cluster'] == i]['final_class'].values[0]
    return final_class_map, data


def save_cluster_mid_info(mid_file_root, mid_info, file_name_1, file_name_2):
    """将中间信息存储起来"""
    if not os.path.exists(mid_file_root):
        os.makedirs(mid_file_root)
        print(f'[INFO] Create new mid file root: {mid_file_root}')
    new_name = file_name_1.strip('.csv') + '-' + file_name_2.strip('.csv') + '.csv'
    mid_info.to_csv(os.path.join(mid_file_root, new_name), index=False)


def cluster_pair_data_all(file_root_1, file_root_2, file_name_1, file_name_2, mid_file_root, eps=0.5, minpts=5,
                          edge_div_node_threshold=5, edge_div_com_threshold=10):
    """聚类，生成最终样本的类别，两个文件的标签都更新"""
    print('[INFO] Cluster uncertain data')
    logging.info('[INFO] Cluster uncertain data')

    data_1 = pd.read_csv(os.path.join(file_root_1, file_name_1))
    data_2 = pd.read_csv(os.path.join(file_root_2, file_name_2))
    data = pd.concat([data_1, data_2], axis=0)
    # 对样本聚类
    clustered_data = cluster_uncertain_data_latent(data, eps, minpts)
    # 获得每个簇的统计信息
    cluster_sta_info = get_each_cluster_info(clustered_data)
    # 判断每个簇是正常还是异常
    final_class_map, cluster_sta_info = get_final_class_map(cluster_sta_info, edge_div_node_threshold,
                                                            edge_div_com_threshold)
    # 存储中间信息，中间信息指的是两个簇一起聚类，聚类后的统计信息
    save_cluster_mid_info(mid_file_root, cluster_sta_info, file_name_1, file_name_2)
    # 生成每个样本的标签，并存储
    data_11 = clustered_data.head(len(data_1))
    data_11['final_class'] = data_11['cluster_uncertain_la'].map(final_class_map)
    data_11.to_csv(os.path.join(file_root_1, file_name_1), index=False)
    data_22 = clustered_data.tail(len(data_2))
    data_22['final_class'] = data_22['cluster_uncertain_la'].map(final_class_map)
    data_22.to_csv(os.path.join(file_root_2, file_name_2), index=False)


def cal_eval_res(normal_normal, normal_attack, attack_attack, attack_normal):
    accuracy = (normal_normal + attack_attack) / (normal_normal + normal_attack + attack_attack + attack_normal + 1e-7)
    recall = attack_attack / (attack_normal + attack_attack + 1e-7)  # 在所有实际为攻击的案例中，模型正确标记为攻击的比例
    precision = attack_attack / (normal_attack + attack_attack + 1e-7)  # 在所有被模型标记为攻击的案例中，实际为攻击的比例
    f1 = 2 * recall * precision / (recall + precision + 1e-7)
    fnr = attack_normal / (attack_attack + attack_normal + 1e-7)  # 漏报率：所有实际为攻击的案例中，模型错误地标记为正常的比例
    far = normal_attack / (attack_attack + normal_attack + 1e-7)  # 误报率：在所有被模型标记为攻击的案例中，实际为正常的比例
    return accuracy, recall, precision, f1, fnr, far


def get_all_eval_res_drift_abl2(cluster_mid_info_path):
    # 把所有漂移的都视为恶意的；把噪声视为正常
    print('==== ecnet way: ====')
    logging.info('==== ecnet way: ====')
    # 评估两个部分对最终结果的影响程度
    uncertain_mid_data = pd.read_csv(cluster_mid_info_path)
    uncertain_noise = uncertain_mid_data[uncertain_mid_data['uncertain_cluster'] == -1]
    uncertain_true = uncertain_mid_data[uncertain_mid_data['uncertain_cluster'] != -1]
    # uncertain_true_normal = uncertain_true[uncertain_true[used_col] <= threshold]  # 被判定为normal的样本的数量
    # uncertain_true_attack = uncertain_true[uncertain_true[used_col] > threshold]  # 被判定是attack的样本的数量
    uncertain_normal_normal = sum(uncertain_noise['len_normal'])  # 实际为normal，被判定为normal
    uncertain_attack_normal = sum(uncertain_noise['len_attack'])  # 实际为attack，被判定为normal
    uncertain_attack_attack = sum(uncertain_true['len_attack'])
    uncertain_normal_attack = sum(uncertain_true['len_normal'])  # 实际为normal，被判定为attack

    print(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')
    logging.info(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')

    # 未知样本的性能
    u_accuracy, u_recall, u_precision, u_f1, u_fnr, u_far = cal_eval_res(uncertain_normal_normal,
                                                                         uncertain_normal_attack,
                                                                         uncertain_attack_attack,
                                                                         uncertain_attack_normal)
    print(f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')
    logging.info(
        f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')


def get_all_eval_res_drift_abl3(cluster_mid_info_path):
    # 把所有漂移的都视为恶意的
    print('==== ecnet way all malicious: ====')
    logging.info('==== ecnet way malicious: ====')
    # 评估两个部分对最终结果的影响程度
    uncertain_mid_data = pd.read_csv(cluster_mid_info_path)
    uncertain_noise = uncertain_mid_data[uncertain_mid_data['uncertain_cluster'] == -1]
    uncertain_true = uncertain_mid_data[uncertain_mid_data['uncertain_cluster'] != -1]
    # uncertain_true_normal = uncertain_true[uncertain_true[used_col] <= threshold]  # 被判定为normal的样本的数量
    # uncertain_true_attack = uncertain_true[uncertain_true[used_col] > threshold]  # 被判定是attack的样本的数量
    uncertain_normal_normal = 0  # 实际为normal，被判定为normal
    uncertain_attack_normal = 0  # 实际为attack，被判定为normal
    uncertain_attack_attack = sum(uncertain_mid_data['len_attack'])
    uncertain_normal_attack = sum(uncertain_mid_data['len_normal'])  # 实际为normal，被判定为attack

    print(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')
    logging.info(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')

    # 未知样本的性能
    u_accuracy, u_recall, u_precision, u_f1, u_fnr, u_far = cal_eval_res(uncertain_normal_normal,
                                                                         uncertain_normal_attack,
                                                                         uncertain_attack_attack,
                                                                         uncertain_attack_normal)
    print(f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')
    logging.info(
        f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')


def get_all_eval_res_drift_abl(cluster_mid_info_path, threshold, used_col):
    print('==== drift identify abl ====')
    print(f'used_col: {used_col}, thre: {threshold}')
    logging.info('==== drift identify abl ====')
    logging.info(f'used_col: {used_col}, thre: {threshold}')
    # 评估两个部分对最终结果的影响程度
    uncertain_mid_data = pd.read_csv(cluster_mid_info_path)
    uncertain_noise = uncertain_mid_data[uncertain_mid_data['uncertain_cluster'] == -1]
    uncertain_true = uncertain_mid_data[uncertain_mid_data['uncertain_cluster'] != -1]
    uncertain_true_normal = uncertain_true[uncertain_true[used_col] <= threshold]  # 被判定为normal的样本的数量
    uncertain_true_attack = uncertain_true[uncertain_true[used_col] > threshold]  # 被判定是attack的样本的数量
    uncertain_normal_normal = sum(uncertain_true_normal['len_normal']) + sum(
        uncertain_noise['len_normal'])  # 实际为normal，被判定为normal
    uncertain_attack_normal = sum(uncertain_true_normal['len_attack']) + sum(
        uncertain_noise['len_attack'])  # 实际为attack，被判定为normal
    uncertain_attack_attack = sum(uncertain_true_attack['len_attack'])
    uncertain_normal_attack = sum(uncertain_true_attack['len_normal'])  # 实际为normal，被判定为attack

    print(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')
    logging.info(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')

    # 未知样本的性能
    u_accuracy, u_recall, u_precision, u_f1, u_fnr, u_far = cal_eval_res(uncertain_normal_normal,
                                                                         uncertain_normal_attack,
                                                                         uncertain_attack_attack,
                                                                         uncertain_attack_normal)
    print(f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')
    logging.info(
        f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')


def get_all_eval_res_new(certain_normal_path, certain_attack_path, cluster_mid_info_path, threshold):
    # 使用c edge num max生成评价指标，直接使用之前聚类好的结果
    print('[INFO] Generate evaluation result of all data, use c_edge_num_max.')
    logging.info('[INFO] Generate evaluation result of all data, use c_edge_num_max.')
    # 对certain计数
    certain_normal_data = pd.read_csv(certain_normal_path)
    certain_normal_normal = len(certain_normal_data[certain_normal_data['seq_dist_class'] == 0])
    certain_normal_attack = len(certain_normal_data[certain_normal_data['seq_dist_class'] == 1])
    certain_attack_data = pd.read_csv(certain_attack_path)
    certain_attack_attack = len(certain_attack_data[certain_attack_data['seq_dist_class'] == 1])
    certain_attack_normal = len(certain_attack_data[certain_attack_data['seq_dist_class'] == 0])

    logging.info('\n======================================================')
    logging.info('******************* final res, use c_edge_num_max ************************')
    logging.info(f'=== certain normal: {certain_normal_path}')
    print('\n======================================================')
    print('******************* final res, use c_edge_num_max ************************')
    print(f'=== certain normal: {certain_normal_path}')
    print(
        f'certain: a-a {certain_attack_attack}, a-n {certain_attack_normal}, n-n {certain_normal_normal}, n-a {certain_normal_attack}')
    logging.info(
        f'certain: a-a {certain_attack_attack}, a-n {certain_attack_normal}, n-n {certain_normal_normal}, n-a {certain_normal_attack}')
    c_accuracy, c_recall, c_precision, c_f1, c_fnr, c_far = cal_eval_res(certain_normal_normal, certain_normal_attack,
                                                                         certain_attack_attack, certain_attack_normal)
    print(f'certain acc: {c_accuracy}, rec: {c_recall}, pre: {c_precision}, f1: {c_f1}, fnr: {c_fnr}, far: {c_far}')
    logging.info(
        f'certain acc: {c_accuracy}, rec: {c_recall}, pre: {c_precision}, f1: {c_f1}, fnr: {c_fnr}, far: {c_far}')

    uncertain_mid_data = pd.read_csv(cluster_mid_info_path)
    uncertain_noise = uncertain_mid_data[uncertain_mid_data['uncertain_cluster'] == -1]
    uncertain_true = uncertain_mid_data[uncertain_mid_data['uncertain_cluster'] != -1]
    uncertain_true_normal = uncertain_true[uncertain_true['c_ed_nums_avg'] <= threshold]  # 被判定为normal的样本的数量
    uncertain_true_attack = uncertain_true[uncertain_true['c_ed_nums_avg'] > threshold]  # 被判定是attack的样本的数量
    uncertain_normal_normal = sum(uncertain_true_normal['len_normal']) + sum(
        uncertain_noise['len_normal'])  # 实际为normal，被判定为normal
    uncertain_attack_normal = sum(uncertain_true_normal['len_attack']) + sum(
        uncertain_noise['len_attack'])  # 实际为attack，被判定为normal
    uncertain_attack_attack = sum(uncertain_true_attack['len_attack'])
    uncertain_normal_attack = sum(uncertain_true_attack['len_normal'])  # 实际为normal，被判定为attack

    print(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')
    logging.info(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')

    # 未知样本占比
    uncertain_per = (
                            uncertain_normal_normal + uncertain_normal_attack + uncertain_attack_normal + uncertain_attack_attack) / (
                            uncertain_normal_normal + uncertain_normal_attack + uncertain_attack_normal + uncertain_attack_attack + certain_normal_normal + certain_normal_attack + certain_attack_normal + certain_attack_attack)
    normal_uncertain_per = (uncertain_normal_normal + uncertain_normal_attack) / (
            uncertain_normal_normal + uncertain_normal_attack + certain_normal_normal + certain_normal_attack)
    attack_uncertain_per = (uncertain_attack_attack + uncertain_attack_normal) / (
            uncertain_attack_attack + uncertain_attack_normal + certain_attack_normal + certain_attack_attack)
    print(
        f'uncertain_per: {uncertain_per}, normal_uncertain_per: {normal_uncertain_per}, attack_uncertain_per: {attack_uncertain_per}')
    logging.info(
        f'uncertain_per: {uncertain_per}, normal_uncertain_per: {normal_uncertain_per}, attack_uncertain_per: {attack_uncertain_per}')

    # 未知样本的性能
    u_accuracy, u_recall, u_precision, u_f1, u_fnr, u_far = cal_eval_res(uncertain_normal_normal,
                                                                         uncertain_normal_attack,
                                                                         uncertain_attack_attack,
                                                                         uncertain_attack_normal)
    print(f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')
    logging.info(
        f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')

    # 全部样本的性能
    a_accuracy, a_recall, a_precision, a_f1, a_fnr, a_far = cal_eval_res(
        uncertain_normal_normal + certain_normal_normal,
        uncertain_normal_attack + certain_normal_attack,
        uncertain_attack_attack + certain_attack_attack,
        uncertain_attack_normal + certain_attack_normal)
    print(f'all acc: {a_accuracy}, rec: {a_recall}, pre: {a_precision}, f1: {a_f1}, fnr: {a_fnr}, far: {a_far}')
    logging.info(f'all acc: {a_accuracy}, rec: {a_recall}, pre: {a_precision}, f1: {a_f1}, fnr: {a_fnr}, far: {a_far}')

    logging.info('******************************************************')
    logging.info('======================================================\n')
    print('******************************************************')
    print('======================================================\n')


def get_all_eval_res_with_class(certain_normal_path, certain_attack_path, uncertain_normal_path, uncertain_attack_path,
                                used_class_row):
    """指定生成结果的时候用的是哪个列；和get_all_eval_res基本是一样的，只是多了used_class_row"""
    print('[INFO] Generate evaluation result of all data.')
    logging.info('[INFO] Generate evaluation result of all data.')
    # 对certain计数
    certain_normal_data = pd.read_csv(certain_normal_path)
    certain_normal_normal = len(certain_normal_data[certain_normal_data[used_class_row] == 0])
    certain_normal_attack = len(certain_normal_data[certain_normal_data[used_class_row] == 1])
    certain_attack_data = pd.read_csv(certain_attack_path)
    certain_attack_attack = len(certain_attack_data[certain_attack_data[used_class_row] == 1])
    certain_attack_normal = len(certain_attack_data[certain_attack_data[used_class_row] == 0])

    logging.info('\n======================================================')
    logging.info(f'used class row: {used_class_row}')
    logging.info('******************* final res ************************')
    logging.info(f'=== certain normal: {certain_normal_path}')
    print('\n======================================================')
    print(f'used class row: {used_class_row}')
    print('******************* final res ************************')
    print(f'=== certain normal: {certain_normal_path}')
    print(
        f'certain: a-a {certain_attack_attack}, a-n {certain_attack_normal}, n-n {certain_normal_normal}, n-a {certain_normal_attack}')
    logging.info(
        f'certain: a-a {certain_attack_attack}, a-n {certain_attack_normal}, n-n {certain_normal_normal}, n-a {certain_normal_attack}')
    c_accuracy, c_recall, c_precision, c_f1, c_fnr, c_far = cal_eval_res(certain_normal_normal, certain_normal_attack,
                                                                         certain_attack_attack, certain_attack_normal)
    print(f'certain acc: {c_accuracy}, rec: {c_recall}, pre: {c_precision}, f1: {c_f1}, fnr: {c_fnr}, far: {c_far}')
    logging.info(
        f'certain acc: {c_accuracy}, rec: {c_recall}, pre: {c_precision}, f1: {c_f1}, fnr: {c_fnr}, far: {c_far}')

    # 对uncertain计数
    uncertain_normal_data = pd.read_csv(uncertain_normal_path)
    uncertain_normal_normal = len(uncertain_normal_data[uncertain_normal_data['final_class'] == 0])
    uncertain_normal_attack = len(uncertain_normal_data[uncertain_normal_data['final_class'] == 1])
    attack_data = pd.read_csv(uncertain_attack_path)
    uncertain_attack_attack = len(attack_data[attack_data['final_class'] == 1])
    uncertain_attack_normal = len(attack_data[attack_data['final_class'] == 0])
    print(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')
    logging.info(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')

    # 未知样本占比
    uncertain_per = (
                            uncertain_normal_normal + uncertain_normal_attack + uncertain_attack_normal + uncertain_attack_attack) / (
                            uncertain_normal_normal + uncertain_normal_attack + uncertain_attack_normal + uncertain_attack_attack + certain_normal_normal + certain_normal_attack + certain_attack_normal + certain_attack_attack)
    normal_uncertain_per = (uncertain_normal_normal + uncertain_normal_attack) / (
            uncertain_normal_normal + uncertain_normal_attack + certain_normal_normal + certain_normal_attack)
    attack_uncertain_per = (uncertain_attack_attack + uncertain_attack_normal) / (
            uncertain_attack_attack + uncertain_attack_normal + certain_attack_normal + certain_attack_attack)
    print(
        f'uncertain_per: {uncertain_per}, normal_uncertain_per: {normal_uncertain_per}, attack_uncertain_per: {attack_uncertain_per}')
    logging.info(
        f'uncertain_per: {uncertain_per}, normal_uncertain_per: {normal_uncertain_per}, attack_uncertain_per: {attack_uncertain_per}')

    # 未知样本的性能
    u_accuracy, u_recall, u_precision, u_f1, u_fnr, u_far = cal_eval_res(uncertain_normal_normal,
                                                                         uncertain_normal_attack,
                                                                         uncertain_attack_attack,
                                                                         uncertain_attack_normal)
    print(f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')
    logging.info(
        f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')

    # 全部样本的性能
    a_accuracy, a_recall, a_precision, a_f1, a_fnr, a_far = cal_eval_res(
        uncertain_normal_normal + certain_normal_normal,
        uncertain_normal_attack + certain_normal_attack,
        uncertain_attack_attack + certain_attack_attack,
        uncertain_attack_normal + certain_attack_normal)
    print(f'all acc: {a_accuracy}, rec: {a_recall}, pre: {a_precision}, f1: {a_f1}, fnr: {a_fnr}, far: {a_far}')
    logging.info(f'all acc: {a_accuracy}, rec: {a_recall}, pre: {a_precision}, f1: {a_f1}, fnr: {a_fnr}, far: {a_far}')

    logging.info('******************************************************')
    logging.info('======================================================\n')
    print('******************************************************')
    print('======================================================\n')


def get_all_eval_res(certain_normal_path, certain_attack_path, uncertain_normal_path, uncertain_attack_path):
    """生成整体的评估结果"""
    print('[INFO] Generate evaluation result of all data.')
    logging.info('[INFO] Generate evaluation result of all data.')
    # 对certain计数
    certain_normal_data = pd.read_csv(certain_normal_path)
    certain_normal_normal = len(certain_normal_data[certain_normal_data['seq_dist_class'] == 0])
    certain_normal_attack = len(certain_normal_data[certain_normal_data['seq_dist_class'] == 1])
    certain_attack_data = pd.read_csv(certain_attack_path)
    certain_attack_attack = len(certain_attack_data[certain_attack_data['seq_dist_class'] == 1])
    certain_attack_normal = len(certain_attack_data[certain_attack_data['seq_dist_class'] == 0])

    logging.info('\n======================================================')
    logging.info('******************* final res ************************')
    logging.info(f'=== certain normal: {certain_normal_path}')
    print('\n======================================================')
    print('******************* final res ************************')
    print(f'=== certain normal: {certain_normal_path}')
    print(
        f'certain: a-a {certain_attack_attack}, a-n {certain_attack_normal}, n-n {certain_normal_normal}, n-a {certain_normal_attack}')
    logging.info(
        f'certain: a-a {certain_attack_attack}, a-n {certain_attack_normal}, n-n {certain_normal_normal}, n-a {certain_normal_attack}')
    c_accuracy, c_recall, c_precision, c_f1, c_fnr, c_far = cal_eval_res(certain_normal_normal, certain_normal_attack,
                                                                         certain_attack_attack, certain_attack_normal)
    print(f'certain acc: {c_accuracy}, rec: {c_recall}, pre: {c_precision}, f1: {c_f1}, fnr: {c_fnr}, far: {c_far}')
    logging.info(
        f'certain acc: {c_accuracy}, rec: {c_recall}, pre: {c_precision}, f1: {c_f1}, fnr: {c_fnr}, far: {c_far}')

    # 对uncertain计数
    uncertain_normal_data = pd.read_csv(uncertain_normal_path)
    uncertain_normal_normal = len(uncertain_normal_data[uncertain_normal_data['final_class'] == 0])
    uncertain_normal_attack = len(uncertain_normal_data[uncertain_normal_data['final_class'] == 1])
    attack_data = pd.read_csv(uncertain_attack_path)
    uncertain_attack_attack = len(attack_data[attack_data['final_class'] == 1])
    uncertain_attack_normal = len(attack_data[attack_data['final_class'] == 0])
    print(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')
    logging.info(
        f'uncertain: a-a {uncertain_attack_attack}, a-n {uncertain_attack_normal}, n-n {uncertain_normal_normal}, n-a {uncertain_normal_attack}')

    # 未知样本占比
    uncertain_per = (
                            uncertain_normal_normal + uncertain_normal_attack + uncertain_attack_normal + uncertain_attack_attack) / (
                            uncertain_normal_normal + uncertain_normal_attack + uncertain_attack_normal + uncertain_attack_attack + certain_normal_normal + certain_normal_attack + certain_attack_normal + certain_attack_attack)
    normal_uncertain_per = (uncertain_normal_normal + uncertain_normal_attack) / (
            uncertain_normal_normal + uncertain_normal_attack + certain_normal_normal + certain_normal_attack)
    attack_uncertain_per = (uncertain_attack_attack + uncertain_attack_normal) / (
            uncertain_attack_attack + uncertain_attack_normal + certain_attack_normal + certain_attack_attack)
    print(
        f'uncertain_per: {uncertain_per}, normal_uncertain_per: {normal_uncertain_per}, attack_uncertain_per: {attack_uncertain_per}')
    logging.info(
        f'uncertain_per: {uncertain_per}, normal_uncertain_per: {normal_uncertain_per}, attack_uncertain_per: {attack_uncertain_per}')

    # 未知样本的性能
    u_accuracy, u_recall, u_precision, u_f1, u_fnr, u_far = cal_eval_res(uncertain_normal_normal,
                                                                         uncertain_normal_attack,
                                                                         uncertain_attack_attack,
                                                                         uncertain_attack_normal)
    print(f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')
    logging.info(
        f'uncertain acc: {u_accuracy}, rec: {u_recall}, pre: {u_precision}, f1: {u_f1}, fnr: {u_fnr}, far: {u_far}')

    # 全部样本的性能
    a_accuracy, a_recall, a_precision, a_f1, a_fnr, a_far = cal_eval_res(
        uncertain_normal_normal + certain_normal_normal,
        uncertain_normal_attack + certain_normal_attack,
        uncertain_attack_attack + certain_attack_attack,
        uncertain_attack_normal + certain_attack_normal)
    print(f'all acc: {a_accuracy}, rec: {a_recall}, pre: {a_precision}, f1: {a_f1}, fnr: {a_fnr}, far: {a_far}')
    logging.info(f'all acc: {a_accuracy}, rec: {a_recall}, pre: {a_precision}, f1: {a_f1}, fnr: {a_fnr}, far: {a_far}')

    logging.info('******************************************************')
    logging.info('======================================================\n')
    print('******************************************************')
    print('======================================================\n')


def fine_split_certain(ori_file_path, saved_root, classtype, used_detail_col='seq_min_dist_col'):
    """将certain分成多个部分，按照离哪个类别更近拆分成多个文件；使用第一阶段训练的模型给第二阶段的数据打标签，用于模型的继续更新"""
    print('[INFO] Fine split certain, save certain data with identified label')
    logging.info('[INFO] Fine split certain, save certain data with identified label')
    data_all = pd.read_csv(ori_file_path)
    # 虽然seq和sta都是判断为normal，但距离哪个更近也还不确定，所以在这里还是把具体类别不一致的去掉
    data_all['equal'] = data_all['sta_min_dist_col'] == data_all['seq_min_dist_col']
    data = data_all[data_all['equal'] == True]
    print(f'[INFO] Certain, {classtype}: equal: {len(data)} / {len(data_all)}')
    logging.info(f'[INFO] Certain, {classtype}: equal: {len(data)} / {len(data_all)}')

    pred_detail_classes = np.unique(data[used_detail_col])
    for dcl in pred_detail_classes:
        cur_data = data[data[used_detail_col] == dcl]
        cur_data.to_csv(os.path.join(saved_root, f'{classtype}-{dcl}.csv'), index=False)
        print(f'{classtype}-{dcl}: {len(cur_data)}')
        logging.info(f'{classtype}-{dcl}: {len(cur_data)}')


def fine_split_uncertain(ori_file_path, saved_root, classtype):
    """将uncertain拆分成多个部分，按照最终类别和簇来决定；使用第一阶段训练的模型给第二阶段的数据打标签，用于模型的继续更新"""
    print('[INFO] Fine split uncertain, save uncertain data with identified label')
    logging.info('[INFO] Fine split uncertain, save uncertain data with identified label')
    data = pd.read_csv(ori_file_path)
    print(f'Uncertain, {classtype}')
    logging.info(f'Uncertain, {classtype}')

    final_classes = np.unique(data['final_class'])
    final_clusters = np.unique(data['cluster_uncertain_la'])
    for final_class in final_classes:
        for final_cluster in final_clusters:
            if final_cluster == -1:
                # 把噪声去掉了，不用于更新
                continue
            cur_data = data[(data['final_class'] == final_class) & (data['cluster_uncertain_la'] == final_cluster)]
            if len(cur_data) != 0:
                cur_data.to_csv(os.path.join(saved_root, f'{classtype}-cla{final_class}_clu{final_cluster}.csv'),
                                index=False)
                print(f'{classtype}-cls{final_class}_clu{final_cluster}: {len(cur_data)}')
                logging.info(f'{classtype}-cls{final_class}_clu{final_cluster}: {len(cur_data)}')


def get_certain_label_cluster(file_path):
    # certain的簇都是在之前训练的时候就出现过的，所以不需要重复定义
    file_name = os.path.basename(file_path).split('-')[-1].rstrip(').csv').lstrip('dist_(')
    tmp = file_name.split(', ')
    # print(f'tmp: {tmp}, filename {os.path.basename(file_path)}')
    label = int(tmp[0])
    cluster = int(tmp[1])
    return label, cluster


class CustomSubset(Dataset):
    def __init__(self, ori_dataset, indices):
        self.data = []
        self.labels = []
        self.msks = []
        self.detail_labels = []
        self.clusters = []
        # 统计信息
        self.sta_scalered = []

        for i in indices:
            self.data.append(ori_dataset.data[i])
            self.labels.append(ori_dataset.labels[i])
            self.msks.append(ori_dataset.msks[i])
            self.detail_labels.append(ori_dataset.detail_labels[i])
            self.clusters.append(ori_dataset.clusters[i])
            self.sta_scalered.append(ori_dataset.sta_scalered[i])
        self.sta_scalered = torch.stack(self.sta_scalered)

        print(f'custom: {len(self.data)}, {len(self.sta_scalered)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.msks[idx], self.detail_labels[idx], self.clusters[idx], \
            self.sta_scalered[idx]


class StaSeqTrafficNormalizedDatasetUpdateOri(Dataset):
    def __init__(self, pickle_files, data_root, label_root, scaler_path, class_dict, myconfig):
        self.data = []
        self.labels = []
        self.msks = []
        self.detail_labels = []
        self.clusters = []
        # 统计信息
        self.sta_data = []

        for file in pickle_files:
            df = pd.read_pickle(os.path.join(data_root, file))
            clusters_ = pd.read_pickle(os.path.join(label_root, file))
            assert len(df) == len(clusters_)
            label = class_dict[file]

            # 序列信息
            df['lengths'] = df['lengths'].apply(ast.literal_eval)
            df['directions'] = df['directions'].apply(ast.literal_eval)
            df['intervals'] = df['intervals'].apply(ast.literal_eval)

            directions = df['lengths'].values
            # 对包长序列归一化
            df['new_directions'] = df['directions'].apply(normalize_row_new)
            lengths = df['new_directions'].values
            flow_num = len(lengths)
            seq, cur_msk = self.process_single(directions, lengths, myconfig.sequence_length)

            self.data.extend(seq)
            self.labels.extend([label] * flow_num)
            self.msks.extend(cur_msk)
            self.clusters.extend(clusters_.values)
            self.detail_labels.extend([-2] * flow_num)  # detail label在后面好像没有用到，这里统一设置成-2保持格式的一致

            # 统计信息
            sta = df[['bidirectional_duration_ms', 'bidirectional_packets', 'bidirectional_bytes', 'src2dst_packets',
                      'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes', 'bidirectional_min_ps',
                      'bidirectional_mean_ps', 'bidirectional_stddev_ps', 'bidirectional_max_ps',
                      'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms',
                      'bidirectional_max_piat_ms', 'bidirectional_syn_packets', 'bidirectional_cwr_packets',
                      'bidirectional_ece_packets', 'bidirectional_urg_packets', 'bidirectional_ack_packets',
                      'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_fin_packets']]
            self.sta_data.append(sta)

        # 统计信息，加载之前的scaler做归一化
        self.sta_data = pd.concat(self.sta_data, axis=0)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        self.sta_scalered = scaler.transform(self.sta_data)
        self.sta_scalered = torch.tensor(self.sta_scalered, dtype=torch.float32)

    def process_single(self, directions, lengths, target_len):
        new_seq = []
        new_masklen = []
        for (direction, length) in zip(directions, lengths):
            assert len(direction) == len(length)
            if len(direction) < target_len:
                new_masklen.append(len(direction))
            else:
                new_masklen.append(target_len)
                direction = direction[:target_len]
                length = length[:target_len]

            vector = torch.zeros(2, target_len)
            for i, (d, l) in enumerate(zip(direction, length)):
                if d == 1:
                    vector[0, i] = l
                else:
                    vector[1, i] = l
            new_seq.append(vector)

        return new_seq, new_masklen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.msks[idx], self.detail_labels[idx], self.clusters[idx], \
            self.sta_scalered[idx]


class StaSeqTrafficNormalizedDatasetUpdate(Dataset):
    # 使用drift identify生成的标签来更新模型时的数据集
    def __init__(self, csv_paths, scaler_path, myconfig):
        self.data = []
        self.labels = []
        self.msks = []
        self.detail_labels = []
        self.clusters = []
        # 统计信息
        self.sta_data = []

        for csv_file in csv_paths:
            label, cluster = get_certain_label_cluster(csv_file)
            df = pd.read_csv(csv_file)

            if len(df) < myconfig.min_sample_each_file_certain:
                print(f'skip {os.path.basename(csv_file)}, len {len(df)}')
                logging.info(f'skip {os.path.basename(csv_file)}, len {len(df)}')
                continue
            else:
                print(f'use {os.path.basename(csv_file)}, len {len(df)}')
                logging.info(f'use {os.path.basename(csv_file)}, len {len(df)}')

            # 序列信息
            df['lengths'] = df['lengths'].apply(ast.literal_eval)
            df['directions'] = df['directions'].apply(ast.literal_eval)
            df['intervals'] = df['intervals'].apply(ast.literal_eval)

            directions = df['lengths'].values
            # 对包长序列归一化
            df['new_directions'] = df['directions'].apply(normalize_row_new)
            lengths = df['new_directions'].values
            flow_num = len(lengths)
            seq, cur_msk = self.process_single(directions, lengths, myconfig.sequence_length)
            self.data.extend(seq)
            self.labels.extend([label] * flow_num)
            self.msks.extend(cur_msk)
            self.clusters.extend([cluster] * flow_num)
            self.detail_labels.extend([-2] * flow_num)  # detail label在后面好像没有用到，这里统一设置成-2保持格式的一致

            # 统计信息
            sta = df[['bidirectional_duration_ms', 'bidirectional_packets', 'bidirectional_bytes', 'src2dst_packets',
                      'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes', 'bidirectional_min_ps',
                      'bidirectional_mean_ps', 'bidirectional_stddev_ps', 'bidirectional_max_ps',
                      'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms',
                      'bidirectional_max_piat_ms', 'bidirectional_syn_packets', 'bidirectional_cwr_packets',
                      'bidirectional_ece_packets', 'bidirectional_urg_packets', 'bidirectional_ack_packets',
                      'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_fin_packets']]
            self.sta_data.append(sta)
        # 统计信息，加载之前的scaler做归一化
        self.sta_data = pd.concat(self.sta_data, axis=0)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        self.sta_scalered = scaler.transform(self.sta_data)
        self.sta_scalered = torch.tensor(self.sta_scalered, dtype=torch.float32)

    def process_single(self, directions, lengths, target_len):
        new_seq = []
        new_masklen = []
        for (direction, length) in zip(directions, lengths):
            assert len(direction) == len(length)
            if len(direction) < target_len:
                new_masklen.append(len(direction))
            else:
                new_masklen.append(target_len)
                direction = direction[:target_len]
                length = length[:target_len]

            vector = torch.zeros(2, target_len)
            for i, (d, l) in enumerate(zip(direction, length)):
                if d == 1:
                    vector[0, i] = l
                else:
                    vector[1, i] = l
            new_seq.append(vector)

        return new_seq, new_masklen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.msks[idx], self.detail_labels[idx], self.clusters[idx], \
            self.sta_scalered[idx]


# def cal_eudist_sum(tensor1, tensor2):
#     distances = torch.norm(tensor1 - tensor2, dim=1)
#     total_distance = torch.sum(distances) / tensor1.shape[0]
#     return total_distance


def certain_train_update_model_noconstrain(myconfig, train_loader_weighted, old_seq_model_path, old_sta_model_path,
                                           new_model_root):
    """和certain_train_update_model类似，但在更新的过程中没有使用约束"""

    """适用certain数据更新模型"""
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model.load_state_dict(torch.load(old_sta_model_path))
    seq_model.load_state_dict(torch.load(old_seq_model_path))

    # 旧模型用于计算这些数据在原来的模型上是怎么样的
    sta_model_old = MyModelStaAE().to(myconfig.device)
    seq_model_old = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                       d_model=myconfig.d_model, nhead=myconfig.nhead,
                                       num_encoder_layers=myconfig.num_encoder_layers,
                                       num_decoder_layers=myconfig.num_decoder_layers,
                                       dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model_old.load_state_dict(torch.load(old_sta_model_path))
    seq_model_old.load_state_dict(torch.load(old_seq_model_path))

    seq_reconst_criterion = nn.MSELoss(reduction='sum')
    seq_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.seq_margin)
    sta_reconst_criterion = nn.MSELoss(reduction='sum')
    sta_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.sta_margin)

    seq_optimizer = optim.AdamW(seq_model.parameters(), lr=myconfig.seq_lr)
    seq_scheduler = CosineAnnealingLR(seq_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.seq_eta_min)
    sta_optimizer = optim.AdamW(sta_model.parameters(), lr=myconfig.sta_lr)
    sta_scheduler = CosineAnnealingLR(sta_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.sta_eta_min)

    for epoch in range(myconfig.train_epoch):
        sta_model.train()
        seq_model.train()
        sta_model_old.train()
        seq_model_old.train()

        seq_contra_loss_epoch = 0
        seq_recon_loss_epoch = 0
        seq_distance_newold = 0
        seq_loss_epoch = 0
        sta_contra_loss_epoch = 0
        sta_recon_loss_epoch = 0
        sta_distance_newold = 0
        sta_loss_epoch = 0

        for inputs, labels, masks, detaillabels, clusters, stas in train_loader_weighted:
            # 使用序列信息进行训练
            inputs = inputs.transpose(1, 2)
            inputs = inputs.to(myconfig.device)
            labels = labels.to(myconfig.device)
            clusters = clusters.to(myconfig.device)

            seq_optimizer.zero_grad()

            seq_recon, seq_latent = seq_model(inputs)
            seq_contra_loss = seq_contra_criterion(seq_latent, labels, clusters)
            seq_recon_loss = seq_reconst_criterion(seq_recon, inputs)

            with torch.no_grad():
                # 计算样本特征和旧模型生成的特征之间的距离，需要使用nograd，否则显存会爆炸
                _, seq_latent_old = seq_model_old(inputs)
            # 但是在计算距离的时候需要有梯度，否则无法作用到模型上
            cur_dists_seq = cal_eudist_sum(seq_latent_old, seq_latent)

            seq_loss = myconfig.seq_contra_lamda * seq_contra_loss + myconfig.seq_recon_lamda * seq_recon_loss

            seq_contra_loss_epoch += seq_contra_loss
            seq_recon_loss_epoch += seq_recon_loss
            seq_distance_newold += cur_dists_seq
            seq_loss_epoch += seq_loss

            seq_loss.backward()
            seq_optimizer.step()

            # 使用统计信息进行训练
            stas = stas.to(myconfig.device)

            sta_optimizer.zero_grad()

            sta_latent, sta_recon = sta_model(stas)
            sta_contra_loss = sta_contra_criterion(sta_latent, labels, clusters)
            sta_recon_loss = sta_reconst_criterion(sta_recon, stas)

            with torch.no_grad():
                sta_latent_old, _ = sta_model_old(stas)
            cur_dists_sta = cal_eudist_sum(sta_latent_old, sta_latent)

            sta_loss = myconfig.sta_contra_lamda * sta_contra_loss + myconfig.sta_recon_lamda * sta_recon_loss

            sta_contra_loss_epoch += sta_contra_loss
            sta_recon_loss_epoch += sta_recon_loss
            sta_distance_newold += cur_dists_sta
            sta_loss_epoch += sta_loss

            sta_loss.backward()
            sta_optimizer.step()

        torch.save(seq_model.state_dict(), os.path.join(new_model_root, f'seq_model_{str(epoch)}.pth'))
        torch.save(sta_model.state_dict(), os.path.join(new_model_root, f'sta_model_{str(epoch)}.pth'))

        print(f'Epoch {epoch}')
        print(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist: {seq_distance_newold}')
        print(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist: {sta_distance_newold}')
        logging.info(f'Epoch {epoch}')
        logging.info(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist: {seq_distance_newold}')
        logging.info(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist: {sta_distance_newold}')

        seq_scheduler.step()
        sta_scheduler.step()


def certain_train_update_model(myconfig, train_loader_weighted, old_seq_model_path, old_sta_model_path, new_model_root,
                               use_dist_constraint=True):
    """适用certain数据更新模型"""
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model.load_state_dict(torch.load(old_sta_model_path))
    seq_model.load_state_dict(torch.load(old_seq_model_path))

    # 旧模型用于计算这些数据在原来的模型上是怎么样的
    sta_model_old = MyModelStaAE().to(myconfig.device)
    seq_model_old = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                       d_model=myconfig.d_model, nhead=myconfig.nhead,
                                       num_encoder_layers=myconfig.num_encoder_layers,
                                       num_decoder_layers=myconfig.num_decoder_layers,
                                       dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model_old.load_state_dict(torch.load(old_sta_model_path))
    seq_model_old.load_state_dict(torch.load(old_seq_model_path))

    seq_reconst_criterion = nn.MSELoss(reduction='sum')
    seq_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.seq_margin)
    sta_reconst_criterion = nn.MSELoss(reduction='sum')
    sta_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.sta_margin)

    seq_optimizer = optim.AdamW(seq_model.parameters(), lr=myconfig.seq_lr)
    seq_scheduler = CosineAnnealingLR(seq_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.seq_eta_min)
    sta_optimizer = optim.AdamW(sta_model.parameters(), lr=myconfig.sta_lr)
    sta_scheduler = CosineAnnealingLR(sta_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.sta_eta_min)

    for epoch in range(myconfig.train_epoch):
        sta_model.train()
        seq_model.train()
        sta_model_old.train()
        seq_model_old.train()

        seq_contra_loss_epoch = 0
        seq_recon_loss_epoch = 0
        seq_distance_newold = 0
        seq_loss_epoch = 0
        sta_contra_loss_epoch = 0
        sta_recon_loss_epoch = 0
        sta_distance_newold = 0
        sta_loss_epoch = 0

        for inputs, labels, masks, detaillabels, clusters, stas in train_loader_weighted:
            # 使用序列信息进行训练
            inputs = inputs.transpose(1, 2)
            inputs = inputs.to(myconfig.device)
            labels = labels.to(myconfig.device)
            clusters = clusters.to(myconfig.device)

            seq_optimizer.zero_grad()

            seq_recon, seq_latent = seq_model(inputs)
            seq_contra_loss = seq_contra_criterion(seq_latent, labels, clusters)
            seq_recon_loss = seq_reconst_criterion(seq_recon, inputs)

            with torch.no_grad():
                # 计算样本特征和旧模型生成的特征之间的距离，需要使用nograd，否则显存会爆炸
                _, seq_latent_old = seq_model_old(inputs)
            # 但是在计算距离的时候需要有梯度，否则无法作用到模型上
            cur_dists_seq = cal_eudist_sum(seq_latent_old, seq_latent)

            if use_dist_constraint:
                seq_loss = myconfig.seq_contra_lamda * seq_contra_loss + myconfig.seq_recon_lamda * seq_recon_loss \
                           + myconfig.seq_dist_lamda * cur_dists_seq
            else:
                seq_loss = myconfig.seq_contra_lamda * seq_contra_loss + myconfig.seq_recon_lamda * seq_recon_loss

            seq_contra_loss_epoch += seq_contra_loss
            seq_recon_loss_epoch += seq_recon_loss
            seq_distance_newold += cur_dists_seq
            seq_loss_epoch += seq_loss

            seq_loss.backward()
            seq_optimizer.step()

            # 使用统计信息进行训练
            stas = stas.to(myconfig.device)

            sta_optimizer.zero_grad()

            sta_latent, sta_recon = sta_model(stas)
            sta_contra_loss = sta_contra_criterion(sta_latent, labels, clusters)
            sta_recon_loss = sta_reconst_criterion(sta_recon, stas)

            with torch.no_grad():
                sta_latent_old, _ = sta_model_old(stas)
            cur_dists_sta = cal_eudist_sum(sta_latent_old, sta_latent)

            if use_dist_constraint:
                sta_loss = myconfig.sta_contra_lamda * sta_contra_loss + myconfig.sta_recon_lamda * sta_recon_loss \
                           + myconfig.sta_dist_lamda * cur_dists_sta
            else:
                sta_loss = myconfig.sta_contra_lamda * sta_contra_loss + myconfig.sta_recon_lamda * sta_recon_loss

            sta_contra_loss_epoch += sta_contra_loss
            sta_recon_loss_epoch += sta_recon_loss
            sta_distance_newold += cur_dists_sta
            sta_loss_epoch += sta_loss

            sta_loss.backward()
            sta_optimizer.step()

        torch.save(seq_model.state_dict(), os.path.join(new_model_root, f'seq_model_{str(epoch)}.pth'))
        torch.save(sta_model.state_dict(), os.path.join(new_model_root, f'sta_model_{str(epoch)}.pth'))

        print(f'Epoch {epoch}')
        print(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist: {seq_distance_newold}')
        print(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist: {sta_distance_newold}')
        logging.info(f'Epoch {epoch}')
        logging.info(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist: {seq_distance_newold}')
        logging.info(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist: {sta_distance_newold}')

        seq_scheduler.step()
        sta_scheduler.step()


def get_certain_related(myconfig, train_loader_noweight, seq_model_path, sta_model_path, save_data_root):
    """在更新模型的时候，获得更新好的模型中的隐藏特征"""
    # 获得用certain更新的模型得到的各种潜在特征，统计信息等等
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model.load_state_dict(torch.load(sta_model_path))
    seq_model.load_state_dict(torch.load(seq_model_path))

    # # 获得新模型产生的潜在特征
    print('Get latent feature')
    logging.info('Get latent feature')
    get_seqmid_info_of_train_transformer_sta(seq_model, train_loader_noweight, myconfig, save_data_root)
    get_stamid_info_of_train_ae_sta(sta_model, train_loader_noweight, myconfig, save_data_root)

    # 获得（新模型，新训练数据）每个簇的簇心，将重构损失、到簇心的距离等拟合到95%的数据
    print('Fit distribution with 95% data')
    logging.info('Fit distribution with 95% data')
    get_sta_info_of_train_95_stadata(save_data_root)  # sta
    get_sta_info_of_train_95(save_data_root)  # seq

    # 使用所有数据（新模型，新训练数据）拟合分布
    print('Fit distribution with all data')
    logging.info('Fit distribution with all data')
    get_sta_info_of_train_stadata(save_data_root)
    get_sta_info_of_train(save_data_root)


def update_center(old_centers, new_centers, old_sample_nums, new_sample_nums):
    updated_centers = {}
    for k, v in old_centers.items():
        if k == (-1, -1):
            continue
        # 如果在更新的数据中没有这个簇，就直接保存原来的簇心
        if k not in new_centers:
            updated_centers[k] = v
        else:
            old_num = old_sample_nums[k]
            new_num = new_sample_nums[k]
            upcen = old_num / (new_num + old_num) * old_centers[k] + new_num / (new_num + old_num) * new_centers[k]
            updated_centers[k] = upcen
    return updated_centers


def combined_mean_var(n1, loc1, scale1, n2, loc2, scale2):
    """计算合并后的均值和方差"""
    # 计算合并后的均值
    combined_loc = (n1 * loc1 + n2 * loc2) / (n1 + n2)
    # 计算合并后的方差
    combined_scale = ((n1 - 1) * scale1 + (n2 - 1) * scale2 + n1 * (loc1 - combined_loc) ** 2 + n2 * (
            loc2 - combined_loc) ** 2) / (n1 + n2 - 1)
    return combined_scale, combined_scale


def update_mean_var(old_distri, new_distri, old_sample_nums, new_sample_nums):
    updated_distri = {}
    for k, v in old_distri.items():
        # -1 -1 也按照样本数量合并
        # if k == (-1, -1):
        #     continue
        if k not in new_distri:
            updated_distri[k] = v
        else:
            old_mean = v['loc']
            old_var = v['scale']
            new_mean = new_distri[k]['loc']
            new_var = new_distri[k]['scale']
            umean, uvar = combined_mean_var(old_sample_nums[k], old_mean, old_var, new_sample_nums[k], new_mean,
                                            new_var)
            updated_distri[k] = {'loc': umean, 'scale': uvar}
    return updated_distri


def get_cluster_sample_nums(save_data_root):
    """ 统计每个簇的样本的数量，单独存在一个文件中；在第一次训练的时候没用到这些信息，
    但是在模型更新的时候可能需要根据数量去更新，所以在这里重新生成一个文件 """
    ori_path = os.path.join(save_data_root, 'train_mid_info.pickle')
    data = pd.read_pickle(ori_path)
    count_dict = {}
    for l in np.unique(data['label']):
        for c in np.unique(data['cluster']):
            tmp_data = data[(data['label'] == l) & (data['cluster'] == c)]
            count_dict[(l, c)] = len(tmp_data)
    print(count_dict)
    count_dict[(-1, -1)] = len(data)
    with open(os.path.join(save_data_root, 'train_sample_count.pickle'), 'wb') as f:
        pickle.dump(count_dict, f)


def combine_two_distribution_certain(old_save_data_root, new_save_data_root, updated_save_data_root):
    """将新模型新数据得到的簇心、重构损失等和之前的模型的这些内容按照样本数量比例融合到一起"""

    # # 计算新旧数据每个簇的样本数量
    get_cluster_sample_nums(old_save_data_root)
    get_cluster_sample_nums(new_save_data_root)

    old_sample_num_path = os.path.join(old_save_data_root, 'train_sample_count.pickle')
    new_sample_num_path = os.path.join(new_save_data_root, 'train_sample_count.pickle')

    # 合并簇心
    old_seq_centerinfo_path = os.path.join(old_save_data_root, 'train_sta_info.pickle')
    old_sta_centerinfo_path = os.path.join(old_save_data_root, 'train_sta_info_stadata.pickle')
    new_seq_centerinfo_path = os.path.join(new_save_data_root, 'train_sta_info.pickle')
    new_sta_centerinfo_path = os.path.join(new_save_data_root, 'train_sta_info_stadata.pickle')
    with open(old_seq_centerinfo_path, 'rb') as f:
        old_seqcenter_data = pickle.load(f)['center_dict']
    with open(new_seq_centerinfo_path, 'rb') as f:
        new_seqcenter_data = pickle.load(f)['center_dict']
    with open(old_sta_centerinfo_path, 'rb') as f:
        old_stacenter_data = pickle.load(f)['center_dict']
    with open(new_sta_centerinfo_path, 'rb') as f:
        new_stacenter_data = pickle.load(f)['center_dict']
    with open(old_sample_num_path, 'rb') as f:
        old_sample_num = pickle.load(f)
    with open(new_sample_num_path, 'rb') as f:
        new_sample_num = pickle.load(f)
    updated_seq_centers = update_center(old_seqcenter_data, new_seqcenter_data, old_sample_num, new_sample_num)
    updated_sta_centers = update_center(old_stacenter_data, new_stacenter_data, old_sample_num, new_sample_num)
    # 保存到相应文件中
    updated_seq_centers_dict = {'center_dict': updated_seq_centers}
    updated_sta_centers_dict = {'center_dict': updated_sta_centers}
    with open(os.path.join(updated_save_data_root, 'train_sta_info.pickle'), 'wb') as f:
        pickle.dump(updated_seq_centers_dict, f)
    with open(os.path.join(updated_save_data_root, 'train_sta_info_stadata.pickle'), 'wb') as f:
        pickle.dump(updated_sta_centers_dict, f)

    # 合并到簇心距离的均值和方差
    old_seq_distrinfo_path = os.path.join(old_save_data_root, 'train_distri_para_info.pickle')
    new_seq_distrinfo_path = os.path.join(new_save_data_root, 'train_distri_para_info.pickle')
    old_sta_distrinfo_path = os.path.join(old_save_data_root, 'train_distri_para_info_stadata.pickle')
    new_sta_distrinfo_path = os.path.join(new_save_data_root, 'train_distri_para_info_stadata.pickle')
    with open(old_seq_distrinfo_path, 'rb') as f:
        old_seq_distri = pickle.load(f)
    with open(new_seq_distrinfo_path, 'rb') as f:
        new_seq_distri = pickle.load(f)
    with open(old_sta_distrinfo_path, 'rb') as f:
        old_sta_distri = pickle.load(f)
    with open(new_sta_distrinfo_path, 'rb') as f:
        new_sta_distri = pickle.load(f)
    update_seq_dist_distri = update_mean_var(old_seq_distri['dist_distri'], new_seq_distri['dist_distri'],
                                             old_sample_num, new_sample_num)
    update_seq_recon_distri = update_mean_var(old_seq_distri['reclos_distri'], new_seq_distri['reclos_distri'],
                                              old_sample_num, new_sample_num)
    update_sta_dist_distri = update_mean_var(old_sta_distri['dist_distri'], new_sta_distri['dist_distri'],
                                             old_sample_num, new_sample_num)
    update_sta_recon_distri = update_mean_var(old_sta_distri['reclos_distri'], new_sta_distri['reclos_distri'],
                                              old_sample_num, new_sample_num)
    # 保存到相应文件中
    update_seq_distri_dict = {'dist_distri': update_seq_dist_distri, 'reclos_distri': update_seq_recon_distri}
    update_sta_distri_dict = {'dist_distri': update_sta_dist_distri, 'reclos_distri': update_sta_recon_distri}
    with open(os.path.join(updated_save_data_root, 'train_distri_para_info.pickle'), 'wb') as f:
        pickle.dump(update_seq_distri_dict, f)
    with open(os.path.join(updated_save_data_root, 'train_distri_para_info_stadata.pickle'), 'wb') as f:
        pickle.dump(update_sta_distri_dict, f)


def eval_model_stage2(eval_part, myconfig, test_loader, model_root, res_root, train_save_data_root,
                      test_save_data_root, used_seq_model, used_sta_model):
    print('+++++++++++++++++++++++++++++++++++++++++++++')
    print(f'++++++++++++++++ {eval_part} ++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++')
    logging.info('+++++++++++++++++++++++++++++++++++++++++++++')
    logging.info(f'++++++++++++++++ {eval_part} ++++++++++++++++')
    logging.info('+++++++++++++++++++++++++++++++++++++++++++++')

    # ===============
    # 加载新模型，计算测试样本的重构损失和隐藏层特征
    # ===============
    # 获得用certain更新的模型得到的各种潜在特征，统计信息等等
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model.load_state_dict(torch.load(os.path.join(model_root, used_sta_model)))
    seq_model.load_state_dict(torch.load(os.path.join(model_root, used_seq_model)))

    # 获得隐含特征，测试数据在新模型上的中间特征
    print('[INFO] Get latent feature')
    get_seqmid_info_of_test_transformer_sta(seq_model, test_loader, myconfig, test_save_data_root)
    get_stamid_info_of_test_ae_sta(sta_model, test_loader, myconfig, test_save_data_root)

    # 计算测试样本到每个簇心的距离
    print('[INFO] Calculate distance to cluster center')
    get_distance_to_center_test_seq(train_save_data_root, test_save_data_root)
    get_distance_to_center_test_sta(train_save_data_root, test_save_data_root)

    # 计算样本属于每个类别的概率（在更新的数据上没有考虑95%的情况，这里直接使用前面合成的全部数据的分布去拟合）
    print('[INFO] Get k sigma of test samples using all train data.')
    get_test_result_ksigma_seq(train_save_data_root, test_save_data_root, res_root)
    get_test_result_ksigma_sta(train_save_data_root, test_save_data_root, res_root)

    # 使用距离评估测试函数
    generate_eval_pre_data_seq(res_root)
    generate_eval_pre_data_sta(res_root)
    eval_test_data(res_root)


def drop_misclassified_files(csv_paths):
    """删掉用于训练的uncertain文件们中被误分类的那些"""
    res = []
    for path in csv_paths:
        basename = os.path.basename(path)
        if basename.startswith('attack-cla0') or basename.startswith('normal-cla1'):
            continue
        res.append(path)
    return res


def get_uncertain_label_cluster(file_path):
    file_name = os.path.basename(file_path)
    # print(f'------------- {file_name}')
    cla = file_name.split('-')[0]
    if cla == 'normal':
        label = 0
    elif cla == 'attack':
        label = 1
    else:
        print(f'file name error: {file_name}')
        return
    cluster = file_name.split('_')[-1].lstrip('clu').rstrip('.csv')
    cluster = int(cluster) + 1000  # 因为还有原来的cluster标签，为避免重复直接加1000
    # print(f'label {label}, cluster {cluster}')
    return label, cluster


class StaSeqTrafficNormalizedDatasetUpdateUncertain(Dataset):
    def __init__(self, csv_paths, scaler_path, use_mis, myconfig):
        self.data = []
        self.labels = []
        self.msks = []
        self.detail_labels = []
        self.clusters = []
        # 统计信息
        self.sta_data = []

        if not use_mis:
            # 如果不考虑误分类的，那就把误分类的样本删掉，但是在实际中是无法区分是否是误分类的
            csv_paths = drop_misclassified_files(csv_paths)

        for csv_file in csv_paths:
            label, cluster = get_uncertain_label_cluster(csv_file)
            df = pd.read_csv(csv_file)

            if len(df) < myconfig.min_sample_each_file_uncertain:
                print(f'skip {os.path.basename(csv_file)}, len {len(df)}')
                logging.info(f'skip {os.path.basename(csv_file)}, len {len(df)}')
                continue

            # 序列信息
            df['lengths'] = df['lengths'].apply(ast.literal_eval)
            df['directions'] = df['directions'].apply(ast.literal_eval)
            df['intervals'] = df['intervals'].apply(ast.literal_eval)

            directions = df['lengths'].values
            # 对包长序列归一化
            df['new_directions'] = df['directions'].apply(normalize_row_new)
            lengths = df['new_directions'].values
            flow_num = len(lengths)
            seq, cur_msk = self.process_single(directions, lengths, myconfig.sequence_length)
            self.data.extend(seq)
            self.labels.extend([label] * flow_num)
            self.msks.extend(cur_msk)
            self.clusters.extend([cluster] * flow_num)
            self.detail_labels.extend([-2] * flow_num)  # detail label在后面好像没有用到，这里统一设置成-2保持格式的一致

            # 统计信息
            sta = df[['bidirectional_duration_ms', 'bidirectional_packets', 'bidirectional_bytes', 'src2dst_packets',
                      'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes', 'bidirectional_min_ps',
                      'bidirectional_mean_ps', 'bidirectional_stddev_ps', 'bidirectional_max_ps',
                      'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms',
                      'bidirectional_max_piat_ms', 'bidirectional_syn_packets', 'bidirectional_cwr_packets',
                      'bidirectional_ece_packets', 'bidirectional_urg_packets', 'bidirectional_ack_packets',
                      'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_fin_packets']]
            self.sta_data.append(sta)

        # 统计信息，加载之前的scaler做归一化
        self.sta_data = pd.concat(self.sta_data, axis=0)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        self.sta_scalered = scaler.transform(self.sta_data)
        self.sta_scalered = torch.tensor(self.sta_scalered, dtype=torch.float32)

    def process_single(self, directions, lengths, target_len):
        new_seq = []
        new_masklen = []
        for (direction, length) in zip(directions, lengths):
            assert len(direction) == len(length)
            if len(direction) < target_len:
                new_masklen.append(len(direction))
            else:
                new_masklen.append(target_len)
                direction = direction[:target_len]
                length = length[:target_len]

            vector = torch.zeros(2, target_len)
            for i, (d, l) in enumerate(zip(direction, length)):
                if d == 1:
                    vector[0, i] = l
                else:
                    vector[1, i] = l
            new_seq.append(vector)

        return new_seq, new_masklen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.msks[idx], self.detail_labels[idx], self.clusters[idx], \
            self.sta_scalered[idx]


# 用在对比学习更新过程中有新类增加时的损失，约束新类和之前的簇心的距离
class ContraLossUpdateNewclass(nn.Module):
    def __init__(self, margin):
        super(ContraLossUpdateNewclass, self).__init__()
        self.margin = margin

    def forward(self, inputs, centers):
        # print('##############################')
        # print(f'inputs: {inputs.shape}')
        # print(f'centers: {centers.shape}')
        distance = cal_distance_to_center_for_loss(inputs, centers)
        # print(f'distance: {distance.shape}')
        # print(f'-- {distance}')
        distance_ = torch.clamp(distance, max=self.margin)
        # print(f'-- {distance_}')
        distance_ = (self.margin - distance_) ** 2
        # print(f'--distance sum: {distance_.sum()}')
        loss = distance_.sum() / (distance_.shape[0] * distance_.shape[1])
        return loss


def cal_distance_to_center_for_loss(features, centers):
    features = features.unsqueeze(1)
    centers = centers.unsqueeze(0)
    distances = torch.norm(features - centers, dim=2)
    return distances


def cal_eudist_sum(tensor1, tensor2):
    distances = torch.norm(tensor1 - tensor2, dim=1)
    total_distance = torch.sum(distances)
    return total_distance


def get_uncertain_to_center_constrain_loss(old_labels, old_clusters, old_inputs, new_inputs, center_dict, myconfig):
    # print('+++++++++++++++++++++++++++++++ to center constrain')
    # 创建一个与 old_inputs 同形状的矩阵来存储簇心向量
    centers = torch.zeros_like(old_inputs)

    # 填充簇心矩阵
    for i in range(old_inputs.size(0)):
        label_cluster_key = (old_labels[i].item(), old_clusters[i].item())
        center_array = center_dict[label_cluster_key]
        center_tensor = torch.from_numpy(center_array).to(myconfig.device)
        centers[i] = center_tensor

    distances_old = torch.norm(old_inputs - centers, p=2, dim=1)
    distances_new = torch.norm(new_inputs - centers, p=2, dim=1)

    distance_diff = torch.sum(distances_new - distances_old)
    # print('distance diff: ')
    # print(distances_new-distances_old)
    # print(distance_diff)
    return distance_diff


def uncertain_train_update_model_to_center_uc_noconstrain(myconfig, train_loader_weighted, old_val_loader,
                                                          old_seq_model_path,
                                                          old_sta_model_path, new_model_path,
                                                          old_seq_centers_path, old_sta_centers_path):
    """和uncertain_train_update_model_to_center_uc这个函数类似，但是没有约束，直接更新"""

    # uncertain_train_update_model在更新的时候是约束正常样本的新表示和原始表示的距离，这个函数约束的是新表示到簇心的距离的变化（其实约束的也是簇心的变化）

    # 在to_center的基础上修改，加了uc，在训练过程中使用旧数据，同时直接用新计算出来的簇心表示；去掉了uncertain样本到已知类簇心的约束

    # 这里加载的是certain更新过的模型
    # 加载旧模型（在此基础上更新）
    # 加载旧模型
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model.load_state_dict(torch.load(old_sta_model_path))
    seq_model.load_state_dict(torch.load(old_seq_model_path))

    # 还需要再加载一次旧模型，用于生成旧样本在原来的特征空间的范围
    sta_model_old = MyModelStaAE().to(myconfig.device)
    seq_model_old = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                       d_model=myconfig.d_model, nhead=myconfig.nhead,
                                       num_encoder_layers=myconfig.num_encoder_layers,
                                       num_decoder_layers=myconfig.num_decoder_layers,
                                       dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model_old.load_state_dict(torch.load(old_sta_model_path))
    seq_model_old.load_state_dict(torch.load(old_seq_model_path))

    # 加载旧样本的簇心
    with open(old_seq_centers_path, 'rb') as f:
        old_p2cer_seq_centers_f = pickle.load(f)
        old_p2cer_seq_centers = []
        for key, value in old_p2cer_seq_centers_f['center_dict'].items():
            old_p2cer_seq_centers.append(value)
        old_p2cer_seq_centers = torch.tensor(old_p2cer_seq_centers).to(myconfig.device)
        print('old_seq_center: ', old_p2cer_seq_centers.shape)
        logging.info(f'old_seq_center: {old_p2cer_seq_centers.shape}')
    with open(old_sta_centers_path, 'rb') as f:
        old_p2cer_sta_centers_f = pickle.load(f)
        old_p2cer_sta_centers = []
        for key, value in old_p2cer_sta_centers_f['center_dict'].items():
            old_p2cer_sta_centers.append(value)
        old_p2cer_sta_centers = torch.tensor(old_p2cer_sta_centers).to(myconfig.device)
        print('old sta center: ', old_p2cer_sta_centers.shape)
        logging.info('old sta center: ', old_p2cer_sta_centers.shape)

    seq_reconst_criterion = nn.MSELoss(reduction='sum')
    seq_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.seq_margin)
    # 到每个簇心的均值
    # seq_newclass_update_criterion = ContraLossUpdateNewclass(myconfig.seq_margin * myconfig.alpha_margin)

    sta_reconst_criterion = nn.MSELoss(reduction='sum')
    sta_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.sta_margin)
    # sta_newclass_update_criterion = ContraLossUpdateNewclass(myconfig.sta_margin * myconfig.alpha_margin)

    seq_optimizer = optim.AdamW(seq_model.parameters(), lr=myconfig.seq_lr)
    seq_scheduler = CosineAnnealingLR(seq_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.seq_eta_min)
    sta_optimizer = optim.AdamW(sta_model.parameters(), lr=myconfig.sta_lr)
    sta_scheduler = CosineAnnealingLR(sta_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.sta_eta_min)

    for epoch in range(myconfig.train_epoch):
        sta_model.train()
        seq_model.train()
        sta_model_old.train()
        seq_model_old.train()
        seq_contra_loss_epoch = 0
        seq_recon_loss_epoch = 0
        seq_tocenter_loss_epoch = 0
        seq_distance_newold_tocenter_epoch = 0  # 新旧样本的差异
        seq_loss_epoch = 0
        sta_contra_loss_epoch = 0
        sta_recon_loss_epoch = 0
        sta_tocenter_loss_epoch = 0
        sta_distance_newold_tocenter_epoch = 0
        sta_loss_epoch = 0

        for inputs, labels, masks, detaillabels, clusters, stas in train_loader_weighted:
            # 使用序列信息进行训练
            inputs = inputs.transpose(1, 2)
            inputs = inputs.to(myconfig.device)
            labels = labels.to(myconfig.device)
            clusters = clusters.to(myconfig.device)
            stas = stas.to(myconfig.device)

            # 选择一些就样本，约束数据偏移的不太多
            old_inputs, old_labels, _, old_detail_lables, old_clusters, old_stas = next(old_val_loader)
            old_inputs = old_inputs.transpose(1, 2).to(myconfig.device)
            old_stas = old_stas.to(myconfig.device)

            # ====================
            # 使用序列信息
            seq_optimizer.zero_grad()
            seq_optimizer.zero_grad()
            seq_recon, seq_latent = seq_model(inputs)
            seq_contra_loss = seq_contra_criterion(seq_latent, labels, clusters)
            seq_recon_loss = seq_reconst_criterion(seq_recon, inputs)
            # seq_newclass_update_loss = seq_newclass_update_criterion(seq_latent, old_p2cer_seq_centers)

            # 还要再选择一些旧样本，来约束数据偏移的不太多
            # 使用旧样本进行约束，其实目的也不完全是让簇心没有偏移，而是作为一个惩罚来约束更新的过程
            with torch.no_grad():
                _, seq_latent_oi_om = seq_model_old(old_inputs)
                _, seq_latent_oi_nm = seq_model(old_inputs)
            # cur_dists_seq = cal_eudist_sum(seq_latent_oi_om, seq_latent_oi_nm)
            cur_dists_seq_tocenter = get_uncertain_to_center_constrain_loss(old_labels, old_clusters, seq_latent_oi_om,
                                                                            seq_latent_oi_nm,
                                                                            old_p2cer_seq_centers_f['center_dict'],
                                                                            myconfig)

            # seq_contra_loss： 新增的样本内部的对比损失
            # seq_recon_loss：新增的样本的重构损失
            # seq_newclass_update_loss：新增加的样本到之前的簇心的距离
            # cur_dists_seq：旧样本距离变化的约束
            seq_loss = myconfig.seq_contra_lamda * seq_contra_loss + myconfig.seq_recon_lamda * seq_recon_loss
            seq_contra_loss_epoch += seq_contra_loss
            seq_recon_loss_epoch += seq_recon_loss
            seq_distance_newold_tocenter_epoch += cur_dists_seq_tocenter
            seq_loss_epoch += seq_loss

            seq_loss.backward()
            seq_optimizer.step()

            # ==================
            # 使用统计信息
            # 使用统计信息进行训练

            sta_optimizer.zero_grad()
            sta_latent, sta_recon = sta_model(stas)
            sta_contra_loss = sta_contra_criterion(sta_latent, labels, clusters)
            sta_recon_loss = sta_reconst_criterion(sta_recon, stas)

            with torch.no_grad():
                sta_latent_oi_om, _ = sta_model_old(old_stas)
                sta_latent_oi_nm, _ = sta_model(old_stas)
            # cur_dists_sta = cal_eudist_sum(sta_latent_oi_om, sta_latent_oi_nm)
            cur_dists_sta_tocenter = get_uncertain_to_center_constrain_loss(old_labels, old_clusters, sta_latent_oi_om,
                                                                            sta_latent_oi_nm,
                                                                            old_p2cer_sta_centers_f['center_dict'],
                                                                            myconfig)

            sta_loss = myconfig.sta_contra_lamda * sta_contra_loss + myconfig.sta_recon_lamda * sta_recon_loss
            sta_contra_loss_epoch += sta_contra_loss
            sta_recon_loss_epoch += sta_recon_loss
            sta_distance_newold_tocenter_epoch += cur_dists_sta_tocenter
            sta_loss_epoch += sta_loss

            sta_loss.backward()
            sta_optimizer.step()

        torch.save(seq_model.state_dict(), os.path.join(new_model_path, f'seq_model_{str(epoch)}.pth'))
        torch.save(sta_model.state_dict(), os.path.join(new_model_path, f'sta_model_{str(epoch)}.pth'))

        print(f'Epoch {epoch}')
        print(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist_newold_tocenter: {seq_distance_newold_tocenter_epoch}, tocenter: {seq_tocenter_loss_epoch}')
        print(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist_newold_tocenter: {sta_distance_newold_tocenter_epoch}, tocenter: {sta_tocenter_loss_epoch}')
        logging.info(f'Epoch {epoch}')
        logging.info(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist_newold_tocenter: {seq_distance_newold_tocenter_epoch}, tocenter: {seq_tocenter_loss_epoch}')
        logging.info(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist_newold_tocenter: {sta_distance_newold_tocenter_epoch}, tocenter: {sta_tocenter_loss_epoch}')

        seq_scheduler.step()
        sta_scheduler.step()


def uncertain_train_update_model_to_center_uc(myconfig, train_loader_weighted, old_val_loader, old_seq_model_path,
                                              old_sta_model_path, new_model_path,
                                              old_seq_centers_path, old_sta_centers_path):
    # uncertain_train_update_model在更新的时候是约束正常样本的新表示和原始表示的距离，这个函数约束的是新表示到簇心的距离的变化（其实约束的也是簇心的变化）

    # 在to_center的基础上修改，加了uc，在训练过程中使用旧数据，同时直接用新计算出来的簇心表示；去掉了uncertain样本到已知类簇心的约束

    # 这里加载的是certain更新过的模型
    # 加载旧模型（在此基础上更新）
    # 加载旧模型
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model.load_state_dict(torch.load(old_sta_model_path))
    seq_model.load_state_dict(torch.load(old_seq_model_path))

    # 还需要再加载一次旧模型，用于生成旧样本在原来的特征空间的范围
    sta_model_old = MyModelStaAE().to(myconfig.device)
    seq_model_old = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                       d_model=myconfig.d_model, nhead=myconfig.nhead,
                                       num_encoder_layers=myconfig.num_encoder_layers,
                                       num_decoder_layers=myconfig.num_decoder_layers,
                                       dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model_old.load_state_dict(torch.load(old_sta_model_path))
    seq_model_old.load_state_dict(torch.load(old_seq_model_path))

    # 加载旧样本的簇心
    with open(old_seq_centers_path, 'rb') as f:
        old_p2cer_seq_centers_f = pickle.load(f)
        old_p2cer_seq_centers = []
        for key, value in old_p2cer_seq_centers_f['center_dict'].items():
            old_p2cer_seq_centers.append(value)
        old_p2cer_seq_centers = torch.tensor(old_p2cer_seq_centers).to(myconfig.device)
        print('old_seq_center: ', old_p2cer_seq_centers.shape)
        logging.info(f'old_seq_center: {old_p2cer_seq_centers.shape}')
    with open(old_sta_centers_path, 'rb') as f:
        old_p2cer_sta_centers_f = pickle.load(f)
        old_p2cer_sta_centers = []
        for key, value in old_p2cer_sta_centers_f['center_dict'].items():
            old_p2cer_sta_centers.append(value)
        old_p2cer_sta_centers = torch.tensor(old_p2cer_sta_centers).to(myconfig.device)
        print('old sta center: ', old_p2cer_sta_centers.shape)
        logging.info('old sta center: ', old_p2cer_sta_centers.shape)

    seq_reconst_criterion = nn.MSELoss(reduction='sum')
    seq_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.seq_margin)
    # 到每个簇心的均值
    # seq_newclass_update_criterion = ContraLossUpdateNewclass(myconfig.seq_margin * myconfig.alpha_margin)

    sta_reconst_criterion = nn.MSELoss(reduction='sum')
    sta_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.sta_margin)
    # sta_newclass_update_criterion = ContraLossUpdateNewclass(myconfig.sta_margin * myconfig.alpha_margin)

    seq_optimizer = optim.AdamW(seq_model.parameters(), lr=myconfig.seq_lr)
    seq_scheduler = CosineAnnealingLR(seq_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.seq_eta_min)
    sta_optimizer = optim.AdamW(sta_model.parameters(), lr=myconfig.sta_lr)
    sta_scheduler = CosineAnnealingLR(sta_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.sta_eta_min)

    for epoch in range(myconfig.train_epoch):
        sta_model.train()
        seq_model.train()
        sta_model_old.train()
        seq_model_old.train()
        seq_contra_loss_epoch = 0
        seq_recon_loss_epoch = 0
        seq_tocenter_loss_epoch = 0
        seq_distance_newold_tocenter_epoch = 0  # 新旧样本的差异
        seq_loss_epoch = 0
        sta_contra_loss_epoch = 0
        sta_recon_loss_epoch = 0
        sta_tocenter_loss_epoch = 0
        sta_distance_newold_tocenter_epoch = 0
        sta_loss_epoch = 0

        for inputs, labels, masks, detaillabels, clusters, stas in train_loader_weighted:
            # 使用序列信息进行训练
            inputs = inputs.transpose(1, 2)
            inputs = inputs.to(myconfig.device)
            labels = labels.to(myconfig.device)
            clusters = clusters.to(myconfig.device)
            stas = stas.to(myconfig.device)

            # 选择一些就样本，约束数据偏移的不太多
            old_inputs, old_labels, _, old_detail_lables, old_clusters, old_stas = next(old_val_loader)
            old_inputs = old_inputs.transpose(1, 2).to(myconfig.device)
            old_stas = old_stas.to(myconfig.device)

            # ====================
            # 使用序列信息
            seq_optimizer.zero_grad()
            seq_optimizer.zero_grad()
            seq_recon, seq_latent = seq_model(inputs)
            seq_contra_loss = seq_contra_criterion(seq_latent, labels, clusters)
            seq_recon_loss = seq_reconst_criterion(seq_recon, inputs)
            # seq_newclass_update_loss = seq_newclass_update_criterion(seq_latent, old_p2cer_seq_centers)

            # 还要再选择一些旧样本，来约束数据偏移的不太多
            # 使用旧样本进行约束，其实目的也不完全是让簇心没有偏移，而是作为一个惩罚来约束更新的过程
            with torch.no_grad():
                _, seq_latent_oi_om = seq_model_old(old_inputs)
                _, seq_latent_oi_nm = seq_model(old_inputs)
            # cur_dists_seq = cal_eudist_sum(seq_latent_oi_om, seq_latent_oi_nm)
            cur_dists_seq_tocenter = get_uncertain_to_center_constrain_loss(old_labels, old_clusters, seq_latent_oi_om,
                                                                            seq_latent_oi_nm,
                                                                            old_p2cer_seq_centers_f['center_dict'],
                                                                            myconfig)

            # seq_contra_loss： 新增的样本内部的对比损失
            # seq_recon_loss：新增的样本的重构损失
            # seq_newclass_update_loss：新增加的样本到之前的簇心的距离
            # cur_dists_seq：旧样本距离变化的约束
            seq_loss = myconfig.seq_contra_lamda * seq_contra_loss + myconfig.seq_recon_lamda * seq_recon_loss \
                       + myconfig.seq_dist_lamda * cur_dists_seq_tocenter
            seq_contra_loss_epoch += seq_contra_loss
            seq_recon_loss_epoch += seq_recon_loss
            seq_distance_newold_tocenter_epoch += cur_dists_seq_tocenter
            seq_loss_epoch += seq_loss

            seq_loss.backward()
            seq_optimizer.step()

            # ==================
            # 使用统计信息
            # 使用统计信息进行训练

            sta_optimizer.zero_grad()
            sta_latent, sta_recon = sta_model(stas)
            sta_contra_loss = sta_contra_criterion(sta_latent, labels, clusters)
            sta_recon_loss = sta_reconst_criterion(sta_recon, stas)

            with torch.no_grad():
                sta_latent_oi_om, _ = sta_model_old(old_stas)
                sta_latent_oi_nm, _ = sta_model(old_stas)
            # cur_dists_sta = cal_eudist_sum(sta_latent_oi_om, sta_latent_oi_nm)
            cur_dists_sta_tocenter = get_uncertain_to_center_constrain_loss(old_labels, old_clusters, sta_latent_oi_om,
                                                                            sta_latent_oi_nm,
                                                                            old_p2cer_sta_centers_f['center_dict'],
                                                                            myconfig)

            sta_loss = myconfig.sta_contra_lamda * sta_contra_loss + myconfig.sta_recon_lamda * sta_recon_loss \
                       + myconfig.sta_dist_lamda * cur_dists_sta_tocenter
            sta_contra_loss_epoch += sta_contra_loss
            sta_recon_loss_epoch += sta_recon_loss
            sta_distance_newold_tocenter_epoch += cur_dists_sta_tocenter
            sta_loss_epoch += sta_loss

            sta_loss.backward()
            sta_optimizer.step()

        torch.save(seq_model.state_dict(), os.path.join(new_model_path, f'seq_model_{str(epoch)}.pth'))
        torch.save(sta_model.state_dict(), os.path.join(new_model_path, f'sta_model_{str(epoch)}.pth'))

        print(f'Epoch {epoch}')
        print(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist_newold_tocenter: {seq_distance_newold_tocenter_epoch}, tocenter: {seq_tocenter_loss_epoch}')
        print(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist_newold_tocenter: {sta_distance_newold_tocenter_epoch}, tocenter: {sta_tocenter_loss_epoch}')
        logging.info(f'Epoch {epoch}')
        logging.info(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist_newold_tocenter: {seq_distance_newold_tocenter_epoch}, tocenter: {seq_tocenter_loss_epoch}')
        logging.info(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist_newold_tocenter: {sta_distance_newold_tocenter_epoch}, tocenter: {sta_tocenter_loss_epoch}')

        seq_scheduler.step()
        sta_scheduler.step()


def uncertain_train_update_model_to_center(myconfig, train_loader_weighted, old_val_loader, old_seq_model_path,
                                           old_sta_model_path, new_model_path,
                                           old_seq_centers_path, old_sta_centers_path):
    # uncertain_train_update_model在更新的时候是约束正常样本的新表示和原始表示的距离，这个函数约束的是新表示到簇心的距离的变化（其实约束的也是簇心的变化）
    # 这里加载的是certain更新过的模型
    # 加载旧模型（在此基础上更新）
    # 加载旧模型
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model.load_state_dict(torch.load(old_sta_model_path))
    seq_model.load_state_dict(torch.load(old_seq_model_path))

    # 还需要再加载一次旧模型，用于生成旧样本在原来的特征空间的范围
    sta_model_old = MyModelStaAE().to(myconfig.device)
    seq_model_old = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                       d_model=myconfig.d_model, nhead=myconfig.nhead,
                                       num_encoder_layers=myconfig.num_encoder_layers,
                                       num_decoder_layers=myconfig.num_decoder_layers,
                                       dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model_old.load_state_dict(torch.load(old_sta_model_path))
    seq_model_old.load_state_dict(torch.load(old_seq_model_path))

    # 加载旧样本的簇心
    with open(old_seq_centers_path, 'rb') as f:
        old_p2cer_seq_centers_f = pickle.load(f)
        old_p2cer_seq_centers = []
        for key, value in old_p2cer_seq_centers_f['center_dict'].items():
            old_p2cer_seq_centers.append(value)
        old_p2cer_seq_centers = torch.tensor(old_p2cer_seq_centers).to(myconfig.device)
        print('old_seq_center: ', old_p2cer_seq_centers.shape)
        logging.info(f'old_seq_center: {old_p2cer_seq_centers.shape}')
    with open(old_sta_centers_path, 'rb') as f:
        old_p2cer_sta_centers_f = pickle.load(f)
        old_p2cer_sta_centers = []
        for key, value in old_p2cer_sta_centers_f['center_dict'].items():
            old_p2cer_sta_centers.append(value)
        old_p2cer_sta_centers = torch.tensor(old_p2cer_sta_centers).to(myconfig.device)
        print('old sta center: ', old_p2cer_sta_centers.shape)
        logging.info('old sta center: ', old_p2cer_sta_centers.shape)

    seq_reconst_criterion = nn.MSELoss(reduction='sum')
    seq_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.seq_margin)
    # 到每个簇心的均值
    seq_newclass_update_criterion = ContraLossUpdateNewclass(myconfig.seq_margin * myconfig.alpha_margin)

    sta_reconst_criterion = nn.MSELoss(reduction='sum')
    sta_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.sta_margin)
    sta_newclass_update_criterion = ContraLossUpdateNewclass(myconfig.sta_margin * myconfig.alpha_margin)

    seq_optimizer = optim.AdamW(seq_model.parameters(), lr=myconfig.seq_lr)
    seq_scheduler = CosineAnnealingLR(seq_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.seq_eta_min)
    sta_optimizer = optim.AdamW(sta_model.parameters(), lr=myconfig.sta_lr)
    sta_scheduler = CosineAnnealingLR(sta_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.sta_eta_min)

    for epoch in range(myconfig.train_epoch):
        sta_model.train()
        seq_model.train()
        sta_model_old.train()
        seq_model_old.train()
        seq_contra_loss_epoch = 0
        seq_recon_loss_epoch = 0
        seq_tocenter_loss_epoch = 0
        seq_distance_newold_tocenter_epoch = 0  # 新旧样本的差异
        seq_loss_epoch = 0
        sta_contra_loss_epoch = 0
        sta_recon_loss_epoch = 0
        sta_tocenter_loss_epoch = 0
        sta_distance_newold_tocenter_epoch = 0
        sta_loss_epoch = 0

        for inputs, labels, masks, detaillabels, clusters, stas in train_loader_weighted:
            # 使用序列信息进行训练
            inputs = inputs.transpose(1, 2)
            inputs = inputs.to(myconfig.device)
            labels = labels.to(myconfig.device)
            clusters = clusters.to(myconfig.device)
            stas = stas.to(myconfig.device)

            # 选择一些就样本，约束数据偏移的不太多
            old_inputs, old_labels, _, old_detail_lables, old_clusters, old_stas = next(old_val_loader)
            old_inputs = old_inputs.transpose(1, 2).to(myconfig.device)
            old_stas = old_stas.to(myconfig.device)

            # ====================
            # 使用序列信息
            seq_optimizer.zero_grad()
            seq_optimizer.zero_grad()
            seq_recon, seq_latent = seq_model(inputs)
            seq_contra_loss = seq_contra_criterion(seq_latent, labels, clusters)
            seq_recon_loss = seq_reconst_criterion(seq_recon, inputs)
            seq_newclass_update_loss = seq_newclass_update_criterion(seq_latent, old_p2cer_seq_centers)

            # 还要再选择一些旧样本，来约束数据偏移的不太多
            # 使用旧样本进行约束，其实目的也不完全是让簇心没有偏移，而是作为一个惩罚来约束更新的过程
            with torch.no_grad():
                _, seq_latent_oi_om = seq_model_old(old_inputs)
                _, seq_latent_oi_nm = seq_model(old_inputs)
            # cur_dists_seq = cal_eudist_sum(seq_latent_oi_om, seq_latent_oi_nm)
            cur_dists_seq_tocenter = get_uncertain_to_center_constrain_loss(old_labels, old_clusters, seq_latent_oi_om,
                                                                            seq_latent_oi_nm,
                                                                            old_p2cer_seq_centers_f['center_dict'],
                                                                            myconfig)

            # seq_contra_loss： 新增的样本内部的对比损失
            # seq_recon_loss：新增的样本的重构损失
            # seq_newclass_update_loss：新增加的样本到之前的簇心的距离
            # cur_dists_seq：旧样本距离变化的约束
            seq_loss = myconfig.seq_contra_lamda * seq_contra_loss + myconfig.seq_recon_lamda * seq_recon_loss \
                       + myconfig.seq_tocenter_lamda * seq_newclass_update_loss \
                       + myconfig.seq_dist_lamda * cur_dists_seq_tocenter
            seq_contra_loss_epoch += seq_contra_loss
            seq_recon_loss_epoch += seq_recon_loss
            seq_tocenter_loss_epoch += seq_newclass_update_loss
            seq_distance_newold_tocenter_epoch += cur_dists_seq_tocenter
            seq_loss_epoch += seq_loss

            seq_loss.backward()
            seq_optimizer.step()

            # ==================
            # 使用统计信息
            # 使用统计信息进行训练

            sta_optimizer.zero_grad()
            sta_latent, sta_recon = sta_model(stas)
            sta_contra_loss = sta_contra_criterion(sta_latent, labels, clusters)
            sta_recon_loss = sta_reconst_criterion(sta_recon, stas)
            sta_newclass_update_loss = sta_newclass_update_criterion(sta_latent, old_p2cer_sta_centers)

            with torch.no_grad():
                sta_latent_oi_om, _ = sta_model_old(old_stas)
                sta_latent_oi_nm, _ = sta_model(old_stas)
            # cur_dists_sta = cal_eudist_sum(sta_latent_oi_om, sta_latent_oi_nm)
            cur_dists_sta_tocenter = get_uncertain_to_center_constrain_loss(old_labels, old_clusters, sta_latent_oi_om,
                                                                            sta_latent_oi_nm,
                                                                            old_p2cer_sta_centers_f['center_dict'],
                                                                            myconfig)

            sta_loss = myconfig.sta_contra_lamda * sta_contra_loss + myconfig.sta_recon_lamda * sta_recon_loss \
                       + myconfig.sta_tocenter_lamda * sta_newclass_update_loss \
                       + myconfig.sta_dist_lamda * cur_dists_sta_tocenter
            sta_contra_loss_epoch += sta_contra_loss
            sta_recon_loss_epoch += sta_recon_loss
            sta_tocenter_loss_epoch += sta_newclass_update_loss
            sta_distance_newold_tocenter_epoch += cur_dists_sta_tocenter
            sta_loss_epoch += sta_loss

            sta_loss.backward()
            sta_optimizer.step()

        torch.save(seq_model.state_dict(), os.path.join(new_model_path, f'seq_model_{str(epoch)}.pth'))
        torch.save(sta_model.state_dict(), os.path.join(new_model_path, f'sta_model_{str(epoch)}.pth'))

        print(f'Epoch {epoch}')
        print(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist_newold_tocenter: {seq_distance_newold_tocenter_epoch}, tocenter: {seq_tocenter_loss_epoch}')
        print(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist_newold_tocenter: {sta_distance_newold_tocenter_epoch}, tocenter: {sta_tocenter_loss_epoch}')
        logging.info(f'Epoch {epoch}')
        logging.info(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist_newold_tocenter: {seq_distance_newold_tocenter_epoch}, tocenter: {seq_tocenter_loss_epoch}')
        logging.info(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist_newold_tocenter: {sta_distance_newold_tocenter_epoch}, tocenter: {sta_tocenter_loss_epoch}')

        seq_scheduler.step()
        sta_scheduler.step()


def uncertain_train_update_model(myconfig, train_loader_weighted, old_val_loader, old_seq_model_path,
                                 old_sta_model_path, new_model_path,
                                 old_seq_centers_path, old_sta_centers_path):
    # 加载旧模型
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model.load_state_dict(torch.load(old_sta_model_path))
    seq_model.load_state_dict(torch.load(old_seq_model_path))

    # 还需要再加载一次旧模型，用于生成旧样本在原来的特征空间的范围
    sta_model_old = MyModelStaAE().to(myconfig.device)
    seq_model_old = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                       d_model=myconfig.d_model, nhead=myconfig.nhead,
                                       num_encoder_layers=myconfig.num_encoder_layers,
                                       num_decoder_layers=myconfig.num_decoder_layers,
                                       dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model_old.load_state_dict(torch.load(old_sta_model_path))
    seq_model_old.load_state_dict(torch.load(old_seq_model_path))

    # 加载旧样本的簇心
    with open(old_seq_centers_path, 'rb') as f:
        old_p2cer_seq_centers_f = pickle.load(f)
        old_p2cer_seq_centers = []
        for key, value in old_p2cer_seq_centers_f['center_dict'].items():
            old_p2cer_seq_centers.append(value)
        old_p2cer_seq_centers = torch.tensor(old_p2cer_seq_centers).to(myconfig.device)
        print('old_seq_center: ', old_p2cer_seq_centers.shape)
        logging.info(f'old_seq_center: {old_p2cer_seq_centers.shape}')
    with open(old_sta_centers_path, 'rb') as f:
        old_p2cer_sta_centers_f = pickle.load(f)
        old_p2cer_sta_centers = []
        for key, value in old_p2cer_sta_centers_f['center_dict'].items():
            old_p2cer_sta_centers.append(value)
        old_p2cer_sta_centers = torch.tensor(old_p2cer_sta_centers).to(myconfig.device)
        print('old sta center: ', old_p2cer_sta_centers.shape)
        logging.info('old sta center: ', old_p2cer_sta_centers.shape)

    seq_reconst_criterion = nn.MSELoss(reduction='sum')
    seq_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.seq_margin)
    # 到每个簇心的均值
    seq_newclass_update_criterion = ContraLossUpdateNewclass(myconfig.seq_margin * myconfig.alpha_margin)
    sta_reconst_criterion = nn.MSELoss(reduction='sum')
    sta_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.sta_margin)
    sta_newclass_update_criterion = ContraLossUpdateNewclass(myconfig.sta_margin * myconfig.alpha_margin)

    seq_optimizer = optim.AdamW(seq_model.parameters(), lr=myconfig.seq_lr)
    seq_scheduler = CosineAnnealingLR(seq_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.seq_eta_min)
    sta_optimizer = optim.AdamW(sta_model.parameters(), lr=myconfig.sta_lr)
    sta_scheduler = CosineAnnealingLR(sta_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.sta_eta_min)

    for epoch in range(myconfig.train_epoch):
        sta_model.train()
        seq_model.train()
        sta_model_old.train()
        seq_model_old.train()
        seq_contra_loss_epoch = 0
        seq_recon_loss_epoch = 0
        seq_tocenter_loss_epoch = 0
        seq_distance_newold_epoch = 0
        seq_loss_epoch = 0
        sta_contra_loss_epoch = 0
        sta_recon_loss_epoch = 0
        sta_tocenter_loss_epoch = 0
        sta_distance_newold_epoch = 0
        sta_loss_epoch = 0

        for inputs, labels, masks, detaillabels, clusters, stas in train_loader_weighted:
            # 使用序列信息进行训练
            inputs = inputs.transpose(1, 2)
            inputs = inputs.to(myconfig.device)
            labels = labels.to(myconfig.device)
            clusters = clusters.to(myconfig.device)
            stas = stas.to(myconfig.device)

            # 选择一些就样本，约束数据偏移的不太多
            old_inputs, _, _, _, _, old_stas = next(old_val_loader)
            old_inputs = old_inputs.transpose(1, 2).to(myconfig.device)
            old_stas = old_stas.to(myconfig.device)

            # ====================
            # 使用序列信息
            seq_optimizer.zero_grad()
            seq_optimizer.zero_grad()
            seq_recon, seq_latent = seq_model(inputs)
            seq_contra_loss = seq_contra_criterion(seq_latent, labels, clusters)
            seq_recon_loss = seq_reconst_criterion(seq_recon, inputs)
            seq_newclass_update_loss = seq_newclass_update_criterion(seq_latent, old_p2cer_seq_centers)

            # 还要再选择一些旧样本，来约束数据偏移的不太多
            # 使用旧样本进行约束，其实目的也不完全是让簇心没有偏移，而是作为一个惩罚来约束更新的过程
            with torch.no_grad():
                _, seq_latent_oi_om = seq_model_old(old_inputs)
                _, seq_latent_oi_nm = seq_model(old_inputs)
            cur_dists_seq = cal_eudist_sum(seq_latent_oi_om, seq_latent_oi_nm)

            # seq_contra_loss： 新增的样本内部的对比损失
            # seq_recon_loss：新增的样本的重构损失
            # seq_newclass_update_loss：新增加的样本到之前的簇心的距离
            # cur_dists_seq：旧样本距离变化的约束
            seq_loss = myconfig.seq_contra_lamda * seq_contra_loss + myconfig.seq_recon_lamda * seq_recon_loss \
                       + myconfig.seq_tocenter_lamda * seq_newclass_update_loss \
                       + myconfig.seq_dist_lamda * cur_dists_seq
            seq_contra_loss_epoch += seq_contra_loss
            seq_recon_loss_epoch += seq_recon_loss
            seq_tocenter_loss_epoch += seq_newclass_update_loss
            seq_distance_newold_epoch += cur_dists_seq
            seq_loss_epoch += seq_loss

            seq_loss.backward()
            seq_optimizer.step()

            # ==================
            # 使用统计信息
            # 使用统计信息进行训练

            sta_optimizer.zero_grad()
            sta_latent, sta_recon = sta_model(stas)
            sta_contra_loss = sta_contra_criterion(sta_latent, labels, clusters)
            sta_recon_loss = sta_reconst_criterion(sta_recon, stas)
            sta_newclass_update_loss = sta_newclass_update_criterion(sta_latent, old_p2cer_sta_centers)

            with torch.no_grad():
                sta_latent_oi_om, _ = sta_model_old(old_stas)
                sta_latent_oi_nm, _ = sta_model(old_stas)
            cur_dists_sta = cal_eudist_sum(sta_latent_oi_om, sta_latent_oi_nm)

            sta_loss = myconfig.sta_contra_lamda * sta_contra_loss + myconfig.sta_recon_lamda * sta_recon_loss \
                       + myconfig.sta_tocenter_lamda * sta_newclass_update_loss \
                       + myconfig.sta_dist_lamda * cur_dists_sta
            sta_contra_loss_epoch += sta_contra_loss
            sta_recon_loss_epoch += sta_recon_loss
            sta_tocenter_loss_epoch += sta_newclass_update_loss
            sta_distance_newold_epoch += cur_dists_sta
            sta_loss_epoch += sta_loss

            sta_loss.backward()
            sta_optimizer.step()

        torch.save(seq_model.state_dict(), os.path.join(new_model_path, f'seq_model_{str(epoch)}.pth'))
        torch.save(sta_model.state_dict(), os.path.join(new_model_path, f'sta_model_{str(epoch)}.pth'))

        print(f'Epoch {epoch}')
        print(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist: {seq_distance_newold_epoch}, tocenter: {seq_tocenter_loss_epoch}')
        print(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist: {sta_distance_newold_epoch}, tocenter: {sta_tocenter_loss_epoch}')
        logging.info(f'Epoch {epoch}')
        logging.info(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist: {seq_distance_newold_epoch}, tocenter: {seq_tocenter_loss_epoch}')
        logging.info(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist: {sta_distance_newold_epoch}, tocenter: {sta_tocenter_loss_epoch}')

        seq_scheduler.step()
        sta_scheduler.step()


def uncertain_train_update_model_uc(myconfig, train_loader_weighted, old_val_loader, old_seq_model_path,
                                    old_sta_model_path, new_model_path,
                                    old_seq_centers_path, old_sta_centers_path):
    # 加载旧模型
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model.load_state_dict(torch.load(old_sta_model_path))
    seq_model.load_state_dict(torch.load(old_seq_model_path))

    # 还需要再加载一次旧模型，用于生成旧样本在原来的特征空间的范围
    sta_model_old = MyModelStaAE().to(myconfig.device)
    seq_model_old = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                       d_model=myconfig.d_model, nhead=myconfig.nhead,
                                       num_encoder_layers=myconfig.num_encoder_layers,
                                       num_decoder_layers=myconfig.num_decoder_layers,
                                       dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model_old.load_state_dict(torch.load(old_sta_model_path))
    seq_model_old.load_state_dict(torch.load(old_seq_model_path))

    # 加载旧样本的簇心
    with open(old_seq_centers_path, 'rb') as f:
        old_p2cer_seq_centers_f = pickle.load(f)
        old_p2cer_seq_centers = []
        for key, value in old_p2cer_seq_centers_f['center_dict'].items():
            old_p2cer_seq_centers.append(value)
        old_p2cer_seq_centers = torch.tensor(old_p2cer_seq_centers).to(myconfig.device)
        print('old_seq_center: ', old_p2cer_seq_centers.shape)
        logging.info(f'old_seq_center: {old_p2cer_seq_centers.shape}')
    with open(old_sta_centers_path, 'rb') as f:
        old_p2cer_sta_centers_f = pickle.load(f)
        old_p2cer_sta_centers = []
        for key, value in old_p2cer_sta_centers_f['center_dict'].items():
            old_p2cer_sta_centers.append(value)
        old_p2cer_sta_centers = torch.tensor(old_p2cer_sta_centers).to(myconfig.device)
        print('old sta center: ', old_p2cer_sta_centers.shape)
        logging.info('old sta center: ', old_p2cer_sta_centers.shape)

    seq_reconst_criterion = nn.MSELoss(reduction='sum')
    seq_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.seq_margin)
    # 到每个簇心的均值
    # seq_newclass_update_criterion = ContraLossUpdateNewclass(myconfig.seq_margin * myconfig.alpha_margin)
    sta_reconst_criterion = nn.MSELoss(reduction='sum')
    sta_contra_criterion = ContraLossEucNewM2(myconfig, myconfig.sta_margin)
    # sta_newclass_update_criterion = ContraLossUpdateNewclass(myconfig.sta_margin * myconfig.alpha_margin)

    seq_optimizer = optim.AdamW(seq_model.parameters(), lr=myconfig.seq_lr)
    seq_scheduler = CosineAnnealingLR(seq_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.seq_eta_min)
    sta_optimizer = optim.AdamW(sta_model.parameters(), lr=myconfig.sta_lr)
    sta_scheduler = CosineAnnealingLR(sta_optimizer, T_max=myconfig.train_epoch, eta_min=myconfig.sta_eta_min)

    for epoch in range(myconfig.train_epoch):
        sta_model.train()
        seq_model.train()
        sta_model_old.train()
        seq_model_old.train()
        seq_contra_loss_epoch = 0
        seq_recon_loss_epoch = 0
        seq_tocenter_loss_epoch = 0
        seq_distance_newold_epoch = 0
        seq_loss_epoch = 0
        sta_contra_loss_epoch = 0
        sta_recon_loss_epoch = 0
        sta_tocenter_loss_epoch = 0
        sta_distance_newold_epoch = 0
        sta_loss_epoch = 0

        for inputs, labels, masks, detaillabels, clusters, stas in train_loader_weighted:
            # 使用序列信息进行训练
            inputs = inputs.transpose(1, 2)
            inputs = inputs.to(myconfig.device)
            labels = labels.to(myconfig.device)
            clusters = clusters.to(myconfig.device)
            stas = stas.to(myconfig.device)

            # 选择一些就样本，约束数据偏移的不太多
            old_inputs, _, _, _, _, old_stas = next(old_val_loader)
            old_inputs = old_inputs.transpose(1, 2).to(myconfig.device)
            old_stas = old_stas.to(myconfig.device)

            # ====================
            # 使用序列信息
            seq_optimizer.zero_grad()
            seq_optimizer.zero_grad()
            seq_recon, seq_latent = seq_model(inputs)
            seq_contra_loss = seq_contra_criterion(seq_latent, labels, clusters)
            seq_recon_loss = seq_reconst_criterion(seq_recon, inputs)
            # seq_newclass_update_loss = seq_newclass_update_criterion(seq_latent, old_p2cer_seq_centers)

            # 还要再选择一些旧样本，来约束数据偏移的不太多
            # 使用旧样本进行约束，其实目的也不完全是让簇心没有偏移，而是作为一个惩罚来约束更新的过程
            with torch.no_grad():
                _, seq_latent_oi_om = seq_model_old(old_inputs)
                _, seq_latent_oi_nm = seq_model(old_inputs)
            cur_dists_seq = cal_eudist_sum(seq_latent_oi_om, seq_latent_oi_nm)

            # seq_contra_loss： 新增的样本内部的对比损失
            # seq_recon_loss：新增的样本的重构损失
            # seq_newclass_update_loss：新增加的样本到之前的簇心的距离
            # cur_dists_seq：旧样本距离变化的约束
            seq_loss = myconfig.seq_contra_lamda * seq_contra_loss + myconfig.seq_recon_lamda * seq_recon_loss \
                       + myconfig.seq_dist_lamda * cur_dists_seq
            seq_contra_loss_epoch += seq_contra_loss
            seq_recon_loss_epoch += seq_recon_loss
            # seq_tocenter_loss_epoch += seq_newclass_update_loss
            seq_distance_newold_epoch += cur_dists_seq
            seq_loss_epoch += seq_loss

            seq_loss.backward()
            seq_optimizer.step()

            # ==================
            # 使用统计信息
            # 使用统计信息进行训练

            sta_optimizer.zero_grad()
            sta_latent, sta_recon = sta_model(stas)
            sta_contra_loss = sta_contra_criterion(sta_latent, labels, clusters)
            sta_recon_loss = sta_reconst_criterion(sta_recon, stas)
            # sta_newclass_update_loss = sta_newclass_update_criterion(sta_latent, old_p2cer_sta_centers)

            with torch.no_grad():
                sta_latent_oi_om, _ = sta_model_old(old_stas)
                sta_latent_oi_nm, _ = sta_model(old_stas)
            cur_dists_sta = cal_eudist_sum(sta_latent_oi_om, sta_latent_oi_nm)

            sta_loss = myconfig.sta_contra_lamda * sta_contra_loss + myconfig.sta_recon_lamda * sta_recon_loss \
                       + myconfig.sta_dist_lamda * cur_dists_sta
            sta_contra_loss_epoch += sta_contra_loss
            sta_recon_loss_epoch += sta_recon_loss
            # sta_tocenter_loss_epoch += sta_newclass_update_loss
            sta_distance_newold_epoch += cur_dists_sta
            sta_loss_epoch += sta_loss

            sta_loss.backward()
            sta_optimizer.step()

        torch.save(seq_model.state_dict(), os.path.join(new_model_path, f'seq_model_{str(epoch)}.pth'))
        torch.save(sta_model.state_dict(), os.path.join(new_model_path, f'sta_model_{str(epoch)}.pth'))

        print(f'Epoch {epoch}')
        print(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist: {seq_distance_newold_epoch}, tocenter: {seq_tocenter_loss_epoch}')
        print(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist: {sta_distance_newold_epoch}, tocenter: {sta_tocenter_loss_epoch}')
        logging.info(f'Epoch {epoch}')
        logging.info(
            f'Seq Loss: {seq_loss_epoch}, contra: {seq_contra_loss_epoch}, recon: {seq_recon_loss_epoch}, dist: {seq_distance_newold_epoch}, tocenter: {seq_tocenter_loss_epoch}')
        logging.info(
            f'Sta Loss: {sta_loss_epoch}, contra: {sta_contra_loss_epoch}, recon: {sta_recon_loss_epoch}, dist: {sta_distance_newold_epoch}, tocenter: {sta_tocenter_loss_epoch}')

        seq_scheduler.step()
        sta_scheduler.step()


def get_uncertain_related(myconfig, train_loader_noweight, seq_model_path, sta_model_path, save_data_root):
    sta_model = MyModelStaAE().to(myconfig.device)
    seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                   d_model=myconfig.d_model, nhead=myconfig.nhead,
                                   num_encoder_layers=myconfig.num_encoder_layers,
                                   num_decoder_layers=myconfig.num_decoder_layers,
                                   dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    sta_model.load_state_dict(torch.load(sta_model_path))
    seq_model.load_state_dict(torch.load(seq_model_path))

    # # 获得新模型产生的潜在特征
    print('Get latent feature')
    logging.info('Get latent feature')
    get_seqmid_info_of_train_transformer_sta(seq_model, train_loader_noweight, myconfig, save_data_root)
    get_stamid_info_of_train_ae_sta(sta_model, train_loader_noweight, myconfig, save_data_root)

    # 获得（新模型，新训练数据）每个簇的簇心，将重构损失、到簇心的距离等拟合到95%的数据
    print('Fit distribution with 95% data')
    logging.info('Fit distribution with 95% data')
    get_sta_info_of_train_95_stadata(save_data_root)  # sta
    get_sta_info_of_train_95(save_data_root)  # seq

    # 使用所有数据（新模型，新训练数据）拟合分布
    print('Fit distribution with all data')
    logging.info('Fit distribution with all data')
    get_sta_info_of_train_stadata(save_data_root)
    get_sta_info_of_train(save_data_root)


def combine_two_distribution_uncertain_update(old_save_data_root, new_save_data_root, updated_save_data_root):
    # 不光loc发生变化了，簇心也发生变化了呀
    pass


def combine_two_distribution_uncertain(old_save_data_root, new_save_data_root, updated_save_data_root):
    # 这个时候类没有重合，就直接将两种分布放在一起

    # 合并簇心
    old_seq_centerinfo_path = os.path.join(old_save_data_root, 'train_sta_info.pickle')
    old_sta_centerinfo_path = os.path.join(old_save_data_root, 'train_sta_info_stadata.pickle')
    new_seq_centerinfo_path = os.path.join(new_save_data_root, 'train_sta_info.pickle')
    new_sta_centerinfo_path = os.path.join(new_save_data_root, 'train_sta_info_stadata.pickle')
    with open(old_seq_centerinfo_path, 'rb') as f:
        old_seq_data = pickle.load(f)
        old_seqcenter_data = old_seq_data['center_dict']
    with open(new_seq_centerinfo_path, 'rb') as f:
        new_seq_data = pickle.load(f)
        new_seqcenter_data = new_seq_data['center_dict']
    with open(old_sta_centerinfo_path, 'rb') as f:
        old_sta_data = pickle.load(f)
        old_stacenter_data = old_sta_data['center_dict']
    with open(new_sta_centerinfo_path, 'rb') as f:
        new_sta_data = pickle.load(f)
        new_stacenter_data = new_sta_data['center_dict']

    old_seqcenter_data.update(new_seqcenter_data)
    old_stacenter_data.update(new_stacenter_data)
    updated_seq_centers_dict = {'center_dict': old_seqcenter_data}
    updated_sta_centers_dict = {'center_dict': old_stacenter_data}
    with open(os.path.join(updated_save_data_root, 'train_sta_info.pickle'), 'wb') as f:
        pickle.dump(updated_seq_centers_dict, f)
    with open(os.path.join(updated_save_data_root, 'train_sta_info_stadata.pickle'), 'wb') as f:
        pickle.dump(updated_sta_centers_dict, f)

    # 合并到簇心距离的均值和方差，以及重构损失的均值和方差
    old_seq_distrinfo_path = os.path.join(old_save_data_root, 'train_distri_para_info.pickle')
    new_seq_distrinfo_path = os.path.join(new_save_data_root, 'train_distri_para_info.pickle')
    old_sta_distrinfo_path = os.path.join(old_save_data_root, 'train_distri_para_info_stadata.pickle')
    new_sta_distrinfo_path = os.path.join(new_save_data_root, 'train_distri_para_info_stadata.pickle')
    with open(old_seq_distrinfo_path, 'rb') as f:
        old_seq_distri = pickle.load(f)
        old_seq_distri_dist = old_seq_distri['dist_distri']
        old_seq_distri_recon = old_seq_distri['reclos_distri']
    with open(new_seq_distrinfo_path, 'rb') as f:
        new_seq_distri = pickle.load(f)
        new_seq_distri_dist = new_seq_distri['dist_distri']
        new_seq_distri_recon = new_seq_distri['reclos_distri']
    with open(old_sta_distrinfo_path, 'rb') as f:
        old_sta_distri = pickle.load(f)
        old_sta_distri_dist = old_sta_distri['dist_distri']
        old_sta_distri_recon = old_sta_distri['reclos_distri']
    with open(new_sta_distrinfo_path, 'rb') as f:
        new_sta_distri = pickle.load(f)
        new_sta_distri_dist = new_sta_distri['dist_distri']
        new_sta_distri_recon = new_sta_distri['reclos_distri']
    old_seq_distri_dist.update(new_seq_distri_dist)
    old_seq_distri_recon.update(new_seq_distri_recon)
    old_sta_distri_dist.update(new_sta_distri_dist)
    old_sta_distri_recon.update(new_sta_distri_recon)
    # 保存到相应文件中
    update_seq_distri_dict = {'dist_distri': old_seq_distri_dist, 'reclos_distri': old_seq_distri_recon}
    update_sta_distri_dict = {'dist_distri': old_sta_distri_dist, 'reclos_distri': old_sta_distri_recon}
    with open(os.path.join(updated_save_data_root, 'train_distri_para_info.pickle'), 'wb') as f:
        pickle.dump(update_seq_distri_dict, f)
    with open(os.path.join(updated_save_data_root, 'train_distri_para_info_stadata.pickle'), 'wb') as f:
        pickle.dump(update_sta_distri_dict, f)


def validate_drift(data_loader, old_seq_model_path, old_sta_model_path, new_seq_model_path, new_sta_model_path,
                   myconfig, driftvalidate_root):
    # 验证在新旧模型上，样本偏移了多少
    old_sta_model = MyModelStaAE().to(myconfig.device)
    old_seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                       d_model=myconfig.d_model, nhead=myconfig.nhead,
                                       num_encoder_layers=myconfig.num_encoder_layers,
                                       num_decoder_layers=myconfig.num_decoder_layers,
                                       dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    old_sta_model.load_state_dict(torch.load(old_sta_model_path))
    old_seq_model.load_state_dict(torch.load(old_seq_model_path))

    new_sta_model = MyModelStaAE().to(myconfig.device)
    new_seq_model = MyModelTransformer(seq_len=myconfig.sequence_length, feature_size=myconfig.feature_size,
                                       d_model=myconfig.d_model, nhead=myconfig.nhead,
                                       num_encoder_layers=myconfig.num_encoder_layers,
                                       num_decoder_layers=myconfig.num_decoder_layers,
                                       dim_feedforward=myconfig.dim_feedforward, dropout=myconfig.dropout).to(
        myconfig.device)
    new_sta_model.load_state_dict(torch.load(new_sta_model_path))
    new_seq_model.load_state_dict(torch.load(new_seq_model_path))

    old_driftvalidate_save_data_root = os.path.join(driftvalidate_root, 'old')
    new_driftvalidate_save_data_root = os.path.join(driftvalidate_root, 'new')

    print('[INFO] Generate seq mid info')
    logging.info('[INFO] Generate seq mid info')
    get_seqmid_info_of_train_transformer_sta(old_seq_model, data_loader, myconfig, old_driftvalidate_save_data_root)
    get_seqmid_info_of_train_transformer_sta(new_seq_model, data_loader, myconfig, new_driftvalidate_save_data_root)
    # get_sta_info_of_train_95(old_driftvalidate_save_data_root)  # seq
    # get_sta_info_of_train_95(new_driftvalidate_save_data_root)
    # 修改成使用all 7.28
    get_sta_info_of_train(old_driftvalidate_save_data_root)  # seq
    get_sta_info_of_train(new_driftvalidate_save_data_root)

    print('[INFO] Generate sta mid info')
    logging.info('[INFO] Generate sta mid info')
    get_stamid_info_of_train_ae_sta(old_sta_model, data_loader, myconfig, old_driftvalidate_save_data_root)
    get_stamid_info_of_train_ae_sta(new_sta_model, data_loader, myconfig, new_driftvalidate_save_data_root)
    # get_sta_info_of_train_95_stadata(old_driftvalidate_save_data_root)  # sta
    # get_sta_info_of_train_95_stadata(new_driftvalidate_save_data_root)
    # 修改成使用all 7.28
    get_sta_info_of_train_stadata(old_driftvalidate_save_data_root)  # sta
    get_sta_info_of_train_stadata(new_driftvalidate_save_data_root)

    print('[INFO] Calculate the center drift')
    logging.info('[INFO] Calculate the center drift')
    cal_dist_for_validate(os.path.join(old_driftvalidate_save_data_root, 'train_mid_info.pickle'),
                          os.path.join(new_driftvalidate_save_data_root, 'train_mid_info.pickle'),
                          os.path.join(driftvalidate_root, 'seq_dist.pickle'),
                          os.path.join(driftvalidate_root, 'res.log'))
    cal_dist_for_validate(os.path.join(old_driftvalidate_save_data_root, 'train_stamid_info.pickle'),
                          os.path.join(new_driftvalidate_save_data_root, 'train_stamid_info.pickle'),
                          os.path.join(driftvalidate_root, 'sta_dist.pickle'),
                          os.path.join(driftvalidate_root, 'res.log'))

    print('[INFO] Calculate the distribution drift')
    logging.info('[INFO] Calculate the distribution drift')
    cal_center_distri_drift(os.path.join(old_driftvalidate_save_data_root, 'train_sta_info.pickle'),
                            os.path.join(old_driftvalidate_save_data_root, 'train_distri_para_info.pickle'),
                            os.path.join(new_driftvalidate_save_data_root, 'train_sta_info.pickle'),
                            os.path.join(new_driftvalidate_save_data_root, 'train_distri_para_info.pickle'),
                            os.path.join(driftvalidate_root, 'seq_distr_drift.pickle'),
                            os.path.join(driftvalidate_root, 'res.log'))
    cal_center_distri_drift(os.path.join(old_driftvalidate_save_data_root, 'train_sta_info_stadata.pickle'),
                            os.path.join(old_driftvalidate_save_data_root, 'train_distri_para_info_stadata.pickle'),
                            os.path.join(new_driftvalidate_save_data_root, 'train_sta_info_stadata.pickle'),
                            os.path.join(new_driftvalidate_save_data_root, 'train_distri_para_info_stadata.pickle'),
                            os.path.join(driftvalidate_root, 'sta_distr_drift.pickle'),
                            os.path.join(driftvalidate_root, 'res.log'))


def euclidean_distance(a, b):
    import numpy as np
    return np.sqrt(np.sum((a - b) ** 2))


def cal_dist_for_validate(new_path, old_path, final_save_path, res_path):
    """计算由两个模型生成的中间特征之间的距离，用于验证新的表示发生了多少偏移"""
    new_data = pd.read_pickle(new_path)
    old_data = pd.read_pickle(old_path)
    new_data.columns = ['new_' + str(i) for i in new_data.columns]
    old_data.columns = ['old_' + str(i) for i in old_data.columns]

    data = pd.concat([new_data.reset_index(drop=True), old_data.reset_index(drop=True)], axis=1)
    data['feature_dist'] = data.apply(lambda row: euclidean_distance(row['new_feature'], row['old_feature']), axis=1)
    data['recon_dist'] = data.apply(lambda row: abs(row['new_recon_loss'] - row['old_recon_loss']), axis=1)

    data.to_pickle(final_save_path)

    feature_dist = data['feature_dist'].values
    recon_dist = data['recon_dist'].values
    feature_dist = sorted(feature_dist)
    recon_dist = sorted(recon_dist)
    with open(res_path, 'a') as f:
        f.write(f'{final_save_path}\n')
        f.write(
            f'feature dist: avg: {sum(feature_dist) / len(feature_dist)}, pos_0-1: {feature_dist[0]}, {feature_dist[int(len(feature_dist) * 0.25)]},'
            f'{feature_dist[int(len(feature_dist) * 0.5)]}, {feature_dist[int(len(feature_dist) * 0.75)]}, {feature_dist[-1]}\n')
        f.write(
            f'recon dist: avg: {sum(recon_dist) / len(recon_dist)}, pos_0-1: {recon_dist[0]}, {recon_dist[int(len(recon_dist) * 0.25)]},'
            f'{recon_dist[int(len(recon_dist) * 0.5)]}, {recon_dist[int(len(recon_dist) * 0.75)]}, {recon_dist[-1]}\n')


def cal_distri_dr(distri1, distri2):
    res_distri = {}
    for key, value in distri1.items():
        cur1 = distri1[key]
        cur2 = distri2[key]
        res_distri[key] = {'loc_drift': cur2['loc'] - cur1['loc'], 'scale_drift': cur2['scale'] - cur1['scale']}
    return res_distri


def cal_center_distri_drift(old_center_path, old_distri_path, new_center_path, new_distri_path, drift_save_path,
                            res_log_path):
    """计算簇心和分布的偏移"""

    # cal_center_distri_drift(os.path.join(old_driftvalidate_save_data_root, 'train_sta_info.pickle'),
    #                         os.path.join(old_driftvalidate_save_data_root, 'train_distri_para_info_95.pickle'),
    #                         os.path.join(new_driftvalidate_save_data_root, 'train_sta_info.pickle'),
    #                         os.path.join(new_driftvalidate_save_data_root, 'train_distri_para_info_95.pickle'),
    #                         os.path.join(driftvalidate_root, 'seq_distr_drift.pickle'),
    #                         os.path.join(driftvalidate_root, 'res.log'))

    import numpy as np
    with open(new_center_path, 'rb') as f:
        new_center = pickle.load(f)['center_dict']
    with open(old_center_path, 'rb') as f:
        old_center = pickle.load(f)['center_dict']
    center_dist_dict = {}
    for key, value in new_center.items():
        new_ = new_center[key]
        old_ = old_center[key]
        dist = np.sqrt(sum((new_ - old_) ** 2))
        center_dist_dict[key] = dist

    with open(new_distri_path, 'rb') as f:
        new_distr = pickle.load(f)
    with open(old_distri_path, 'rb') as f:
        old_distr = pickle.load(f)

    dist_distri_drift = cal_distri_dr(old_distr['dist_distri'], new_distr['dist_distri'])
    recon_distri_drift = cal_distri_dr(old_distr['reclos_distri'], new_distr['reclos_distri'])
    distri_drift = {'dist': dist_distri_drift, 'recon': recon_distri_drift}

    with open(drift_save_path, 'wb') as f:
        pickle.dump(distri_drift, f)

    with open(res_log_path, 'a') as f:
        f.write(f'\n{drift_save_path}\n')
        f.write(f'{distri_drift}')


def split_get_uncertain_data(seq_latent_path, attack_seq_latent_path, normal_seq_latent_path,
                             sta_latent_path, attack_sta_latent_path, normal_sta_latent_path,
                             regenerate_cocertain, consider_unmatch, cocertain_row, used_uncertain_row,
                             ori_file_root, result_root,
                             normal_seq_path, normal_sta_path, normal_cocertain_path, normal_files,
                             normal_dist_seq_path, normal_dist_sta_path, certain_normal_path, uncertain_normal_path,
                             attack_seq_path, attack_sta_path, attack_cocertain_path, attack_files,
                             attack_dist_seq_path, attack_dist_sta_path, certain_attack_path, uncertain_attack_path
                             ):
    """
    将测试数据中的未知数据拆分出来
    """
    # 这里是把正常数据和攻击数据拆分开了
    split_latent(seq_latent_path, attack_seq_latent_path, normal_seq_latent_path)
    split_latent(sta_latent_path, attack_sta_latent_path, normal_sta_latent_path)
    # 这里才开始拆分uncertain
    generate_uncertain_data(regenerate_cocertain, consider_unmatch, cocertain_row, used_uncertain_row, normal_seq_path,
                            normal_sta_path,
                            normal_cocertain_path,
                            'normal',
                            normal_files, ori_file_root,
                            normal_dist_seq_path,
                            normal_dist_sta_path, normal_seq_latent_path,
                            normal_sta_latent_path,
                            result_root, certain_normal_path, uncertain_normal_path)
    generate_uncertain_data(regenerate_cocertain, consider_unmatch, cocertain_row, used_uncertain_row, attack_seq_path,
                            attack_sta_path,
                            attack_cocertain_path,
                            'attack',
                            attack_files, ori_file_root,
                            attack_dist_seq_path,
                            attack_dist_sta_path, attack_seq_latent_path,
                            attack_sta_latent_path,
                            result_root, certain_attack_path, uncertain_attack_path)


def save_certain_uncertain_with_class_for_train(certain_normal_path, certain_attack_path, new_certain_data_root,
                                                uncertain_normal_path, uncertain_attack_path, new_uncertain_data_root):
    """前面对确定和不确定的样本的类别进行了判定，将数据和判断得到的标签保存起来，用于后续更新模型"""
    fine_split_certain(certain_normal_path, new_certain_data_root, 'normal')
    fine_split_certain(certain_attack_path, new_certain_data_root, 'attack')
    fine_split_uncertain(uncertain_normal_path, new_uncertain_data_root, 'normal')
    fine_split_uncertain(uncertain_attack_path, new_uncertain_data_root, 'attack')

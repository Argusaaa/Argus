#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import argparse
import torch
import logging
from datetime import datetime
import importlib
import os
from torch.utils.data import WeightedRandomSampler, DataLoader

sys.path.append('/home/driftDetect/newcode0625')
sys.path.append('D:\PycharmProject\DriftDetection/newcode0625')

DatasetConfig1 = importlib.import_module('dataconfig_ids2018').DatasetConfig1
DatasetConfig2 = importlib.import_module('dataconfig_ids2018').DatasetConfig2
label_cluster_normal_for_test = importlib.import_module('myutil').label_cluster_normal_for_test
label_cluster_normal_for_train = importlib.import_module('myutil').label_cluster_normal_for_train
StaSeqTrafficNormalizedDataset = importlib.import_module('myutil').StaSeqTrafficNormalizedDataset
get_sample_weights = importlib.import_module('myutil').get_sample_weights
train_model = importlib.import_module('myutil').train_model
eval_model_stage1 = importlib.import_module('myutil').eval_model_stage1
label_cluster_attack_ids = importlib.import_module('myutil').label_cluster_attack_ids
label_cluster_attack_for_train = importlib.import_module('myutil').label_cluster_attack_for_train
label_cluster_attack_for_test = importlib.import_module('myutil').label_cluster_attack_for_test


parser = argparse.ArgumentParser(description='step1 model train', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--config', type=str, default='config1')
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--cdr', type=float, default=0.05)
parser.add_argument('--a_k', type=int, default=1)    # 攻击流量聚类的簇数
parser.add_argument('--a_cdr', type=float, default=0.05)   # 攻击流量聚类的cdr
parser.add_argument('--prefix_name', type=str, default='kmeans0708')
parser.add_argument('--seq_margin', type=float, default=10)
parser.add_argument('--sta_margin', type=float, default=10)
parser.add_argument('--re_train', type=str, default='True')

parser.add_argument('--sta_model_num', type=str, default='300')
parser.add_argument('--seq_model_num', type=str, default='300')
parser.add_argument('--alpha_seq_contra', type=float, default=1.0)
parser.add_argument('--alpha_seq_recon', type=float, default=1.0)
parser.add_argument('--alpha_sta_contra', type=float, default=1.0)
parser.add_argument('--alpha_sta_recon', type=float, default=1.0)
parser.add_argument('--execute_folder_name', type=str, default='7_ids_script_new')

parser.add_argument('--single_eval_p4', type=str, default='no')  # 是否是单独评估p4的结果  0801补充
parser.add_argument('--single_eval_p2p4', type=str, default='no')  # 0815，对p2和p4中的内容进行了一些修改，p3的攻击加入到p2中，p4中的正常流量替换成ids2018中的内容

args = parser.parse_args()


class InitTrainConfig:
    def __init__(self, used_config, k=1, cdr=0.05, prefix_name='kmeans0708', seq_margin=10, sta_margin=10,
                 seq_model_num='300', sta_model_num='300', alpha_seq_contra=1.0, alpha_seq_recon=1.0,
                 alpha_sta_contra=1.0, alpha_sta_recon=1.0, execute_folder_name='none', a_k=1, a_cdr=0.05):
        self.used_config = used_config
        self.prefix_name = prefix_name

        self.alpha_seq_contra = alpha_seq_contra
        self.alpha_seq_recon = alpha_seq_recon
        self.alpha_sta_contra = alpha_sta_contra
        self.alpha_sta_recon = alpha_sta_recon

        self.sequence_length = 50  # 将序列截断或增大成多少
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1024
        self.train_epoch = 300
        self.seq_lr = 0.0015
        self.sta_lr = 0.0015
        self.seq_eta_min = 1e-05
        self.sta_eta_min = 1e-05

        # kmeans参数
        self.k = k
        self.class_drop_ratio = cdr  # 如果一个类别的样本数少于总数/类别数的多少，则不考虑
        self.a_k = a_k
        self.a_class_drop_ratio = a_cdr

        # 对比损失的参数
        self.seq_margin = seq_margin
        self.sta_margin = sta_margin
        self.temperature = 1  # 在欧式距离的时候没有用

        # transformer相关的参数
        self.feature_size = 2
        self.d_model = 16
        self.nhead = 2
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.dim_feedforward = 32
        self.dropout = 0.1

        logging.basicConfig(level=logging.DEBUG,
                            filename=f'log/{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}'
                                     f'_ak{self.a_k}_adm{self.a_class_drop_ratio}_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted'
                                     f'_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}.log',
                            format='%(message)s')
        logging.info('\n\n ===========================================================')
        logging.info('====== ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ======')
        logging.info('===========================================================')
        logging.info(
            f'Config info: seq margin: {self.seq_margin}, sta margin: {self.sta_margin}, length: {self.sequence_length}')
        logging.info(f'Normal Cluster info: k {self.k}, cdr: {self.class_drop_ratio}')
        logging.info(f'Attack cluster info: k: {self.a_k}, cdr: {self.a_class_drop_ratio}')
        logging.info(f'current file: {os.path.basename(__file__)}\n')

        # 导入数据配置文件
        if self.used_config == 'config1':
            self.dataconfig = DatasetConfig1()
        elif self.used_config == 'config2':
            self.dataconfig = DatasetConfig2()
        else:
            print(f'[ERROR] Config input unmatch, input: {args.config}')
            logging.info(f'[ERROR] Config input unmatch, input: {args.config}')

        # 文件存储路径
        self.train_normal_root = '/home/driftDetect/0_data/1_cicids2018/3_nfstream_data/nfstream_pickle/normal'
        self.train_attack_root = '/home/driftDetect/0_data/1_cicids2018/3_nfstream_data/nfstream_pickle/reorganized_attack'
        self.test_normal_root = self.train_normal_root
        self.test_attack_root = self.train_attack_root

        self.train_coarse_root_normal = f'/home/driftDetect/0_data/1_cicids2018/4_ids_all_run_data/' \
                                        f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}' \
                                        f'_ak{self.a_k}_adm{self.a_class_drop_ratio}_weighted'
        if not os.path.exists(self.train_coarse_root_normal):
            os.makedirs(self.train_coarse_root_normal)
            print(f'[INFO] Make dirs: {self.train_coarse_root_normal}')
        self.train_coarse_root_attack = self.train_coarse_root_normal
        self.test_coarse_root_normal = self.train_coarse_root_normal
        self.test_coarse_root_attack = self.train_coarse_root_normal

        self.train_normal_files = self.dataconfig.part1_normal_files
        self.train_attack_files = self.dataconfig.part1_attack_files
        self.test_normal_files_p1val = self.dataconfig.part1_normal_files_val
        self.test_attack_files_p1val = self.dataconfig.part1_attack_files_val

        # 2024.8.15 补充p2和p4的代码，新写的
        self.test_normal_files_p2new = self.dataconfig.part2_normal_files_new
        self.test_normal_files_p2valnew = self.dataconfig.part2_normal_files_val_new
        self.test_attack_files_p2new = self.dataconfig.part2_attack_files_new
        self.test_attack_files_p2valnew = self.dataconfig.part2_attack_files_val_new

        self.test_normal_files_p4new = self.dataconfig.part4_normal_files_new
        self.test_normal_files_p4valnew = self.dataconfig.part4_normal_files_val_new
        self.test_attack_files_p4new = self.dataconfig.part4_attack_files_new
        self.test_attack_files_p4valnew = self.dataconfig.part4_attack_files_val_new


        self.class_dict = self.dataconfig.class_dict
        self.detail_class_dict = self.dataconfig.detail_class_dict
        # 增加聚类新产生的文件的标签映射
        self.class_dict[f'normal_kmeans_k{self.k}_dm{self.class_drop_ratio}.pickle'] = 0
        self.detail_class_dict[f'normal_kmeans_k{self.k}_dm{self.class_drop_ratio}.pickle'] = 0
        self.class_dict[f'attack_kmeans_k{self.a_k}_dm{self.a_class_drop_ratio}.pickle'] = 1
        self.detail_class_dict[f'attack_kmeans_k{self.a_k}_dm{self.a_class_drop_ratio}.pickle'] = 0

        self.model_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/model/part1/' \
                          f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}_' \
                          f'qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}'

        if not os.path.exists(self.model_root):
            os.makedirs(self.model_root)
        self.used_seq_model = f'seq_model_{seq_model_num}.pth'
        self.used_sta_model = f'sta_model_{sta_model_num}.pth'

        self.train_save_data_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/part1/' \
                                    f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                                    f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}'
        self.test_save_data_root_p1val = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/part1/' \
                                         f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                                         f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}/p1val'

        self.test_save_data_root_p2new = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/part1/' \
                                      f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                                      f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}/p2new'
        self.test_save_data_root_p2valnew = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/part1/' \
                                         f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                                         f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}/p2valnew'

        self.test_save_data_root_p4new = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/part1/' \
                                      f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                                      f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}/p4new'
        self.test_save_data_root_p4valnew = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/part1/' \
                                         f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                                         f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}/p4valnew'

        if not os.path.exists(self.train_save_data_root):
            os.makedirs(self.train_save_data_root)
        if not os.path.exists(self.test_save_data_root_p1val):
            os.makedirs(self.test_save_data_root_p1val)
        if not os.path.exists(self.test_save_data_root_p2new):
            os.makedirs(self.test_save_data_root_p2new)
        if not os.path.exists(self.test_save_data_root_p2valnew):
            os.makedirs(self.test_save_data_root_p2valnew)
        if not os.path.exists(self.test_save_data_root_p4new):
            os.makedirs(self.test_save_data_root_p4new)
        if not os.path.exists(self.test_save_data_root_p4valnew):
            os.makedirs(self.test_save_data_root_p4valnew)

        self.res_root_p1val = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/part1/' \
                              f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                              f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}/p1val'
        self.res_root_p2new = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/part1/' \
                           f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                           f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}/p2new'
        self.res_root_p2valnew = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/part1/' \
                              f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                              f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}/p2valnew'
        self.res_root_p4new = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/part1/' \
                           f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                           f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}/p4new'
        self.res_root_p4valnew = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/part1/' \
                              f'{self.prefix_name}_{self.used_config}_k{self.k}_dm{self.class_drop_ratio}_ak{self.a_k}_adm{self.a_class_drop_ratio}' \
                              f'_qm{self.seq_margin}tm{self.sta_margin}_l{self.sequence_length}_weighted_alq{self.alpha_seq_contra}_{self.alpha_seq_recon}_alt{self.alpha_sta_contra}_{self.alpha_sta_recon}_mq{seq_model_num}t{sta_model_num}/p4valnew'

        if not os.path.exists(self.res_root_p1val):
            os.makedirs(self.res_root_p1val)
        if not os.path.exists(self.res_root_p2new):
            os.makedirs(self.res_root_p2new)
        if not os.path.exists(self.res_root_p2valnew):
            os.makedirs(self.res_root_p2valnew)
        if not os.path.exists(self.res_root_p4new):
            os.makedirs(self.res_root_p4new)
        if not os.path.exists(self.res_root_p4valnew):
            os.makedirs(self.res_root_p4valnew)

        # 处理聚类标签，生成簇数据
        if len(os.listdir(self.train_coarse_root_normal)) == 0:
            print('[Attention] fine feature root is empty, please generate data for train.')
        self.relabel_attack = True
        if self.relabel_attack:

            print('regenerate attack cluster label')
            label_cluster_attack_for_train('kmeans', self.train_attack_files, self.train_attack_root, self.train_coarse_root_attack, [self.a_k, self.a_class_drop_ratio])
            label_cluster_attack_for_test(self.test_attack_files_p1val, self.test_attack_root, self.test_coarse_root_attack)
            label_cluster_attack_for_test(self.test_attack_files_p2new, self.test_attack_root,
                                          self.test_coarse_root_attack)
            label_cluster_attack_for_test(self.test_attack_files_p2valnew, self.test_attack_root,
                                          self.test_coarse_root_attack)
            label_cluster_attack_for_test(self.test_attack_files_p4new, self.test_attack_root,
                                          self.test_coarse_root_attack)
            label_cluster_attack_for_test(self.test_attack_files_p4valnew, self.test_attack_root,
                                          self.test_coarse_root_attack)

            print('regenerate normal cluster label')
            label_cluster_normal_for_train('kmeans', self.train_normal_files, self.train_normal_root, self.train_coarse_root_normal, [self.k, self.class_drop_ratio])
            label_cluster_normal_for_test(self.test_normal_files_p1val, self.test_normal_root, self.test_coarse_root_normal)
            label_cluster_normal_for_test(self.test_normal_files_p2new, self.test_normal_root,
                                          self.test_coarse_root_normal)
            label_cluster_normal_for_test(self.test_normal_files_p2valnew, self.test_normal_root,
                                          self.test_coarse_root_normal)
            label_cluster_normal_for_test(self.test_normal_files_p4new, self.test_normal_root,
                                          self.test_coarse_root_normal)
            label_cluster_normal_for_test(self.test_normal_files_p4valnew, self.test_normal_root,
                                          self.test_coarse_root_normal)

            # 聚类之后，所有正常文件被合并到同一个文件里了
        self.train_normal_paths = [os.path.join(self.train_coarse_root_normal,
                                                f'normal_kmeans_k{self.k}_dm{self.class_drop_ratio}.pickle')]
        self.train_attack_paths = [os.path.join(self.train_coarse_root_attack,
                                                f'attack_kmeans_k{self.a_k}_dm{self.a_class_drop_ratio}.pickle')]
        self.test_normal_paths_p1val = [os.path.join(self.test_coarse_root_normal, i) for i in
                                        self.test_normal_files_p1val]
        self.test_attack_paths_p1val = [os.path.join(self.test_coarse_root_attack, i) for i in
                                        self.test_attack_files_p1val]

        self.test_normal_paths_p2new = [os.path.join(self.test_coarse_root_normal, i) for i in
                                        self.test_normal_files_p2new]
        self.test_attack_paths_p2new = [os.path.join(self.test_coarse_root_attack, i) for i in
                                        self.test_attack_files_p2new]
        self.test_normal_paths_p2valnew = [os.path.join(self.test_coarse_root_normal, i) for i in
                                           self.test_normal_files_p2valnew]
        self.test_attack_paths_p2valnew = [os.path.join(self.test_coarse_root_attack, i) for i in
                                           self.test_attack_files_p2valnew]

        self.test_normal_paths_p4new = [os.path.join(self.test_coarse_root_normal, i) for i in
                                        self.test_normal_files_p4new]
        self.test_attack_paths_p4new = [os.path.join(self.test_coarse_root_attack, i) for i in
                                        self.test_attack_files_p4new]
        self.test_normal_paths_p4valnew = [os.path.join(self.test_coarse_root_normal, i) for i in
                                           self.test_normal_files_p4valnew]
        self.test_attack_paths_p4valnew = [os.path.join(self.test_coarse_root_attack, i) for i in
                                           self.test_attack_files_p4valnew]

        self.train_paths = self.train_normal_paths + self.train_attack_paths
        self.test_paths_p1val = self.test_normal_paths_p1val + self.test_attack_paths_p1val
        self.test_paths_p2new = self.test_normal_paths_p2new + self.test_attack_paths_p2new
        self.test_paths_p2valnew = self.test_normal_paths_p2valnew + self.test_attack_paths_p2valnew
        self.test_paths_p4new = self.test_normal_paths_p4new + self.test_attack_paths_p4new
        self.test_paths_p4valnew = self.test_normal_paths_p4valnew + self.test_attack_paths_p4valnew


if __name__ == '__main__':
    myconfig = InitTrainConfig(args.config, args.k, args.cdr, args.prefix_name, args.seq_margin, args.sta_margin,
                               args.seq_model_num, args.sta_model_num, args.alpha_seq_contra, args.alpha_seq_recon,
                               args.alpha_sta_contra, args.alpha_sta_recon, args.execute_folder_name, args.a_k, args.a_cdr)

    print('[INFO] Begin load train dataset.')
    train_dataset = StaSeqTrafficNormalizedDataset(myconfig.train_paths, myconfig.class_dict,
                                                   myconfig.detail_class_dict,
                                                   myconfig, False,
                                                   os.path.join(myconfig.train_save_data_root, 'scaler.pickle'))
    sample_weights = get_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, int(len(sample_weights) / 4))
    train_loader_weihted = DataLoader(train_dataset, batch_size=myconfig.batch_size, sampler=sampler)  # 训练的时候使用有权重的
    train_loader_noweighted = DataLoader(train_dataset, batch_size=myconfig.batch_size,
                                         shuffle=True)  # 计算训练数据的分布的时候使用没有权重的

    if args.re_train == 'True' or args.re_train == 'true':
        re_train = True
    else:
        re_train = False

    print('[INFO] Train model')
    if re_train:
        print('[INFO] retrain is true')
        logging.info('[INFO] retrain is true')
        train_model(myconfig, train_loader_weihted, myconfig.model_root, myconfig.alpha_seq_contra,
                    myconfig.alpha_seq_recon, myconfig.alpha_sta_contra, myconfig.alpha_sta_recon)
    else:
        if (not os.path.exists(os.path.join(myconfig.model_root, myconfig.used_seq_model))) or (
                not os.path.exists(os.path.join(myconfig.model_root, myconfig.used_sta_model))):
            print('[INFO] retrain is false, but no model exists.')
            logging.info('[INFO] retrain is false, but no model exists.')
            train_model(myconfig, train_loader_weihted, myconfig.model_root, myconfig.alpha_seq_contra,
                        myconfig.alpha_seq_recon, myconfig.alpha_sta_contra, myconfig.alpha_sta_recon)
        else:
            print('[INFO] retrain is false, skip step 1 train model')
            logging.info('[INFO] retrain is false, skip step 1 train model')

    if args.single_eval_p2p4 == 'yes':
        test_dataset_p1val = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p1val, myconfig.class_dict,
                                                            myconfig.detail_class_dict,
                                                            myconfig, False,
                                                            os.path.join(myconfig.train_save_data_root,
                                                                         'scaler.pickle'))
        test_dataset_p2new = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p2new, myconfig.class_dict,
                                                            myconfig.detail_class_dict,
                                                            myconfig, False,
                                                            os.path.join(myconfig.train_save_data_root,
                                                                         'scaler.pickle'))
        test_dataset_p2valnew = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p2valnew, myconfig.class_dict,
                                                               myconfig.detail_class_dict,
                                                               myconfig, False,
                                                               os.path.join(myconfig.train_save_data_root,
                                                                            'scaler.pickle'))
        test_dataset_p4new = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p4new, myconfig.class_dict,
                                                            myconfig.detail_class_dict,
                                                            myconfig, False,
                                                            os.path.join(myconfig.train_save_data_root,
                                                                         'scaler.pickle'))
        test_dataset_p4valnew = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p4valnew, myconfig.class_dict,
                                                               myconfig.detail_class_dict,
                                                               myconfig, False,
                                                               os.path.join(myconfig.train_save_data_root,
                                                                            'scaler.pickle'))
        test_loader_p1val = DataLoader(test_dataset_p1val, batch_size=myconfig.batch_size, shuffle=False)
        test_loader_p2new = DataLoader(test_dataset_p2new, batch_size=myconfig.batch_size, shuffle=False)
        test_loader_p2valnew = DataLoader(test_dataset_p2valnew, batch_size=myconfig.batch_size, shuffle=False)
        test_loader_p4new = DataLoader(test_dataset_p4new, batch_size=myconfig.batch_size, shuffle=False)
        test_loader_p4valnew = DataLoader(test_dataset_p4valnew, batch_size=myconfig.batch_size, shuffle=False)

        eval_model_stage1('train part1, test part1val', myconfig, train_loader_noweighted, test_loader_p1val,
                          myconfig.model_root,
                          myconfig.res_root_p1val, myconfig.train_save_data_root,
                          myconfig.test_save_data_root_p1val, myconfig.used_seq_model, myconfig.used_sta_model,
                          recalculate_train_related=True)
        eval_model_stage1('train part1, test part2new', myconfig, train_loader_noweighted, test_loader_p2new,
                          myconfig.model_root,
                          myconfig.res_root_p2new, myconfig.train_save_data_root,
                          myconfig.test_save_data_root_p2new, myconfig.used_seq_model, myconfig.used_sta_model,
                          recalculate_train_related=False)
        eval_model_stage1('train part1, test part2valnew', myconfig, train_loader_noweighted, test_loader_p2valnew,
                          myconfig.model_root,
                          myconfig.res_root_p2valnew, myconfig.train_save_data_root,
                          myconfig.test_save_data_root_p2valnew, myconfig.used_seq_model, myconfig.used_sta_model,
                          recalculate_train_related=False)
        eval_model_stage1('train part1, test part4new', myconfig, train_loader_noweighted, test_loader_p4new,
                          myconfig.model_root,
                          myconfig.res_root_p4new, myconfig.train_save_data_root,
                          myconfig.test_save_data_root_p4new, myconfig.used_seq_model, myconfig.used_sta_model,
                          recalculate_train_related=False)
        eval_model_stage1('train part1, test part4valnew', myconfig, train_loader_noweighted, test_loader_p4valnew,
                          myconfig.model_root,
                          myconfig.res_root_p4valnew, myconfig.train_save_data_root,
                          myconfig.test_save_data_root_p4valnew, myconfig.used_seq_model, myconfig.used_sta_model,
                          recalculate_train_related=False)
    else:
        print('[INFO] Begin load test dataset.')
        if args.single_eval_p4 == 'no':
            # 如果不是单独评估p4
            test_dataset_p1val = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p1val, myconfig.class_dict,
                                                                myconfig.detail_class_dict,
                                                                myconfig, False,
                                                                os.path.join(myconfig.train_save_data_root,
                                                                             'scaler.pickle'))
            # test_dataset_p2 = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p2, myconfig.class_dict,
            #                                                  myconfig.detail_class_dict,
            #                                                  myconfig, False,
            #                                                  os.path.join(myconfig.train_save_data_root,
            #                                                               'scaler.pickle'))
            # test_dataset_p2val = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p2val, myconfig.class_dict,
            #                                                     myconfig.detail_class_dict,
            #                                                     myconfig, False,
            #                                                     os.path.join(myconfig.train_save_data_root,
            #                                                                  'scaler.pickle'))
            # test_dataset_p3 = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p3, myconfig.class_dict,
            #                                                  myconfig.detail_class_dict,
            #                                                  myconfig, False,
            #                                                  os.path.join(myconfig.train_save_data_root,
            #                                                               'scaler.pickle'))

            test_loader_p1val = DataLoader(test_dataset_p1val, batch_size=myconfig.batch_size, shuffle=False)
            # test_loader_p2 = DataLoader(test_dataset_p2, batch_size=myconfig.batch_size, shuffle=False)
            # test_loader_p2val = DataLoader(test_dataset_p2val, batch_size=myconfig.batch_size, shuffle=False)
            # test_loader_p3 = DataLoader(test_dataset_p3, batch_size=myconfig.batch_size, shuffle=False)

            print('[INFO] Evaluate model ')
            eval_model_stage1('train part1, test part1val', myconfig, train_loader_noweighted, test_loader_p1val,
                              myconfig.model_root,
                              myconfig.res_root_p1val, myconfig.train_save_data_root,
                              myconfig.test_save_data_root_p1val, myconfig.used_seq_model, myconfig.used_sta_model)
        #     eval_model_stage1('train part1, test part2', myconfig, train_loader_noweighted, test_loader_p2,
        #                       myconfig.model_root,
        #                       myconfig.res_root_p2, myconfig.train_save_data_root,
        #                       myconfig.test_save_data_root_p2, myconfig.used_seq_model, myconfig.used_sta_model)
        #     eval_model_stage1('train part1, test part2val', myconfig, train_loader_noweighted, test_loader_p2val,
        #                       myconfig.model_root,
        #                       myconfig.res_root_p2val, myconfig.train_save_data_root,
        #                       myconfig.test_save_data_root_p2val, myconfig.used_seq_model, myconfig.used_sta_model)
        #     eval_model_stage1('train part1, test part3', myconfig, train_loader_noweighted, test_loader_p3,
        #                       myconfig.model_root,
        #                       myconfig.res_root_p3, myconfig.train_save_data_root,
        #                       myconfig.test_save_data_root_p3, myconfig.used_seq_model, myconfig.used_sta_model)
        #
        # test_dataset_p4 = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p4, myconfig.class_dict,
        #                                                  myconfig.detail_class_dict,
        #                                                  myconfig, False,
        #                                                  os.path.join(myconfig.train_save_data_root, 'scaler.pickle'))
        # test_loader_p4 = DataLoader(test_dataset_p4, batch_size=myconfig.batch_size, shuffle=False)
        # print('[INFO] Finish load dataset p4.')
        # eval_model_stage1('train part1, test part4', myconfig, train_loader_noweighted, test_loader_p4,
        #                   myconfig.model_root,
        #                   myconfig.res_root_p4, myconfig.train_save_data_root,
        #                   myconfig.test_save_data_root_p4, myconfig.used_seq_model, myconfig.used_sta_model)
        #
        # test_dataset_p4val = StaSeqTrafficNormalizedDataset(myconfig.test_paths_p4val, myconfig.class_dict,
        #                                                     myconfig.detail_class_dict,
        #                                                     myconfig, False,
        #                                                     os.path.join(myconfig.train_save_data_root,
        #                                                                  'scaler.pickle'))
        # test_loader_p4val = DataLoader(test_dataset_p4val, batch_size=myconfig.batch_size, shuffle=False)
        # print('[INFO] Finish load dataset p4val.')
        # eval_model_stage1('train part1, test part4val', myconfig, train_loader_noweighted, test_loader_p4val,
        #                   myconfig.model_root,
        #                   myconfig.res_root_p4val, myconfig.train_save_data_root,
        #                   myconfig.test_save_data_root_p4val, myconfig.used_seq_model, myconfig.used_sta_model)
        #




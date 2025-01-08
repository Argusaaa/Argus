#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import random
import sys, importlib
import torch
import logging
from datetime import datetime
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, Subset, ConcatDataset
import pickle
import itertools

sys.path.append('/home/driftDetect/newcode0625')
sys.path.append('D:\PycharmProject\DriftDetection/newcode0625')

DatasetConfig1 = importlib.import_module('dataconfig_ids2018').DatasetConfig1
StaSeqTrafficNormalizedDatasetUpdateUncertain = importlib.import_module(
    'myutil').StaSeqTrafficNormalizedDatasetUpdateUncertain
get_sample_weights = importlib.import_module('myutil').get_sample_weights
get_sample_weights_for_multiple_dataset = importlib.import_module('myutil').get_sample_weights_for_multiple_dataset
ContraLossEucNewM2 = importlib.import_module('myutil').ContraLossEucNewM2
StaSeqTrafficNormalizedDataset = importlib.import_module('myutil').StaSeqTrafficNormalizedDataset

uncertain_train_update_model = importlib.import_module('myutil').uncertain_train_update_model
get_uncertain_related = importlib.import_module('myutil').get_uncertain_related
combine_two_distribution_uncertain = importlib.import_module('myutil').combine_two_distribution_uncertain
validate_drift = importlib.import_module('myutil').validate_drift
eval_model_stage2 = importlib.import_module('myutil').eval_model_stage2
StaSeqTrafficNormalizedDatasetUpdateOri = importlib.import_module('myutil').StaSeqTrafficNormalizedDatasetUpdateOri
uncertain_train_update_model_to_center = importlib.import_module('myutil').uncertain_train_update_model_to_center
CustomSubset = importlib.import_module('myutil').CustomSubset
uncertain_train_update_model_to_center_uc = importlib.import_module('myutil').uncertain_train_update_model_to_center_uc
uncertain_train_update_model_uc = importlib.import_module('myutil').uncertain_train_update_model_uc

parser = argparse.ArgumentParser(description='step 4 uncertain update',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--used_config', type=str, default='config1')
parser.add_argument('--config_folder', type=str)
parser.add_argument('--data_folder', type=str)
parser.add_argument('--new_normal_file', type=str)
parser.add_argument('--new_attack_file', type=str)
parser.add_argument('--old_test_part_name', type=str)
parser.add_argument('--used_seq_model_ori_num', type=int, default=266)
parser.add_argument('--used_sta_model_ori_num', type=int, default=266)
parser.add_argument('--used_seq_model_certain_num', type=int, default=10)
parser.add_argument('--used_sta_model_certain_num', type=int, default=10)
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--used_seq_model_uncertain_num', type=int)
parser.add_argument('--used_sta_model_uncertain_num', type=int)
parser.add_argument('--min_sample_each_file_uncertain', type=int, default=100)
parser.add_argument('--execute_folder_name', type=str, default='7_ids_script_new')
parser.add_argument('--alpha_margin', type=float, default=1.1)
parser.add_argument('--un_seq_margin', type=float, default=10)
parser.add_argument('--un_sta_margin', type=float, default=10)
parser.add_argument('--seq_contra_lamda', type=float, default=1)
parser.add_argument('--seq_recon_lamda', type=float, default=0.05)
parser.add_argument('--seq_tocenter_lamda', type=float, default=1)
parser.add_argument('--seq_dist_lamda', type=float, default=0.001)
parser.add_argument('--sta_contra_lamda', type=float, default=1)
parser.add_argument('--sta_recon_lamda', type=float, default=0.05)
parser.add_argument('--sta_tocenter_lamda', type=float, default=1)
parser.add_argument('--sta_dist_lamda', type=float, default=0.001)
parser.add_argument('--used_self_labeled_data', type=str, default='yes')
parser.add_argument('--uncertain_update_way', type=str,
                    default='dist_center')  # 分两种情况，dist_each表示约束原始样本新旧表示之间的距离，dist_center表示约束原始样本到簇心的距离的变化
parser.add_argument('--old_dataset_beishu_uncertain', type=float,
                    default=1.0)  # 在更新数据的时候，旧数据集使用新数据集的多少倍（uncertain更新的时候）
parser.add_argument('--old_dataset_beishu_certain', type=float, default=1.0)  # 在certain更新的时候，旧数据是多少倍（主要是产生certain相关的路径）

parser.add_argument('--idcl_eps', type=float, default=0.5)
parser.add_argument('--idcl_minpts', type=int, default=10)
parser.add_argument('--edge_div_node_threshold', type=float, default=5)
parser.add_argument('--edge_div_com_threshold', type=float, default=10)

parser.add_argument('--use_different_cal_par', type=str, default='no')  # 在重新计算参数时，是否使用不同的数据去计算参数，之前的数据可能不够泛化
parser.add_argument('--train_model', type=str, default='yes')  # 运行代码的过程是否训练模型

parser.add_argument('--refine_uncertain', type=str, default='')

args = parser.parse_args()


def check_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class UpdateUncertainConfig:
    def __init__(self, used_config, config_folder, data_folder, new_normal_file, new_attack_file, old_test_part_name,
                 used_seq_model_ori_num, used_sta_model_ori_num,
                 used_seq_model_certain_num, used_sta_model_certain_num, used_seq_model_uncertain_num,
                 used_sta_model_uncertain_num,
                 min_sample_each_file_uncertain, execute_folder_name,
                 alpha_margin=1.1, un_seq_margin=10, un_sta_margin=10,
                 seq_contra_lamda=1, seq_recon_lamda=0.05, seq_tocenter_lamda=1, seq_dist_lamda=0.001,
                 sta_contra_lamda=1, sta_recon_lamda=0.05, sta_tocenter_lamda=1, sta_dist_lamda=0.001,
                 used_self_labeled_data='yes', train_epoch=100, uncertain_update_way='dist_each',
                 old_dataset_beishu_uncertain=1.0, old_dataset_beishu_certain=1.0,
                 idcl_eps=0.5, idcl_minpts=10, edge_div_node_threshold=5, edge_div_com_threshold=10,
                 refine_uncertain=''):

        self.used_config = used_config  # 'config1'

        self.codb = old_dataset_beishu_certain
        self.uodb = old_dataset_beishu_uncertain

        self.config_folder = config_folder  # 'kmeans0625_config1_k2_dm0.05_qm10tm10_l50_weighted'
        self.data_folder = data_folder  # 'kmeans0625_config1_k2_dm0.05_weighted'
        self.ori_file_root = f'/home/driftDetect/0_data/1_cicids2018/4_ids_all_run_data/{self.data_folder}'

        self.new_normal_file = new_normal_file  # 'normal_kmeans_k2_dm0.05.pickle'
        self.new_attack_file = new_attack_file
        self.old_test_part_name = old_test_part_name

        self.used_self_labeled_data_certain = used_self_labeled_data  # certain数据对应的文件夹没做修改
        if uncertain_update_way == 'dist_each':
            self.used_self_labeled_data_uncertain = used_self_labeled_data
        elif uncertain_update_way == 'dist_center':
            # 后加的情况，所以进行了一点改动
            self.used_self_labeled_data_uncertain = f'{used_self_labeled_data}_center'

        if self.used_config == 'config1':
            if (new_normal_file is not None) and (new_attack_file is not None):
                self.dataconfig = DatasetConfig1(new_normal_file=self.new_normal_file,
                                                 new_attack_file=self.new_attack_file)
            elif (new_normal_file is not None) and (new_attack_file is None):
                self.dataconfig = DatasetConfig1(new_normal_file=self.new_normal_file)
            else:
                print('[ERROR] Dataconfig error occurs, both empty')
            # self.dataconfig = DatasetConfig1(self.new_normal_file)

        self.min_sample_each_file_uncertain = min_sample_each_file_uncertain  # 100  # 如果一个文件中的样本数少于这些，就不使用这个文件

        # 来自step2的参数
        # 确定certain的列
        self.cocertain_row = 'dist_co_certain'  # 看seq和sta的结果是否一致
        self.consider_unmatch = True  # 是否考虑seq和sta结果不一致的情况，true就是把不匹配的也视为uncertain
        self.used_uncertain_row = 'cla_md_recon_uncertain_3'
        self.regenerate_cocertain = False
        # 在identify的时候聚类的参数
        self.idcl_eps = idcl_eps
        self.idcl_minpts = idcl_minpts
        # 在判断聚类得到的簇是否为恶意时的参数
        self.edge_div_node_threshold = edge_div_node_threshold
        self.edge_div_com_threshold = edge_div_com_threshold

        # 使用了更新后的模型的临时的一个路径文件夹，因为后面很多用这个的（主要是result，save data相关的，model的存储路径不是这样
        if refine_uncertain=='':
            certain_update_tmp_folder_name = f'certain_usesl_{self.used_self_labeled_data_certain}_codb{self.codb}_om{used_seq_model_ori_num}_{used_sta_model_ori_num}_cm{used_seq_model_certain_num}_{used_sta_model_certain_num}'
            uncertain_update_tmp_folder_name = f'uncertain_usesl_{self.used_self_labeled_data_uncertain}_codb{self.uodb}_uodb{self.uodb}_om{used_seq_model_ori_num}_{used_sta_model_ori_num}_cm{used_seq_model_certain_num}_{used_sta_model_certain_num}_un{used_seq_model_uncertain_num}_{used_sta_model_uncertain_num}'
        else:
            certain_update_tmp_folder_name = f'certain_usesl_{self.used_self_labeled_data_certain}_codb{self.codb}_om{used_seq_model_ori_num}_{used_sta_model_ori_num}_cm{used_seq_model_certain_num}_{used_sta_model_certain_num}'
            uncertain_update_tmp_folder_name = f'uncertain_usesl_{self.used_self_labeled_data_uncertain}{refine_uncertain}_codb{self.uodb}_uodb{self.uodb}_om{used_seq_model_ori_num}_{used_sta_model_ori_num}_cm{used_seq_model_certain_num}_{used_sta_model_certain_num}_un{used_seq_model_uncertain_num}_{used_sta_model_uncertain_num}'

        # uncertain和certain的folder是一样的
        # 用于更新模型的数据所在的路径（如果使用的是identify drift生成的数据）
        self.certain_name_folder = f'{self.cocertain_row}-{self.used_uncertain_row}-eps{self.idcl_eps}minpts{self.idcl_minpts}-thre{self.edge_div_node_threshold}_{self.edge_div_com_threshold}'
        if refine_uncertain=='':
            if self.old_test_part_name == 'p2p4':
                self.uncertain_data_root_1 = os.path.join(self.ori_file_root,
                                                        f'p2new_uncertain/{self.certain_name_folder}')
                self.uncertain_data_root_2 = os.path.join(self.ori_file_root,
                                                          f'p4new_uncertain/{self.certain_name_folder}')
            else:
                self.uncertain_data_root = os.path.join(self.ori_file_root,
                                                        f'{self.old_test_part_name}_uncertain/{self.certain_name_folder}')

        else:
            if self.old_test_part_name == 'p2p4':
                self.uncertain_data_root_1 = os.path.join(self.ori_file_root,
                                                        f'p2new_uncertain/{refine_uncertain}_{self.certain_name_folder}')
                self.uncertain_data_root_2 = os.path.join(self.ori_file_root,
                                                          f'p4new_uncertain/{refine_uncertain}_{self.certain_name_folder}')
            else:
                self.uncertain_data_root = os.path.join(self.ori_file_root,
                                                        f'{self.old_test_part_name}_uncertain/{refine_uncertain}_{self.certain_name_folder}')


        # 根据原始数据生成的真实的数据标签所在的路径（如果不使用identify drift生成的数据）[这个只和原始的聚类的划分有关，和config folder没有关系]
        self.ori_label_root = os.path.join(self.ori_file_root, f'ori_label_{self.old_test_part_name}')  # 只有标签，没有数据

        # 基于certain数据更新的模型继续更新
        self.certain_model_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/model/p1{self.old_test_part_name}_certain/{self.config_folder}/certain_usesl_{self.used_self_labeled_data_certain}_codb{self.codb}_om{used_seq_model_ori_num}_{used_sta_model_ori_num}'  # 不涉及cm，所以这里单独写
        # 在certain更新的时候，在训练过程中直接保存了簇心这些，所以这里只使用了new文件夹，（7.29）
        self.certain_save_data_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/p1{self.old_test_part_name}_certain/{self.config_folder}/{certain_update_tmp_folder_name}/{self.old_test_part_name}_new'

        # 使用用certain更新好的哪个模型
        self.used_certain_seq_model = f'seq_model_{used_seq_model_certain_num}.pth'
        self.used_certain_sta_model = f'sta_model_{used_sta_model_certain_num}.pth'

        # 使用uncertain更新后的模型、中间数据和结果存储在哪里
        if refine_uncertain=='':
            self.uncertain_model_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/model/p1{self.old_test_part_name}_uncertain' \
                                        f'/{self.config_folder}/uncertain_usesl_{self.used_self_labeled_data_uncertain}_codb{self.codb}_uodb{self.uodb}_om{used_seq_model_ori_num}_{used_sta_model_ori_num}_cm{used_seq_model_certain_num}_{used_sta_model_certain_num}'
        else:
            self.uncertain_model_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/model/p1{self.old_test_part_name}_uncertain' \
                                        f'/{self.config_folder}/uncertain_usesl_{self.used_self_labeled_data_uncertain}{refine_uncertain}_codb{self.codb}_uodb{self.uodb}_om{used_seq_model_ori_num}_{used_sta_model_ori_num}_cm{used_seq_model_certain_num}_{used_sta_model_certain_num}'

        self.uncertain_new_save_data_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/p1{self.old_test_part_name}_uncertain/{self.config_folder}/{uncertain_update_tmp_folder_name}/{self.old_test_part_name}_new'
        self.uncertain_updated_save_data_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/p1{self.old_test_part_name}_uncertain/{self.config_folder}/{uncertain_update_tmp_folder_name}/{self.old_test_part_name}_updated'
        # 验证在使用uncertain更新的模型上，数据偏移了多少，存储中间结果，主要是用过去的数据处理
        self.uncertain_driftvalidate_save_data_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/p1{self.old_test_part_name}_uncertain_driftvalidate/{self.config_folder}/{uncertain_update_tmp_folder_name}'

        check_paths(
            [self.uncertain_model_root, self.uncertain_new_save_data_root, self.uncertain_updated_save_data_root,
             self.uncertain_driftvalidate_save_data_root])

        # 使用更新后的模型进行验证，结果和中间结果保存在哪里
        self.uncertain_updated_test_result_root_p1val = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/p1{self.old_test_part_name}_uncertain/{self.config_folder}/{self.old_test_part_name}_updated/{uncertain_update_tmp_folder_name}/p1val'
        self.uncertain_update_test_save_data_root_p1val = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/p1{self.old_test_part_name}_uncertain/{self.config_folder}/{self.old_test_part_name}_updated/{uncertain_update_tmp_folder_name}/p1val'
        self.uncertain_updated_test_result_root_p2valnew = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/p1{self.old_test_part_name}_uncertain/{self.config_folder}/{self.old_test_part_name}_updated/{uncertain_update_tmp_folder_name}/p2valnew'
        self.uncertain_update_test_save_data_root_p2valnew = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/p1{self.old_test_part_name}_uncertain/{self.config_folder}/{self.old_test_part_name}_updated/{uncertain_update_tmp_folder_name}/p2valnew'
        self.uncertain_updated_test_result_root_p4valnew = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/p1{self.old_test_part_name}_uncertain/{self.config_folder}/{self.old_test_part_name}_updated/{uncertain_update_tmp_folder_name}/p4valnew'
        self.uncertain_update_test_save_data_root_p4valnew = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/p1{self.old_test_part_name}_uncertain/{self.config_folder}/{self.old_test_part_name}_updated/{uncertain_update_tmp_folder_name}/p4valnew'

        check_paths([self.uncertain_updated_test_result_root_p1val, self.uncertain_update_test_save_data_root_p1val,
                     self.uncertain_updated_test_result_root_p2valnew, self.uncertain_update_test_save_data_root_p2valnew,
                     self.uncertain_updated_test_result_root_p4valnew, self.uncertain_update_test_save_data_root_p4valnew])

        # 测试的路径，相关数据
        self.class_dict = self.dataconfig.class_dict
        self.detail_class_dict = self.dataconfig.detail_class_dict
        self.test_files_p1val = self.dataconfig.part1_normal_files_val + self.dataconfig.part1_attack_files_val
        self.test_files_p2valnew = self.dataconfig.part2_normal_files_val_new + self.dataconfig.part2_attack_files_val_new
        self.test_files_p4valnew = self.dataconfig.part4_normal_files_val_new + self.dataconfig.part4_attack_files_val_new
        self.test_files_p1 = self.dataconfig.part1_normal_files + self.dataconfig.part1_attack_files

        # 如果使用原有的标签，所需的信息
        if self.old_test_part_name == 'p2':
            self.normal_certain_files = self.dataconfig.part2_normal_certain_files
            self.normal_uncertain_files = self.dataconfig.part2_normal_uncertain_files
            self.attack_certain_files = self.dataconfig.part2_attack_certain_files
            self.attack_uncertain_files = self.dataconfig.part2_attack_uncertain_files
        elif self.old_test_part_name == 'p4':
            self.normal_certain_files = self.dataconfig.part4_normal_certain_files
            self.normal_uncertain_files = self.dataconfig.part4_normal_uncertain_files
            self.attack_certain_files = self.dataconfig.part4_attack_certain_files
            self.attack_uncertain_files = self.dataconfig.part4_attack_uncertain_files
        elif self.old_test_part_name == 'p2new':
            self.normal_certain_files = self.dataconfig.part2_normal_certain_files_new
            self.normal_uncertain_files = self.dataconfig.part2_normal_uncertain_files_new
            self.attack_certain_files = self.dataconfig.part2_attack_certain_files_new
            self.attack_uncertain_files = self.dataconfig.part2_attack_uncertain_files_new
        elif self.old_test_part_name == 'p4new':
            self.normal_certain_files = self.dataconfig.part4_normal_certain_files_new
            self.normal_uncertain_files = self.dataconfig.part4_normal_uncertain_files_new
            self.attack_certain_files = self.dataconfig.part4_attack_certain_files_new
            self.attack_uncertain_files = self.dataconfig.part4_attack_uncertain_files_new
        elif self.old_test_part_name == 'p2p4':
            self.normal_certain_files_1 = self.dataconfig.part2_normal_certain_files_new
            self.normal_uncertain_files_1 = self.dataconfig.part2_normal_uncertain_files_new
            self.attack_certain_files_1 = self.dataconfig.part2_attack_certain_files_new
            self.attack_uncertain_files_1 = self.dataconfig.part2_attack_uncertain_files_new
            self.normal_certain_files_2 = self.dataconfig.part4_normal_certain_files_new
            self.normal_uncertain_files_2 = self.dataconfig.part4_normal_uncertain_files_new
            self.attack_certain_files_2 = self.dataconfig.part4_attack_certain_files_new
            self.attack_uncertain_files_2 = self.dataconfig.part4_attack_uncertain_files_new
        else:
            self.normal_certain_files = None
            self.normal_uncertain_files = None
            self.attack_certain_files = None
            self.attack_uncertain_files = None

        # 使用uncertain更新后的模型用哪个
        self.used_seq_model_uncertain_update = f'seq_model_{used_seq_model_uncertain_num}.pth'
        self.used_sta_model_uncertain_update = f'sta_model_{used_sta_model_uncertain_num}.pth'

        # 原来的簇心
        self.old_p2cer_sta_centers_path = os.path.join(self.certain_save_data_root, 'train_sta_info_stadata.pickle')
        self.old_p2cer_seq_centers_path = os.path.join(self.certain_save_data_root, 'train_sta_info.pickle')

        # 原来的scaler的路径
        self.old_save_data_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/part1/{self.config_folder}'
        self.ori_scaler_path = os.path.join(self.old_save_data_root, 'scaler.pickle')

        # 旧 seq model 的参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_size = 2
        self.d_model = 16
        self.nhead = 2
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.dim_feedforward = 32
        self.dropout = 0.1
        self.sequence_length = 50

        # =======
        # 更新训练模型的参数
        self.batch_size = 1024
        self.train_epoch = train_epoch
        self.seq_lr = 0.0005  # 0.001
        self.sta_lr = 0.0005
        self.seq_eta_min = 5e-06
        self.sta_eta_min = 5e-06

        # 更新时margin是原来的多少倍
        self.alpha_margin = alpha_margin  # 1.1
        self.seq_margin = un_seq_margin  # 10
        self.sta_margin = un_sta_margin  # 10
        self.temperature = 1

        # 在计算损失时各项的权重
        self.seq_contra_lamda = seq_contra_lamda  # 1
        self.seq_recon_lamda = seq_recon_lamda  # 0.05  # /20
        self.seq_tocenter_lamda = seq_tocenter_lamda  # 1  # 距离原来的簇心的距离对应的损失
        self.seq_dist_lamda = seq_dist_lamda  # 0.001  # 和原来的样本的距离的约束
        self.sta_contra_lamda = sta_contra_lamda  # 1
        self.sta_recon_lamda = sta_recon_lamda  # 0.05  # /20
        self.sta_tocenter_lamda = sta_tocenter_lamda  # 1
        self.sta_dist_lamda = sta_dist_lamda  # 0.001

        # 旧的配置文件
        # if refine_uncertain=='':
        #     logging.basicConfig(level=logging.DEBUG,
        #                         filename=f'log/updata-uncertain-{self.config_folder}-uncertain_usesl_{self.used_self_labeled_data_uncertain}.log',
        #                         format='%(message)s')
        # else:
        #     logging.basicConfig(level=logging.DEBUG,
        #                         filename=f'log/updata-uncertain-{self.config_folder}-uncertain_usesl_{self.used_self_labeled_data_uncertain}{refine_uncertain}.log',
        #                         format='%(message)s')
        #
        # logging.info('\n\n ====== ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ======\n\n')
        # logging.info('\n******** update uncertain ********\n')
        # logging.info(f'** uncertain: {self.certain_name_folder} **')  # certain folder和uncertain folder是一样的
        # logging.info(f'** used model: {self.used_seq_model_uncertain_update}, {self.used_sta_model_uncertain_update}')
        # logging.info(f'** codb: {self.codb}, uodb: {self.uodb} **')

        # 为了验证权重的敏感性，重新增加的log文件配置 11.5
        if refine_uncertain=='':
            logging.basicConfig(level=logging.DEBUG,
                                filename=f'log/new1105-update-uncertain-{self.config_folder}-uncertain_usesl_{self.used_self_labeled_data_uncertain}.log',
                                format='%(message)s')
        else:
            logging.basicConfig(level=logging.DEBUG,
                                filename=f'log/new1105-update-uncertain-{self.config_folder}-uncertain_usesl_{self.used_self_labeled_data_uncertain}{refine_uncertain}.log',
                                format='%(message)s')

        logging.info('\n\n ====== ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ======\n\n')
        logging.info('here is the result after uncertain update\n')
        logging.info(
            f'seq_contra_lamda: {self.seq_contra_lamda}, seq_recon_lamda: {self.seq_recon_lamda}, seq_dist_lamda: {self.seq_dist_lamda}, seq_tocenter_lamda: {self.seq_tocenter_lamda}')
        logging.info(
            f'sta_contra_lamda: {self.sta_contra_lamda}, sta_recon_lamda: {self.sta_recon_lamda}, sta_dist_lamda: {self.sta_dist_lamda}, sta_tocenter_lamda: {self.sta_tocenter_lamda}')
        logging.info('\n******** update uncertain ********\n')
        logging.info(f'** uncertain: {self.certain_name_folder} **')  # certain folder和uncertain folder是一样的
        logging.info(f'** used model: {self.used_seq_model_uncertain_update}, {self.used_sta_model_uncertain_update}')
        logging.info(f'** codb: {self.codb}, uodb: {self.uodb} **')


if __name__ == '__main__':
    config = UpdateUncertainConfig(args.used_config, args.config_folder, args.data_folder, args.new_normal_file,
                                   args.new_attack_file, args.old_test_part_name,
                                   args.used_seq_model_ori_num, args.used_sta_model_ori_num,
                                   args.used_seq_model_certain_num, args.used_sta_model_certain_num,
                                   args.used_seq_model_uncertain_num, args.used_sta_model_uncertain_num,
                                   args.min_sample_each_file_uncertain, args.execute_folder_name,
                                   args.alpha_margin, args.un_seq_margin, args.un_sta_margin,
                                   args.seq_contra_lamda, args.seq_recon_lamda, args.seq_tocenter_lamda,
                                   args.seq_dist_lamda,
                                   args.sta_contra_lamda, args.sta_recon_lamda, args.sta_tocenter_lamda,
                                   args.sta_dist_lamda,
                                   args.used_self_labeled_data, args.train_epoch, args.uncertain_update_way,
                                   args.old_dataset_beishu_uncertain, args.old_dataset_beishu_certain,
                                   args.idcl_eps, args.idcl_minpts, args.edge_div_node_threshold,
                                   args.edge_div_com_threshold, args.refine_uncertain)

    if args.used_self_labeled_data == 'yes':
        print('[INFO] Load train dataset')
        logging.info('[INFO] Load train dataset')
        if config.old_test_part_name == 'p2p4':
            train_dataset = StaSeqTrafficNormalizedDatasetUpdateUncertain(
                [os.path.join(config.uncertain_data_root_1, i) for i in os.listdir(config.uncertain_data_root_1)]
                +[os.path.join(config.uncertain_data_root_1, i) for i in os.listdir(config.uncertain_data_root_1)],
                config.ori_scaler_path, False, config)
        else:
            train_dataset = StaSeqTrafficNormalizedDatasetUpdateUncertain(
                [os.path.join(config.uncertain_data_root, i) for i in os.listdir(config.uncertain_data_root)],
                config.ori_scaler_path, False, config)

        train_dataset_old_p1 = StaSeqTrafficNormalizedDataset(
            [os.path.join(config.ori_file_root, i) for i in config.test_files_p1], config.class_dict,
            config.detail_class_dict, config, False, config.ori_scaler_path)
        print(f'LEN: p2 dataset {len(train_dataset)}, p1 dataset {len(train_dataset_old_p1)}')
        # p1的数据集采样和p2相同数量的数据进行训练
        train_dataset_old_p1_subset_indices = random.sample(range(0, len(train_dataset_old_p1)),
                                                            int(len(train_dataset) * args.old_dataset_beishu_uncertain)
                                                            if int(
                                                                len(train_dataset) * args.old_dataset_beishu_uncertain) <
                                                               len(train_dataset_old_p1) else len(train_dataset_old_p1))
        train_dataset_old_p1_subset = CustomSubset(train_dataset_old_p1, train_dataset_old_p1_subset_indices)
        # 合并新数据和旧数据，生成用于certain更新的新数据集，后面就用这个了
        new_train_dataset = ConcatDataset([train_dataset, train_dataset_old_p1_subset])
        sample_weights = get_sample_weights_for_multiple_dataset([train_dataset, train_dataset_old_p1_subset])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader_weighted = DataLoader(new_train_dataset, batch_size=config.batch_size, sampler=sampler)
        train_loader_noweight = DataLoader(new_train_dataset, batch_size=config.batch_size, shuffle=False)

        # 适用训练初始模型的数据，约束新的特征空间和之前不是特别偏移
        print('[INFO] Load old p1 dataset')
        logging.info('[INFO] Load old p1 dataset')
        val_dataset_p1 = StaSeqTrafficNormalizedDataset(
            [os.path.join(config.ori_file_root, i) for i in config.test_files_p1], config.class_dict,
            config.detail_class_dict, config, False, config.ori_scaler_path
        )
        val_p1_part_sampler = RandomSampler(val_dataset_p1, num_samples=3000,
                                            generator=torch.Generator().manual_seed(42))
        # val_p1_part_loader = DataLoader(val_dataset_p1, batch_size=config.batch_size, sampler=val_p1_part_sampler)
        val_p1_part_loader = DataLoader(val_dataset_p1, batch_size=config.batch_size,
                                        shuffle=True)  # sample的样本太少，这里直接在全部的样本上进行选择
        val_p1_part_loader_cycle = itertools.cycle(val_p1_part_loader)

        # 用于验证差异
        val_p1_part_sampler_2 = RandomSampler(val_dataset_p1, num_samples=3000,
                                              generator=torch.Generator().manual_seed(50))
        val_p1_part_loader_2 = DataLoader(val_dataset_p1, batch_size=config.batch_size, sampler=val_p1_part_sampler_2)

        print('[INFO] Begin update model with uncertain')
        logging.info('[INFO] Begin update model with uncertain')
        if args.uncertain_update_way == 'dist_each':
            print('[INFO] Update uncertain with dist each constrain')
            logging.info('[INFO] Update uncertain with dist each constrain')
            if args.train_model == 'yes':
                uncertain_train_update_model_uc(config, train_loader_weighted, val_p1_part_loader_cycle,
                                                os.path.join(config.certain_model_root, config.used_certain_seq_model),
                                                os.path.join(config.certain_model_root, config.used_certain_sta_model),
                                                config.uncertain_model_root,
                                                config.old_p2cer_seq_centers_path, config.old_p2cer_sta_centers_path)
            else:
                print('[INFO] Skip model train')
                logging.info('[INFO] Skip model train')


        elif args.uncertain_update_way == 'dist_center':
            print('[INFO] Update uncertain with dist center constrain')
            logging.info('[INFO] Update uncertain with dist center constrain')
            if args.train_model == 'yes':
                uncertain_train_update_model_to_center_uc(config, train_loader_weighted, val_p1_part_loader_cycle,
                                                      os.path.join(config.certain_model_root,
                                                                   config.used_certain_seq_model),
                                                      os.path.join(config.certain_model_root,
                                                                   config.used_certain_sta_model),
                                                      config.uncertain_model_root,
                                                      config.old_p2cer_seq_centers_path,
                                                      config.old_p2cer_sta_centers_path)
            else:
                print('[INFO] Skip model train')
                logging.info('[INFO] Skip model train')

        if args.use_different_cal_par == 'yes':
            # 如果使用不同的结果，那这里就重置一下dataset noweight
            train_dataset_old_p1_subset_indices_2 = random.sample(range(0, len(train_dataset_old_p1)),
                                                                  len(train_dataset_old_p1) // 3)
            train_dataset_old_p1_subset_2 = CustomSubset(train_dataset_old_p1, train_dataset_old_p1_subset_indices_2)
            # 合并新数据和旧数据，生成用于certain更新的新数据集，后面就用这个了
            new_train_dataset_2 = ConcatDataset([train_dataset, train_dataset_old_p1_subset_2])
            train_loader_noweight = DataLoader(new_train_dataset, batch_size=config.batch_size, shuffle=False)

        get_uncertain_related(config, train_loader_noweight,
                              os.path.join(config.uncertain_model_root, config.used_seq_model_uncertain_update),
                              os.path.join(config.uncertain_model_root, config.used_sta_model_uncertain_update),
                              config.uncertain_new_save_data_root)

        # 需要先运行validate，然后再运行combine了（7.28）
        print('[INFO] Validate latent drift after uncertain update')
        logging.info('[INFO] Validate latent drift after uncertain update')
        validate_drift(val_p1_part_loader_2, os.path.join(config.certain_model_root, config.used_certain_seq_model),
                       os.path.join(config.certain_model_root, config.used_certain_sta_model),
                       os.path.join(config.uncertain_model_root, config.used_seq_model_uncertain_update),
                       os.path.join(config.uncertain_model_root, config.used_sta_model_uncertain_update),
                       config, config.uncertain_driftvalidate_save_data_root)

        # print('[INFO] Update distribution')
        # logging.info('[INFO] Update distribution')
        # combine_two_distribution_uncertain(config.old_save_data_root, config.uncertain_new_save_data_root,
        #                                    config.uncertain_updated_save_data_root)

        print('[INFO] Load test data p1val')
        logging.info('[INFO] Evaluate model')
        logging.info('[INFO] Load test data p1val')
        test_dataset_p1val = StaSeqTrafficNormalizedDataset(
            [os.path.join(config.ori_file_root, i) for i in config.test_files_p1val],
            config.class_dict, config.detail_class_dict, config, False, config.ori_scaler_path)
        test_loader_p1val = DataLoader(test_dataset_p1val, batch_size=config.batch_size, shuffle=False)
        print('[INFO] Evaluate uncertain trained model, use part1val')
        logging.info('[INFO] Evaluate uncertain trained model, use part1val')
        eval_model_stage2('update part2 uncertain, test p1val', config, test_loader_p1val, config.uncertain_model_root,
                          config.uncertain_updated_test_result_root_p1val, config.uncertain_new_save_data_root,
                          config.uncertain_update_test_save_data_root_p1val, config.used_seq_model_uncertain_update,
                          config.used_sta_model_uncertain_update)

        print('[INFO] Load test data p2valnew')
        logging.info('[INFO] Evaluate model')
        logging.info('[INFO] Load test data p2valnew')
        test_dataset_p2valnew = StaSeqTrafficNormalizedDataset(
            [os.path.join(config.ori_file_root, i) for i in config.test_files_p2valnew],
            config.class_dict, config.detail_class_dict, config, False, config.ori_scaler_path)
        test_loader_p2valnew = DataLoader(test_dataset_p2valnew, batch_size=config.batch_size, shuffle=False)
        print('[INFO] Evaluate uncertain trained model, use part2valnew')
        logging.info('[INFO] Evaluate uncertain trained model, use part2valnew')
        eval_model_stage2('update part2 uncertain, test p2valnew', config, test_loader_p2valnew, config.uncertain_model_root,
                          config.uncertain_updated_test_result_root_p2valnew, config.uncertain_new_save_data_root,
                          config.uncertain_update_test_save_data_root_p2valnew, config.used_seq_model_uncertain_update,
                          config.used_sta_model_uncertain_update)

        print('[INFO] Load test data p4valnew')
        logging.info('[INFO] Evaluate model')
        logging.info('[INFO] Load test data p4valnew')
        test_dataset_p4valnew = StaSeqTrafficNormalizedDataset(
            [os.path.join(config.ori_file_root, i) for i in config.test_files_p4valnew],
            config.class_dict, config.detail_class_dict, config, False, config.ori_scaler_path)
        test_loader_p4valnew = DataLoader(test_dataset_p4valnew, batch_size=config.batch_size, shuffle=False)
        print('[INFO] Evaluate uncertain trained model, use part4valnew')
        logging.info('[INFO] Evaluate uncertain trained model, use part4valnew')
        eval_model_stage2('update part2 uncertain, test p4valnew', config, test_loader_p4valnew, config.uncertain_model_root,
                          config.uncertain_updated_test_result_root_p4valnew, config.uncertain_new_save_data_root,
                          config.uncertain_update_test_save_data_root_p4valnew, config.used_seq_model_uncertain_update,
                          config.used_sta_model_uncertain_update)

    elif args.used_self_labeled_data == 'no':
        # ===================
        # 训练模型
        # ===================
        print('[INFO] Load train data (identify ori dist dict labels)')
        logging.info('[INFO] Load train data (identify ori dist dict labels)')
        # 如果使用根据原始数据生成的标签来更新模型
        train_dataset = StaSeqTrafficNormalizedDatasetUpdateOri(
            config.normal_uncertain_files + config.attack_uncertain_files,
            config.ori_file_root, config.ori_label_root, config.ori_scaler_path,
            config.class_dict, config)

        train_dataset_old_p1 = StaSeqTrafficNormalizedDataset(
            [os.path.join(config.ori_file_root, i) for i in config.test_files_p1], config.class_dict,
            config.detail_class_dict, config, False, config.ori_scaler_path)
        print(f'LEN: p2 dataset {len(train_dataset)}, p1 dataset {len(train_dataset_old_p1)}')
        # p1的数据集采样和p2相同数量的数据进行训练
        train_dataset_old_p1_subset_indices = random.sample(range(0, len(train_dataset_old_p1)),
                                                            int(len(train_dataset) * args.old_dataset_beishu_uncertain)
                                                            if int(
                                                                len(train_dataset) * args.old_dataset_beishu_uncertain) <
                                                               len(train_dataset_old_p1) else len(train_dataset_old_p1))
        train_dataset_old_p1_subset = CustomSubset(train_dataset_old_p1, train_dataset_old_p1_subset_indices)
        # 合并新数据和旧数据，生成用于certain更新的新数据集，后面就用这个了
        new_train_dataset = ConcatDataset([train_dataset, train_dataset_old_p1_subset])

        sample_weights = get_sample_weights_for_multiple_dataset([train_dataset, train_dataset_old_p1_subset])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader_weighted = DataLoader(new_train_dataset, batch_size=config.batch_size, sampler=sampler)
        train_loader_noweight = DataLoader(new_train_dataset, batch_size=config.batch_size, shuffle=False)

        # 适用训练初始模型的数据，约束新的特征空间和之前不是特别偏移
        print('[INFO] Load old p1 dataset')
        logging.info('[INFO] Load old p1 dataset')
        val_dataset_p1 = StaSeqTrafficNormalizedDataset(
            [os.path.join(config.ori_file_root, i) for i in config.test_files_p1], config.class_dict,
            config.detail_class_dict, config, False, config.ori_scaler_path
        )
        val_p1_part_sampler = RandomSampler(val_dataset_p1, num_samples=3000,
                                            generator=torch.Generator().manual_seed(42))
        # val_p1_part_loader = DataLoader(val_dataset_p1, batch_size=config.batch_size, sampler=val_p1_part_sampler)
        val_p1_part_loader = DataLoader(val_dataset_p1, batch_size=config.batch_size, shuffle=True)
        val_p1_part_loader_cycle = itertools.cycle(val_p1_part_loader)

        # 用于验证差异
        val_p1_part_sampler_2 = RandomSampler(val_dataset_p1, num_samples=3000,
                                              generator=torch.Generator().manual_seed(50))
        val_p1_part_loader_2 = DataLoader(val_dataset_p1, batch_size=config.batch_size, sampler=val_p1_part_sampler_2)

        print('[INFO] Begin update model with uncertain')
        logging.info('[INFO] Begin update model with uncertain')
        if args.uncertain_update_way == 'dist_each':
            print('[INFO] Update uncertain with dist each constrain')
            logging.info('[INFO] Update uncertain with dist each constrain')
            if args.train_model=='yes':
                uncertain_train_update_model_uc(config, train_loader_weighted, val_p1_part_loader_cycle,
                                            os.path.join(config.certain_model_root, config.used_certain_seq_model),
                                            os.path.join(config.certain_model_root, config.used_certain_sta_model),
                                            config.uncertain_model_root,
                                            config.old_p2cer_seq_centers_path, config.old_p2cer_sta_centers_path)
            else:
                print('[INFO] Skip model train')
                logging.info('[INFO] Skip model train')
        elif args.uncertain_update_way == 'dist_center':
            print('[INFO] Update uncertain with dist center constrain')
            logging.info('[INFO] Update uncertain with dist center constrain')
            if args.train_model=='yes':
                uncertain_train_update_model_to_center_uc(config, train_loader_weighted, val_p1_part_loader_cycle,
                                                      os.path.join(config.certain_model_root,
                                                                   config.used_certain_seq_model),
                                                      os.path.join(config.certain_model_root,
                                                                   config.used_certain_sta_model),
                                                      config.uncertain_model_root,
                                                      config.old_p2cer_seq_centers_path,
                                                      config.old_p2cer_sta_centers_path)
            else:
                print('[INFO] Skip model train')
                logging.info('[INFO] Skip model train')

        if args.use_different_cal_par == 'yes':
            # 如果使用不同的结果，那这里就重置一下dataset noweight
            train_dataset_old_p1_subset_indices_2 = random.sample(range(0, len(train_dataset_old_p1)),
                                                                  len(train_dataset_old_p1) // 3)
            train_dataset_old_p1_subset_2 = CustomSubset(train_dataset_old_p1, train_dataset_old_p1_subset_indices_2)
            # 合并新数据和旧数据，生成用于certain更新的新数据集，后面就用这个了
            new_train_dataset_2 = ConcatDataset([train_dataset, train_dataset_old_p1_subset_2])
            train_loader_noweight = DataLoader(new_train_dataset, batch_size=config.batch_size, shuffle=False)

        get_uncertain_related(config, train_loader_noweight,
                              os.path.join(config.uncertain_model_root, config.used_seq_model_uncertain_update),
                              os.path.join(config.uncertain_model_root, config.used_sta_model_uncertain_update),
                              config.uncertain_new_save_data_root)

        # print('[INFO] Update distribution')
        # logging.info('[INFO] Update distribution')
        # combine_two_distribution_uncertain(config.old_save_data_root, config.uncertain_new_save_data_root,
        #                                    config.uncertain_updated_save_data_root)

        # ===========================
        # 验证更新后的模型的偏移，及在各个验证数据集上的性能
        # ===========================
        print('[INFO] Validate latent drift after uncertain update')
        logging.info('[INFO] Validate latent drift after uncertain update')
        validate_drift(val_p1_part_loader_2, os.path.join(config.certain_model_root, config.used_certain_seq_model),
                       os.path.join(config.certain_model_root, config.used_certain_sta_model),
                       os.path.join(config.uncertain_model_root, config.used_seq_model_uncertain_update),
                       os.path.join(config.uncertain_model_root, config.used_sta_model_uncertain_update),
                       config, config.uncertain_driftvalidate_save_data_root)

        print('[INFO] Load test data p1val')
        logging.info('[INFO] Evaluate model')
        logging.info('[INFO] Load test data p1val')
        test_dataset_p1val = StaSeqTrafficNormalizedDataset(
            [os.path.join(config.ori_file_root, i) for i in config.test_files_p1val],
            config.class_dict, config.detail_class_dict, config, False, config.ori_scaler_path)
        test_loader_p1val = DataLoader(test_dataset_p1val, batch_size=config.batch_size, shuffle=False)
        print('[INFO] Evaluate uncertain trained model, use part1val')
        logging.info('[INFO] Evaluate uncertain trained model, use part1val')
        eval_model_stage2('update part2 uncertain, test p1val', config, test_loader_p1val, config.uncertain_model_root,
                          config.uncertain_updated_test_result_root_p1val, config.uncertain_new_save_data_root,
                          config.uncertain_update_test_save_data_root_p1val, config.used_seq_model_uncertain_update,
                          config.used_sta_model_uncertain_update)

        print('[INFO] Load test data p2valnew')
        logging.info('[INFO] Evaluate model')
        logging.info('[INFO] Load test data p2valnew')
        test_dataset_p2valnew = StaSeqTrafficNormalizedDataset(
            [os.path.join(config.ori_file_root, i) for i in config.test_files_p2valnew],
            config.class_dict, config.detail_class_dict, config, False, config.ori_scaler_path)
        test_loader_p2valnew = DataLoader(test_dataset_p2valnew, batch_size=config.batch_size, shuffle=False)
        print('[INFO] Evaluate uncertain trained model, use part2valnew')
        logging.info('[INFO] Evaluate uncertain trained model, use part2valnew')
        eval_model_stage2('update part2 uncertain, test p2valnew', config, test_loader_p2valnew, config.uncertain_model_root,
                          config.uncertain_updated_test_result_root_p2valnew, config.uncertain_new_save_data_root,
                          config.uncertain_update_test_save_data_root_p2valnew, config.used_seq_model_uncertain_update,
                          config.used_sta_model_uncertain_update)

        print('[INFO] Load test data p4valnew')
        logging.info('[INFO] Evaluate model')
        logging.info('[INFO] Load test data p4valnew')
        test_dataset_p4valnew = StaSeqTrafficNormalizedDataset(
            [os.path.join(config.ori_file_root, i) for i in config.test_files_p4valnew],
            config.class_dict, config.detail_class_dict, config, False, config.ori_scaler_path)
        test_loader_p4valnew = DataLoader(test_dataset_p4valnew, batch_size=config.batch_size, shuffle=False)
        print('[INFO] Evaluate uncertain trained model, use part4valnew')
        logging.info('[INFO] Evaluate uncertain trained model, use part4valnew')
        eval_model_stage2('update part2 uncertain, test p4valnew', config, test_loader_p4valnew, config.uncertain_model_root,
                          config.uncertain_updated_test_result_root_p4valnew, config.uncertain_new_save_data_root,
                          config.uncertain_update_test_save_data_root_p4valnew, config.used_seq_model_uncertain_update,
                          config.used_sta_model_uncertain_update)

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   6_after_update_eval.py
@Contact :   hanxueying@iie.ac.cn
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2024/8/21 11:26   xueying      1.0       
'''

import sys
import os
import importlib
import logging
from datetime import datetime
import argparse

sys.path.append('/home/driftDetect/newcode0625')
sys.path.append('/newcode0625')

ids_dataset_config = importlib.import_module('dataconfig_ids2018')
generate_co_certain_file = importlib.import_module('myutil').generate_co_certain_file
concat_ori_files = importlib.import_module('myutil').concat_ori_files
generate_uncertain_data = importlib.import_module('myutil').generate_uncertain_data
split_latent = importlib.import_module('myutil').split_latent
cluster_pair_data_all = importlib.import_module('myutil').cluster_pair_data_all
get_all_eval_res = importlib.import_module('myutil').get_all_eval_res
fine_split_certain = importlib.import_module('myutil').fine_split_certain
fine_split_uncertain = importlib.import_module('myutil').fine_split_uncertain
split_get_uncertain_data = importlib.import_module('myutil').split_get_uncertain_data
save_certain_uncertain_with_class_for_train = importlib.import_module(
    'myutil').save_certain_uncertain_with_class_for_train

parser = argparse.ArgumentParser(description='step 4 evaluate after update',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--config_folder', type=str)
parser.add_argument('--data_folder', type=str)
parser.add_argument('--used_config', type=str, default='config1')
parser.add_argument('--ori_seq_num', type=int, default=266)
parser.add_argument('--ori_sta_num', type=int, default=266)
parser.add_argument('--certain_seq_num', type=int, default=10)
parser.add_argument('--certain_sta_num', type=int, default=10)
parser.add_argument('--uncertain_seq_num', type=int, default=10)
parser.add_argument('--uncertain_sta_num', type=int, default=10)
parser.add_argument('--idcl_eps', type=float, default=0.5)
parser.add_argument('--idcl_minpts', type=int, default=10)
parser.add_argument('--edge_div_node_threshold', type=float, default=5)
parser.add_argument('--edge_div_com_threshold', type=float, default=10)
parser.add_argument('--execute_folder_name', type=str, default='7_ids_script_new')
parser.add_argument('--update_part_name', type=str, default='p2')
parser.add_argument('--update_part_folder', type=str, default='p1p2_uncertain')
parser.add_argument('--is_usesl', type=str, default='no')
parser.add_argument('--eval_part_name', type=str, default='p1val')
parser.add_argument('--old_dataset_beishu_uncertain', type=float,
                    default=1.0)  # 在更新数据的时候，旧数据集使用新数据集的多少倍（uncertain更新的时候）
parser.add_argument('--old_dataset_beishu_certain', type=float, default=1.0)  # 在certain更新的时候，旧数据是多少倍（主要是产生certain相关的路径）
parser.add_argument('--uncertain_update_way', type=str, default='dist_center')
parser.add_argument('--refine_uncertain', type=str, default='')

args = parser.parse_args()


class UpdateEvaluateConfig:
    def __init__(self, config_folder, data_folder, used_config, ori_seq_num, ori_sta_num, certain_seq_num,
                 certain_sta_num,
                 uncertain_seq_num, uncertain_sta_num, idcl_eps=0.5, idcl_minpts=10, edge_div_node_threshold=5,
                 edge_div_com_threshold=10,
                 execute_folder_name='7_ids_script_new',
                 update_part_name='p2', update_part_folder='p1p2_uncertain', is_usesl='no', eval_part_name='p1val',
                 old_dataset_beishu_uncertain=1.0, old_dataset_beishu_certain=1.0, uncertain_update_way='dist_center',
                 refine_uncertain=''):
        """
        :param update_part_name: 是哪部分数据更新的结果，p2或p4
        :param update_part_folder: 更新的文件夹是什么，可以指定去验证certain更新后或uncertain更新后的性能
        :param is_usesl:
        :param eval_part_name: 评估的是哪部分数据的性能
        """
        self.dataset = 'ids'
        self.dataset_config_name = used_config
        self.all95 = 'all'

        # 聚类时使用的信息
        self.idcl_eps = idcl_eps
        self.idcl_minpts = idcl_minpts
        self.edge_div_node_threshold = edge_div_node_threshold
        self.edge_div_com_threshold = edge_div_com_threshold

        self.codb = old_dataset_beishu_certain
        self.uodb = old_dataset_beishu_uncertain

        self.eval_part_name = eval_part_name

        if self.dataset == 'ids' and used_config == 'config1':
            self.dataset_config = ids_dataset_config.DatasetConfig1()

        if self.eval_part_name == 'p1val':
            self.test_normal_files = self.dataset_config.part1_normal_files_val
            self.test_attack_files = self.dataset_config.part1_attack_files_val
        elif self.eval_part_name == 'p2valnew':
            self.test_normal_files = self.dataset_config.part2_normal_files_val_new
            self.test_attack_files = self.dataset_config.part2_attack_files_val_new
        elif self.eval_part_name == 'p4valnew':
            self.test_normal_files = self.dataset_config.part4_normal_files_val_new
            self.test_attack_files = self.dataset_config.part4_attack_files_val_new

        # 确定certain的列
        self.cocertain_row = 'dist_co_certain'  # 看seq和sta的结果是否一致
        self.consider_unmatch = True  # 是否考虑seq和sta结果不一致的情况，true就是把不匹配的也视为uncertain
        self.used_uncertain_row = 'cla_md_recon_uncertain_3'
        self.regenerate_cocertain = False

        self.ori_file_root = f'/home/driftDetect/0_data/1_cicids2018/4_ids_all_run_data/{data_folder}'

        if update_part_folder.split('_')[-1] == 'uncertain':
            if uncertain_update_way == 'dist_center':
                if refine_uncertain == '':
                    self.tmp_mid_folder = f'uncertain_usesl_{is_usesl}_center_codb{self.codb}_uodb{self.uodb}_om{ori_seq_num}_{ori_sta_num}_cm{certain_seq_num}_{certain_sta_num}_un{uncertain_seq_num}_{uncertain_sta_num}'
                else:
                    self.tmp_mid_folder = f'uncertain_usesl_{is_usesl}_center{refine_uncertain}_codb{self.codb}_uodb{self.uodb}_om{ori_seq_num}_{ori_sta_num}_cm{certain_seq_num}_{certain_sta_num}_un{uncertain_seq_num}_{uncertain_sta_num}'
            else:
                if refine_uncertain == '':
                    self.tmp_mid_folder = f'uncertain_usesl_{is_usesl}_codb{self.codb}_uodb{self.uodb}_om{ori_seq_num}_{ori_sta_num}_cm{certain_seq_num}_{certain_sta_num}_un{uncertain_seq_num}_{uncertain_sta_num}'
                else:
                    self.tmp_mid_folder = f'uncertain_usesl_{is_usesl}{refine_uncertain}_codb{self.codb}_uodb{self.uodb}_om{ori_seq_num}_{ori_sta_num}_cm{certain_seq_num}_{certain_sta_num}_un{uncertain_seq_num}_{uncertain_sta_num}'
        elif update_part_folder.split('_')[-1] == 'certain':
            self.tmp_mid_folder = f'certain_usesl_{is_usesl}_codb{self.codb}_om{ori_seq_num}_{ori_sta_num}_cm{certain_seq_num}_{certain_sta_num}'
        else:
            print('[ERROR] Unmatch when generate tmp mid folder')
            self.tmp_mid_folder = ''

        self.result_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/{update_part_folder}/{config_folder}/{update_part_name}_updated/{self.tmp_mid_folder}/{eval_part_name}'
        self.test_save_data_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/save_data/{update_part_folder}/{config_folder}/{update_part_name}_updated/{self.tmp_mid_folder}/{eval_part_name}'

        # 生成的新的文件路径是怎么样的（在identify drift的时候会有选择95还是all的选项，这里就直接使用all的结果了）
        self.attack_seq_path = os.path.join(self.result_root, 'attack_test_result_ksigma_deep_new_seq.csv')
        self.attack_sta_path = os.path.join(self.result_root, 'attack_test_result_ksigma_deep_new_sta.csv')
        self.attack_dist_seq_path = os.path.join(self.result_root, 'attack_test_result_ksigma_seq.csv')
        self.attack_dist_sta_path = os.path.join(self.result_root, 'attack_test_result_ksigma_sta.csv')
        self.attack_cocertain_path = os.path.join(self.result_root, 'attack_cocertain.csv')

        self.normal_seq_path = os.path.join(self.result_root, 'normal_test_result_ksigma_deep_new_seq.csv')
        self.normal_sta_path = os.path.join(self.result_root, 'normal_test_result_ksigma_deep_new_sta.csv')
        self.normal_dist_seq_path = os.path.join(self.result_root, 'normal_test_result_ksigma_seq.csv')
        self.normal_dist_sta_path = os.path.join(self.result_root, 'normal_test_result_ksigma_sta.csv')
        self.normal_cocertain_path = os.path.join(self.result_root, 'normal_cocertain.csv')

        # 隐含特征是在一起的
        self.seq_latent_path = os.path.join(self.test_save_data_root, 'test_seqmid_info.pickle')
        self.sta_latent_path = os.path.join(self.test_save_data_root, 'test_stamid_info.pickle')
        self.attack_seq_latent_path = os.path.join(self.test_save_data_root, 'attack_test_seqmid_info.pickle')
        self.attack_sta_latent_path = os.path.join(self.test_save_data_root, 'attack_test_stamid_info.pickle')
        self.normal_seq_latent_path = os.path.join(self.test_save_data_root, 'normal_test_seqmid_info.pickle')
        self.normal_sta_latent_path = os.path.join(self.test_save_data_root, 'normal_test_stamid_info.pickle')

        if self.consider_unmatch:
            self.certain_normal_path = os.path.join(self.result_root,
                                                    f'normal_certain-{self.cocertain_row}-{self.used_uncertain_row}_counmatch.csv')
            self.uncertain_normal_path = os.path.join(self.result_root,
                                                      f'normal_uncertain-{self.cocertain_row}-{self.used_uncertain_row}_counmatch.csv')
            self.certain_attack_path = os.path.join(self.result_root,
                                                    f'attack_certain-{self.cocertain_row}-{self.used_uncertain_row}_counmatch.csv')
            self.uncertain_attack_path = os.path.join(self.result_root,
                                                      f'attack_uncertain-{self.cocertain_row}-{self.used_uncertain_row}_counmatch.csv')
        else:
            self.certain_normal_path = os.path.join(self.result_root,
                                                    f'normal_certain-{self.cocertain_row}-{self.used_uncertain_row}_nocounmatch.csv')
            self.uncertain_normal_path = os.path.join(self.result_root,
                                                      f'normal_uncertain-{self.cocertain_row}-{self.used_uncertain_row}_nocounmatch.csv')
            self.certain_attack_path = os.path.join(self.result_root,
                                                    f'attack_certain-{self.cocertain_row}-{self.used_uncertain_row}_nocounmatch.csv')
            self.uncertain_attack_path = os.path.join(self.result_root,
                                                      f'attack_uncertain-{self.cocertain_row}-{self.used_uncertain_row}_nocounmatch.csv')

        # 保存聚类后的簇的统计特征
        self.cluster_mid_file_root = os.path.join(self.result_root,
                                                  f'cluster_mid_file_eps{self.idcl_eps}_mp{self.idcl_minpts}')

        # 新数据就不保存了，就不放保存数据的路径了

        # =======
        # 设置log信息
        # =======

        logging.basicConfig(level=logging.DEBUG,
                            filename=f'log/stage3eval_{self.dataset}-{config_folder}.log',
                            format='%(message)s')
        logging.info('\n\n===================================================')
        logging.info('====== ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ======')
        logging.info('=====================================================')
        logging.info(f'- data: {self.dataset}')
        logging.info(f'- update_part_folder: {update_part_folder}, eval folder: {eval_part_name}')
        logging.info(f'- mid folder: {self.tmp_mid_folder}')
        logging.info(
            f'- cocertain_row: {self.cocertain_row}, used_uncertain_row: {self.used_uncertain_row}, consider unmatch: {self.consider_unmatch}')
        logging.info(
            f'- identify cluster: eps: {self.idcl_eps}, minpts: {self.idcl_minpts}, edge_div_node_thre: {self.edge_div_node_threshold}, edge_div_com_thre: {self.edge_div_com_threshold}')
        logging.info(f'- codb: {self.codb}, uodb: {self.uodb}')   # 8.25增加，用于增加关于它们的消融实验
        logging.info(f'- refine_uncertain: {refine_uncertain}')
        logging.info('--------')


if __name__ == '__main__':
    config = UpdateEvaluateConfig(args.config_folder, args.data_folder, args.used_config,
                                  args.ori_seq_num, args.ori_sta_num, args.certain_seq_num, args.certain_sta_num,
                                  args.uncertain_seq_num, args.uncertain_sta_num, args.idcl_eps, args.idcl_minpts,
                                  args.edge_div_node_threshold, args.edge_div_com_threshold, args.execute_folder_name,
                                  args.update_part_name, args.update_part_folder, args.is_usesl, args.eval_part_name,
                                  args.old_dataset_beishu_uncertain, args.old_dataset_beishu_certain,
                                  args.uncertain_update_way, args.refine_uncertain)

    # 将被判定为certain和uncertain的原始特征分别存储起来
    print('[INFO] Split get uncertain data')
    logging.info('[INFO] Split get uncertain data')
    split_get_uncertain_data(config.seq_latent_path, config.attack_seq_latent_path, config.normal_seq_latent_path,
                             config.sta_latent_path, config.attack_sta_latent_path, config.normal_sta_latent_path,
                             config.regenerate_cocertain, config.consider_unmatch, config.cocertain_row,
                             config.used_uncertain_row,
                             config.ori_file_root, config.result_root,
                             config.normal_seq_path, config.normal_sta_path, config.normal_cocertain_path,
                             config.test_normal_files,
                             config.normal_dist_seq_path, config.normal_dist_sta_path, config.certain_normal_path,
                             config.uncertain_normal_path,
                             config.attack_seq_path, config.attack_sta_path, config.attack_cocertain_path,
                             config.test_attack_files,
                             config.attack_dist_seq_path, config.attack_dist_sta_path, config.certain_attack_path,
                             config.uncertain_attack_path
                             )

    # 对uncertain的数据进行聚类，判定是攻击还是异常，并生成对整体的评价
    print('[INFO] Identify drift type')
    logging.info('[INFO] Identify drift type')
    cluster_pair_data_all(config.result_root, config.result_root,
                          os.path.basename(config.uncertain_normal_path),
                          os.path.basename(config.uncertain_attack_path), config.cluster_mid_file_root,
                          config.idcl_eps, config.idcl_minpts, config.edge_div_node_threshold,
                          config.edge_div_com_threshold)

    get_all_eval_res(config.certain_normal_path, config.certain_attack_path, config.uncertain_normal_path,
                     config.uncertain_attack_path)

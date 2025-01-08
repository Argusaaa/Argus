import sys
import os
import importlib
import logging
from datetime import datetime
import argparse

sys.path.append('/home/driftDetect/newcode0625')
sys.path.append('/newcode0625')
save_certain_uncertain_with_class_for_train = importlib.import_module(
    'myutil').save_certain_uncertain_with_class_for_train

parser = argparse.ArgumentParser(description='step 2&3 drift detection and identification',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--config_folder', type=str,
                    default='kmeans0708_config1_k1_dm0.05_qm10tm10_l50_weighted')  # 包含了使用的参数和使用的模型
parser.add_argument('--data_folder', type=str, default='kmeans0708_config1_k1_dm0.05_weighted')
parser.add_argument('--used_config', type=str, default='config1')
parser.add_argument('--train_part_name', type=str, default='part1')
parser.add_argument('--test_part_name', type=str, default='p2')
parser.add_argument('--test_part_name_prefix', type=str, default='')
parser.add_argument('--idcl_eps', type=float, default=0.5)
parser.add_argument('--idcl_minpts', type=int, default=10)
parser.add_argument('--edge_div_node_threshold', type=float, default=5)
parser.add_argument('--edge_div_com_threshold', type=float, default=10)
parser.add_argument('--execute_folder_name', type=str, default='7_ids_script_new')
parser.add_argument('--all95', type=str, default='all')

args = parser.parse_args()


class GetUpdateConfig:
    def __init__(self, config_folder, data_folder, used_config='config1', train_part_name='part1', test_part_name='p2',
                 test_part_name_prefix='', idcl_eps=0.5, idcl_minpts=10, edge_div_node_threshold=5,
                 edge_div_com_threshold=10, execute_folder_name='none', all95='all'):
        """
        :param config_folder: 类似 'kmeans0711_config1_k1_dm0.05_qm10.0tm10.0_l50_weighted_alq1.0_0.01_alt1.0_0.01_mq266t266'，可以由多个参数推导得到，包含训练时的参数以及使用的模型
        :param data_folder: 类似 'kmeans0625_config1_k2_dm0.05_weighted'
        :param used_config:
        :param train_part_name:  是由什么数据训练得到的模型生成的结果，如果是第一次训练就是part1，如果是后面更新的就是 p1p2_certain，或者p1p2_uncertain
        :param idcl_eps, idcl_minpts, edge_div_node_threshold, edge_div_com_threshold: 聚类参数
        """
        self.dataset = 'ids'
        self.dataset_config_name = used_config
        self.all95 = all95

        self.train_part_name = train_part_name
        self.test_part_name = test_part_name
        self.test_part_name_prefix = test_part_name_prefix

        # 确定certain的列
        self.cocertain_row = 'dist_co_certain'  # 看seq和sta的结果是否一致
        self.consider_unmatch = True  # 是否考虑seq和sta结果不一致的情况，true就是把不匹配的也视为uncertain
        self.used_uncertain_row = 'cla_md_recon_uncertain_3'

        # 在identify的时候聚类的参数
        self.idcl_eps = idcl_eps
        self.idcl_minpts = idcl_minpts

        # 在判断聚类得到的簇是否为恶意时的参数
        self.edge_div_node_threshold = edge_div_node_threshold
        self.edge_div_com_threshold = edge_div_com_threshold

        self.config_folder = config_folder
        self.data_folder = data_folder

        self.ori_file_root = f'/home/driftDetect/0_data/1_cicids2018/4_ids_all_run_data/{self.data_folder}'

        if len(self.test_part_name_prefix) == 0:
            self.result_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/{self.train_part_name}/{self.config_folder}/{self.test_part_name}'
        else:
            self.result_root = f'/home/driftDetect/newcode0625/{execute_folder_name}/result/{self.train_part_name}/{self.config_folder}/{self.test_part_name_prefix}/{self.test_part_name}'

        if self.consider_unmatch:
            self.certain_normal_path = os.path.join(self.result_root,
                                                    f'normal_certain-{self.cocertain_row}-{self.used_uncertain_row}-{self.all95}_counmatch.csv')
            self.uncertain_normal_path = os.path.join(self.result_root,
                                                      f'normal_uncertain-{self.cocertain_row}-{self.used_uncertain_row}-{self.all95}_counmatch.csv')
            self.certain_attack_path = os.path.join(self.result_root,
                                                    f'attack_certain-{self.cocertain_row}-{self.used_uncertain_row}-{self.all95}_counmatch.csv')
            self.uncertain_attack_path = os.path.join(self.result_root,
                                                      f'attack_uncertain-{self.cocertain_row}-{self.used_uncertain_row}-{self.all95}_counmatch.csv')
        else:
            self.certain_normal_path = os.path.join(self.result_root,
                                                    f'normal_certain-{self.cocertain_row}-{self.used_uncertain_row}-{self.all95}_nocounmatch.csv')
            self.uncertain_normal_path = os.path.join(self.result_root,
                                                      f'normal_uncertain-{self.cocertain_row}-{self.used_uncertain_row}-{self.all95}_nocounmatch.csv')
            self.certain_attack_path = os.path.join(self.result_root,
                                                    f'attack_certain-{self.cocertain_row}-{self.used_uncertain_row}-{self.all95}_nocounmatch.csv')
            self.uncertain_attack_path = os.path.join(self.result_root,
                                                      f'attack_uncertain-{self.cocertain_row}-{self.used_uncertain_row}-{self.all95}_nocounmatch.csv')

        # 更新数据重新保存，用于下一次训练
        self.new_certain_data_root = os.path.join(self.ori_file_root,
                                                  f'{self.test_part_name}_certain/{self.cocertain_row}-'
                                                  f'{self.used_uncertain_row}-eps{self.idcl_eps}minpts'
                                                  f'{self.idcl_minpts}-thre{self.edge_div_node_threshold}_{self.edge_div_com_threshold}')
        self.new_uncertain_data_root = os.path.join(self.ori_file_root,
                                                    f'{self.test_part_name}_uncertain/{self.cocertain_row}-'
                                                    f'{self.used_uncertain_row}-eps{self.idcl_eps}minpts'
                                                    f'{self.idcl_minpts}-thre{self.edge_div_node_threshold}_{self.edge_div_com_threshold}')
        if not os.path.exists(self.new_certain_data_root):
            os.makedirs(self.new_certain_data_root)
        if not os.path.exists(self.new_uncertain_data_root):
            os.makedirs(self.new_uncertain_data_root)



if __name__ == '__main__':
    print('[INFO] Save certain and uncertain test data with detail class label')  # 就一个步骤，不需要写log了
    config = GetUpdateConfig(config_folder=args.config_folder, data_folder=args.data_folder,
                             used_config=args.used_config,
                             train_part_name=args.train_part_name, test_part_name=args.test_part_name,
                             test_part_name_prefix=args.test_part_name_prefix,
                             idcl_eps=args.idcl_eps, idcl_minpts=args.idcl_minpts,
                             edge_div_node_threshold=args.edge_div_node_threshold,
                             edge_div_com_threshold=args.edge_div_com_threshold,
                             execute_folder_name=args.execute_folder_name)
    save_certain_uncertain_with_class_for_train(config.certain_normal_path, config.certain_attack_path,
                                                config.new_certain_data_root,
                                                config.uncertain_normal_path, config.uncertain_attack_path,
                                                config.new_uncertain_data_root)
    print(f'[INFO] new certain data root: {config.new_certain_data_root}')
    print(f'[INFO] new uncertain data root: {config.new_uncertain_data_root}')



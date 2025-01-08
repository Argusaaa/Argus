#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path

import pandas as pd


class Mydict:
    def __init__(self):
        self.class_dict = {
            'fix_1_0214_30min_9_00-9_30_benign.pickle': 0,
            'fix_15_0214_30min_16_00-16_30_benign.pickle': 0,
            'fix_12_0215_30min_14_30-15_00_benign.pickle': 0,
            'fix_15_0215_30min_16_00-16_30_benign.pickle': 0,
            'fix_7_0216_30min_12_00-12_30_benign.pickle': 0,
            'fix_16_0216_30min_16_30-17_00_benign.pickle': 0,
            'fix_2_0220_30min_9_30-10_00_benign.pickle': 0,
            'fix_15_0220_30min_16_00-16_30_benign.pickle': 0,
            'fix_17_0228_30min_17_00-17_30_benign.pickle': 0,
            'fix_1_0302_30min_9_00-9_30_benign.pickle': 0,

            'ids2017_Monday_60000.pickle': 0,
            'ids2017_Monday_690000.pickle': 0,

            'bot_sa_part1.pickle': 1,
            'bot_sa_part2.pickle': 1,
            'ddos_loic_http_sa_part1.pickle': 1,
            'ddos_loic_http_sa_part2.pickle': 1,
            'dos_goldeneye_part1.pickle': 1,
            'dos_goldeneye_part2.pickle': 1,
            'dos_slowhttp_part1.pickle': 1,
            'dos_slowhttp_part2.pickle': 1,
            'dos_hulk_part1.pickle': 1,
            'dos_hulk_part2.pickle': 1,
            'ftp_bruteforce_part1.pickle': 1,
            'ftp_bruteforce_part2.pickle': 1,
            'ssh_bruteforce_part1.pickle': 1,
            'ssh_bruteforce_part2.pickle': 1,
            'bruteforce_web.pickle': 1,
            'bruteforce_xss.pickle': 1,
            'sql_injection.pickle': 1,
            'bruteforce_web_1.pickle': 1,
            'bruteforce_xss_1.pickle': 1,
            'sql_injection_1.pickle': 1,
            'bruteforce_web_2.pickle': 1,
            'bruteforce_xss_2.pickle': 1,
            'sql_injection_2.pickle': 1,
            'bot_sa_part1_val.pickle': 1,
            'ddos_loic_http_sa_part1_val.pickle': 1,

            'ids2017_botnet_1.pickle': 1,
            'ids2017_bruteforce_1.pickle': 1,
            'ids2017_dos_1.pickle': 1,
            'ids2017_portscan_1.pickle': 1,
            'ids2017_webattack_1.pickle': 1,

            'ids2017_botnet_2.pickle': 1,
            'ids2017_bruteforce_2.pickle': 1,
            'ids2017_dos_2.pickle': 1,
            'ids2017_portscan_2.pickle': 1,
            'ids2017_webattack_2.pickle': 1,
        }
        self.detail_class_dict = {
            'fix_1_0214_30min_9_00-9_30_benign.pickle': 0,
            'fix_15_0214_30min_16_00-16_30_benign.pickle': 0,
            'fix_12_0215_30min_14_30-15_00_benign.pickle': 1,
            'fix_15_0215_30min_16_00-16_30_benign.pickle': 1,
            'fix_7_0216_30min_12_00-12_30_benign.pickle': 2,
            'fix_16_0216_30min_16_30-17_00_benign.pickle': 2,
            'fix_2_0220_30min_9_30-10_00_benign.pickle': 3,
            'fix_15_0220_30min_16_00-16_30_benign.pickle': 3,
            'fix_17_0228_30min_17_00-17_30_benign.pickle': 4,
            'fix_1_0302_30min_9_00-9_30_benign.pickle': 5,

            'ddos_loic_http_sa_part1.pickle': 5,
            'ddos_loic_http_sa_part2.pickle': 5,
            'dos_goldeneye_part1.pickle': 6,
            'dos_goldeneye_part2.pickle': 6,
            'dos_slowhttp_part1.pickle': 7,
            'dos_slowhttp_part2.pickle': 7,
            'dos_hulk_part1.pickle': 8,
            'dos_hulk_part2.pickle': 8,
            'ftp_bruteforce_part1.pickle': 9,
            'ftp_bruteforce_part2.pickle': 9,
            'ssh_bruteforce_part1.pickle': 10,
            'ssh_bruteforce_part2.pickle': 10,
            'bruteforce_web.pickle': 11,
            'bruteforce_xss.pickle': 12,
            'sql_injection.pickle': 13,
            'bruteforce_web_1.pickle': 11,
            'bruteforce_xss_1.pickle': 12,
            'sql_injection_1.pickle': 13,
            'bruteforce_web_2.pickle': 11,
            'bruteforce_xss_2.pickle': 12,
            'sql_injection_2.pickle': 13,

            'bot_sa_part1.pickle': 14,
            'bot_sa_part2.pickle': 14,

            'bot_sa_part1_val.pickle': 14,
            'ddos_loic_http_sa_part1_val.pickle': 5,

            'ids2017_Monday_60000.pickle': 15,
            'ids2017_botnet_1.pickle': 16,
            'ids2017_bruteforce_1.pickle': 17,
            'ids2017_dos_1.pickle': 18,
            'ids2017_portscan_1.pickle': 19,
            'ids2017_webattack_1.pickle': 20,

            'ids2017_Monday_690000.pickle': 15,
            'ids2017_botnet_2.pickle': 16,
            'ids2017_bruteforce_2.pickle': 17,
            'ids2017_dos_2.pickle': 18,
            'ids2017_portscan_2.pickle': 19,
            'ids2017_webattack_2.pickle': 20,
        }


class DatasetConfig1:
    def __init__(self, new_normal_file=None, new_attack_file=None):
        self.part1_normal_files = [
            'fix_1_0214_30min_9_00-9_30_benign.pickle',
            'fix_15_0214_30min_16_00-16_30_benign.pickle'
        ]

        self.part1_normal_files_val = [
            'fix_12_0215_30min_14_30-15_00_benign.pickle',
            'fix_15_0215_30min_16_00-16_30_benign.pickle'
        ]

        self.part1_normal_files_val_certain = self.part1_normal_files_val
        self.part1_normal_files_val_uncertain = []

        self.part1_attack_files = [
            'bot_sa_part1.pickle',
            'ddos_loic_http_sa_part1.pickle',
            'dos_goldeneye_part1.pickle',
            'dos_slowhttp_part1.pickle',
            'ftp_bruteforce_part1.pickle',
        ]

        self.part1_attack_files_val = [
            'bot_sa_part1_val.pickle',
            'ddos_loic_http_sa_part1_val.pickle',
            'dos_goldeneye_part2.pickle',
            'dos_slowhttp_part2.pickle',
            'ftp_bruteforce_part2.pickle'
        ]

        self.part2_normal_files = [
            'fix_7_0216_30min_12_00-12_30_benign.pickle',
            'fix_16_0216_30min_16_30-17_00_benign.pickle'
        ]

        self.part2_normal_files_val = [
            'fix_2_0220_30min_9_30-10_00_benign.pickle',
            'fix_15_0220_30min_16_00-16_30_benign.pickle',
        ]

        self.part2_attack_files = [
            'bot_sa_part2.pickle',
            'ddos_loic_http_sa_part2.pickle',
            'dos_hulk_part1.pickle',
            'ssh_bruteforce_part1.pickle',
        ]

        self.part2_attack_files_val = [
            'bot_sa_part2.pickle',
            'ddos_loic_http_sa_part2.pickle',
            'dos_hulk_part2.pickle',
            'ssh_bruteforce_part2.pickle',
        ]

        self.part2_normal_files_new = self.part2_normal_files
        self.part2_normal_files_val_new = self.part2_normal_files_val

        self.part2_normal_certain_files_new = self.part2_normal_files_new
        self.part2_normal_uncertain_files_new = []

        self.part2_normal_files_val_new_certain = self.part2_normal_files_val_new
        self.part2_normal_files_val_new_uncertain = []

        self.part2_attack_files_new = [
            'bot_sa_part2.pickle',
            'ddos_loic_http_sa_part2.pickle',
            'dos_hulk_part1.pickle',
            'ssh_bruteforce_part1.pickle',
            'bruteforce_web_1.pickle',
            'bruteforce_xss_1.pickle',
            'sql_injection_1.pickle'
        ]

        self.part2_attack_certain_files_new = [
            'bot_sa_part2.pickle',
            'ddos_loic_http_sa_part2.pickle',
        ]

        self.part2_attack_uncertain_files_new = [
            'dos_hulk_part1.pickle',
            'ssh_bruteforce_part1.pickle',
            'bruteforce_web_1.pickle',
            'bruteforce_xss_1.pickle',
            'sql_injection_1.pickle'
        ]

        self.part2_attack_files_val_new = [
            'bot_sa_part2.pickle',
            'ddos_loic_http_sa_part2.pickle',
            'dos_hulk_part2.pickle',
            'ssh_bruteforce_part2.pickle',
            'bruteforce_web_2.pickle',
            'bruteforce_xss_2.pickle',
            'sql_injection_2.pickle'
        ]

        self.part2_attack_files_val_new_certain = [
            'bot_sa_part2.pickle',
            'ddos_loic_http_sa_part2.pickle',
        ]

        self.part2_attack_files_val_new_uncertain = [
            'dos_hulk_part2.pickle',
            'ssh_bruteforce_part2.pickle',
            'bruteforce_web_2.pickle',
            'bruteforce_xss_2.pickle',
            'sql_injection_2.pickle'
        ]

        # part3的结果不再使用
        self.part3_normal_files = [
            'fix_17_0228_30min_17_00-17_30_benign.pickle'
        ]

        self.part3_attack_files = [
            'bruteforce_web.pickle',
            'bruteforce_xss.pickle',
            'sql_injection.pickle'
        ]

        self.part4_normal_files = [
            'ids2017_Monday_60000.pickle'
        ]

        self.part4_normal_files_val = [
            'ids2017_Monday_690000.pickle'
        ]

        self.part4_attack_files = [
            'ids2017_botnet_1.pickle',
            'ids2017_bruteforce_1.pickle',
            'ids2017_dos_1.pickle',
            'ids2017_portscan_1.pickle',
            'ids2017_webattack_1.pickle',
        ]

        self.part4_attack_files_val = [
            'ids2017_botnet_2.pickle',
            'ids2017_bruteforce_2.pickle',
            'ids2017_dos_2.pickle',
            'ids2017_portscan_2.pickle',
            'ids2017_webattack_2.pickle',
        ]

        self.part4_normal_files_new = [
            'fix_17_0228_30min_17_00-17_30_benign.pickle',
        ]

        self.part4_normal_files_val_new = [
            'fix_1_0302_30min_9_00-9_30_benign.pickle'
        ]



        self.part4_attack_files_new = self.part4_attack_files
        self.part4_attack_files_val_new = self.part4_attack_files_val

        self.part4_normal_certain_files_new = self.part4_normal_files_new
        self.part4_normal_uncertain_files_new = []
        self.part4_attack_certain_files_new = []
        self.part4_attack_uncertain_files_new = self.part4_attack_files_new

        self.part4_normal_files_val_new_certain = self.part4_normal_files_val_new
        self.part4_normal_files_val_new_uncertain = []
        self.part4_attack_files_val_new_certain = []
        self.part4_attack_files_val_new_uncertain = self.part4_attack_files_val_new

        self.my_dict = Mydict()
        self.class_dict = self.my_dict.class_dict
        self.detail_class_dict = self.my_dict.detail_class_dict

        if new_normal_file is not None:
            self.part1_normal_files = [new_normal_file]
            self.class_dict[new_normal_file] = 0
            self.detail_class_dict[new_normal_file] = 0
        if new_attack_file is not None:
            self.part1_attack_files = [new_attack_file]
            self.class_dict[new_attack_file] = 1
            self.detail_class_dict[new_attack_file] = -1


class DatasetConfig2:
    def __init__(self, new_normal_file=None):
        self.part1_normal_files = [
            'fix_1_0214_30min_9_00-9_30_benign.pickle',
            'fix_15_0214_30min_16_00-16_30_benign.pickle'
        ]

        self.part1_normal_files_val = [
            'fix_12_0215_30min_14_30-15_00_benign.pickle',
            'fix_15_0215_30min_16_00-16_30_benign.pickle'
        ]

        self.part1_attack_files = [
            'dos_goldeneye_part1.pickle',
            'dos_slowhttp_part1.pickle',
            'dos_hulk_part1.pickle',
            'ftp_bruteforce_part1.pickle',
            'ssh_bruteforce_part1.pickle',
        ]

        self.part1_attack_files_val = [
            'dos_goldeneye_part2.pickle',
            'dos_slowhttp_part2.pickle',
            'dos_hulk_part2.pickle',
            'ftp_bruteforce_part1.pickle',
            'ssh_bruteforce_part1.pickle',
        ]

        self.part2_normal_files = [
            'fix_7_0216_30min_12_00-12_30_benign.pickle',
            'fix_16_0216_30min_16_30-17_00_benign.pickle'
        ]

        self.part2_normal_files_val = [
            'fix_2_0220_30min_9_30-10_00_benign.pickle',
            'fix_15_0220_30min_16_00-16_30_benign.pickle',
        ]

        self.part2_attack_files = [
            'bot_sa_part1.pickle',
            'ddos_loic_http_sa_part1.pickle',
            'ftp_bruteforce_part2.pickle',
            'ssh_bruteforce_part2.pickle',
        ]

        self.part2_attack_files_val = [
            'bot_sa_part2.pickle',
            'ddos_loic_http_sa_part2.pickle',
            'ftp_bruteforce_part2.pickle',
            'ssh_bruteforce_part2.pickle',
        ]

        self.part3_normal_files = [
            'fix_17_0228_30min_17_00-17_30_benign.pickle'
        ]

        self.part3_attack_files = [
            'bruteforce_web.pickle',
            'bruteforce_xss.pickle',
            'sql_injection.pickle'
        ]

        self.part4_normal_files = [
            'ids2017_Monday_60000.pickle'
        ]

        self.part4_attack_files = [
            'ids2017_botnet.pickle',
            'ids2017_bruteforce.pickle',
            'ids2017_dos.pickle',
            'ids2017_portscan.pickle',
            'ids2017_webattack.pickle',
        ]

        self.my_dict = Mydict()
        self.class_dict = self.my_dict.class_dict
        self.detail_class_dict = self.my_dict.detail_class_dict

        if new_normal_file is not None:
            self.part1_normal_files = [new_normal_file]
            self.class_dict[new_normal_file] = 0
            self.detail_class_dict[new_normal_file] = 0

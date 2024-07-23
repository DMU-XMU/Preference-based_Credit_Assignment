import os
import shutil
import argparse
import logging
import time
import getpass
import numpy as np
from termcolor import colored
from beautifultable import BeautifulTable

def str2bool(value):
    value = str(value)
    if isinstance(value, bool):
       return value
    if value.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif value.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Get '+str(value.lower()))

def make_dir(dir_name, clear=True):
    if os.path.exists(dir_name):
        if clear:
            try: shutil.rmtree(dir_name)
            except: pass
            try: os.makedirs(dir_name)
            except: pass
    else:
        try: os.makedirs(dir_name)
        except: pass

def dir_ls(dir_path):
    dir_list = os.listdir(dir_path)
    dir_list.sort()
    return dir_list

def system_pause():
    getpass.getpass("Press Enter to Continue")

def get_arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def remove_color(key):
    for i in range(len(key)):
        if key[i]=='@':
            return key[:i]
    return key

def load_npz_info(file_path):
    return np.load(file_path)['info'][()]

class Logger:
    def __init__(self, name):
        make_dir('log',clear=False)
        make_dir('log/text',clear=False)
        if name is None: self.name = time.strftime('%Y-%m-%d-%H:%M:%S')
        else: self.name = name + time.strftime('-(%Y-%m-%d-%H:%M:%S)')

        log_file = 'log/text/'+self.name+'.log'
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.DEBUG)

        FileHandler = logging.FileHandler(log_file)
        FileHandler.setLevel(logging.DEBUG)
        self.logger.addHandler(FileHandler)

        StreamHandler = logging.StreamHandler()
        StreamHandler.setLevel(logging.INFO)
        self.logger.addHandler(StreamHandler)

        self.tabular_reset()

    def debug(self, *args): self.logger.debug(*args)
    def info(self, *args): self.logger.info(*args)  # default level
    def warning(self, *args): self.logger.warning(*args)
    def error(self, *args): self.logger.error(*args)
    def critical(self, *args): self.logger.critical(*args)

    def log_time(self, log_tag=''):
        log_info = time.strftime('%Y-%m-%d %H:%M:%S')
        if log_tag!='': log_info += ' '+log_tag
        self.info(log_info)

    def tabular_reset(self):
        self.keys = []
        self.colors = []
        self.values = {}
        self.counts = {}
        self.summary = []

    def tabular_clear(self):
        for key in self.keys:
            self.counts[key] = 0


    def tabular_show(self, log_tag=''):
        table = BeautifulTable()
        table_c = BeautifulTable()
        for key, color in zip(self.keys, self.colors):
            if self.counts[key]==0: value = ''
            elif self.counts[key]==1: value = self.values[key]
            else: value = self.values[key]/self.counts[key]
            key_c = key if color is None else colored(key, color, attrs=['bold'])
            table.append_row([key, value])
            table_c.append_row([key_c, value])

        def customize(table):
            table.set_style(BeautifulTable.STYLE_NONE)
            table.left_border_char = '|'
            table.right_border_char = '|'
            table.column_separator_char = '|'
            table.top_border_char = '-'
            table.bottom_border_char = '-'
            table.intersect_top_left = '+'
            table.intersect_top_mid = '+'
            table.intersect_top_right = '+'
            table.intersect_bottom_left = '+'
            table.intersect_bottom_mid = '+'
            table.intersect_bottom_right = '+'
            table.column_alignments[0] = BeautifulTable.ALIGN_LEFT
            table.column_alignments[1] = BeautifulTable.ALIGN_LEFT

        customize(table)
        customize(table_c)
        self.log_time(log_tag)
        self.debug(table)
        print(table_c)

    def save_npz(self, info, info_name, folder, subfolder=''):
        make_dir('log/'+folder,clear=False)
        make_dir('log/'+folder+'/'+self.name,clear=False)
        if subfolder!='':
            make_dir('log/'+folder+'/'+self.name+'/'+subfolder,clear=False)
            save_path = 'log/'+folder+'/'+self.name+'/'+subfolder
        else:
            save_path = 'log/'+folder+'/'+self.name
        np.savez(save_path+'/'+info_name+'.npz',info=info)

def get_logger(name=None):
    return Logger(name)

# -*- coding: utf-8 -*-
"""
AI-PAAS ,Ryerson Univesity

Created on Tue Sep 17 12:19:06 2019

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student

"""
# =================================================================================================
# Used to provide input for CMAPSS
# =================================================================================================

import pandas as pd


class cMAPSS:

    @classmethod
    def read_rul(cls, i):

        RUL_input = pd.read_csv(f"{cls.datapath}RUL_FD00{i}.txt", header=None, names=['RUL'])
        return RUL_input

    @classmethod
    def read_data(cls, test_or_train, i):

        if test_or_train == 'test' or test_or_train == 'train':

            d_var = ['Engine ID', 'Cycles'] + cls.OpCond_names + cls.Sensor_names + ['d1', 'd2']
            val_input = pd.read_csv(f"{cls.datapath}{test_or_train}_FD00{i}.txt", " ", header=None, names=d_var)
            val_input.drop(columns=['d1', 'd2'], inplace=True)

            return val_input
        else:
            raise Exception('Invalid File Name')

    @classmethod
    def set_datapath(cls, datapath):
        cls.datapath = datapath

    @classmethod
    def getall_data(cls):

        cls.RUL_input = cls.read_rul(1)
        cls.RUL_input['DataSet'] = ['FD_001'] * RUL_input.shape[0]

        cls.Train_input = cls.read_data('train', 1)
        cls.Train_input['DataSet'] = ['FD_001'] * Train_input.shape[0]

        cls.Test_input = cls.read_data('test', 1)
        cls.Test_input['DataSet'] = ['FD_001'] * Test_input.shape[0]

        for i in range(2, cls.NoOfDS + 1):
            d_var = cls.read_rul(i)
            cls.d_var['DataSet'] = [f'FD_00{i}'] * d_var.shape[0]
            cls.RUL_input = pd.concat([cls.RUL_input, d_var])

            d_var = cls.read_data('train', i)
            cls.d_var['DataSet'] = [f'FD_00{i}'] * d_var.shape[0]
            cls.Train_input = pd.concat([cls.Train_input, d_var])

            d_var = cls.read_data('test', i)
            cls.d_var['DataSet'] = [f'FD_00{i}'] * d_var.shape[0]
            cls.Test_input = pd.concat([cls.Test_input, d_var])

    @classmethod
    def get_data(cls, i):

        cls.RUL_input = cls.read_rul(i)
        cls.Train_input = cls.read_data('train', i)
        cls.Test_input = cls.read_data('test', i)

    # %%

    def __new__(self):
        raise Exception("Cannot Create Object")

    datapath = './DLRADO/CMAPSSData/NASA_RUL_-CMAPS--master/'  # default value
    NoOfSen = 21
    NoOfOPCo = 3
    NoOfDS = 4  # No of Datasets
    Sensor_names = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc',
                    'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
    OpCond_names = ['Altitude', 'Mach Number', 'TRA']


# %%

if __name__ == '__main__':
    cMAPSS.get_data(1)

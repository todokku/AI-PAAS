"""
AI-PAAS ,Ryerson Univesity

@author: 
    Tejas Janardhan
    AI-PAAS Phd Student

"""
import pandas as pd


class CMAPSS:
    """
    Used to provide input for CMAPSS
    """

    def __init__(self, ds_no):

        self.ds_no = ds_no

        self.datapath = '../CMAPSSData/'  # default value
        self.Sensor_names = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf',
                             'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
        self.OpCond_names = ['Altitude', 'Mach Number', 'TRA']

    def _read_rul(self):

        RUL_input = pd.read_csv(f"{self.datapath}RUL_FD00{self.ds_no}.txt", header=None, names=['RUL'])
        return RUL_input

    def _read_data(self, isTrain):

        if isTrain:
            dataset_type = 'train'
        else:
            dataset_type = 'test'

        temp = ['Engine ID', 'Cycles'] + self.OpCond_names + self.Sensor_names + ['d1', 'd2']
        val_input = pd.read_csv(f"{self.datapath}{dataset_type}_FD00{self.ds_no}.txt", " ", header=None, names=temp)
        val_input = val_input.drop(columns=['d1', 'd2'])

        return val_input

    def get_data(self):

        self.RUL_input = self._read_rul()
        self.Train_input = self._read_data(True)
        self.Test_input = self._read_data(False)


if __name__ == '__main__':
    cmapss = CMAPSS(3)
    cmapss.get_data()

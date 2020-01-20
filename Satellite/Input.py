"""
Input for the Satellite Data

"""

import pandas as pd

"""
Column Info

Column 1 to 4 - Speed in RPM
Column 5 to 8 - Command Torque in Nm
Column 9 to 12 - Rotor Temperature in Celsius
Column 13 to 16 - Friction Torque in Nm
Column 17 to 19 - Satellite Attitude Error
"""


class Satellite:

    def __init__(self, year):
        self.dir_path = '../SatelliteData/Nasa-Kepler-Mission'
        self.year = year
        self.datapath = []
        if self.year == '2009':
            self.datapath.append(self.dir_path + '/2009-data/2009_065_to_2009_165_SPD_TCMD_MTRT.csv')
            self.datapath.append(self.dir_path + '/2009-data/2009_065_to_2009_165_TRQF_ATTERR.csv')

        if self.year == '2012':
            self.datapath.append(self.dir_path + '/2012_001_to_2012_100/2012_001_to_2012_100_SPD_TCMD_MTRT.csv')
            self.datapath.append(self.dir_path + '/2012_001_to_2012_100/2012_001_to_2012_100_TRQF_ATTERR.csv')
            self.datapath.append(self.dir_path + '/2012_100_to_2010_180/2012_100_to_2012_180_SPD_TCMD_MTRT.csv')
            self.datapath.append(self.dir_path + '/2012_100_to_2010_180/2012_100_to_2012_180_TRQF_ATTERR.csv')

            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.180-211.RWxSP.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.180-211.RWxTCMD.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.180-211.RWxMTRT.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.180-211.RWxTRQF.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.180-211.ATTERRx.txt')

            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.211-242.SPD.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.211-242.TCMD.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.211-242.MTRT.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.211-242.TRQF.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.211-242.ATTERR.txt')

            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.242-279.SPD.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.242-279.TCMD.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.242-279.MTRT.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.242-279.TRQF.txt')
            self.datapath.append(self.dir_path + '/2012_180_to_2012_279/2012.242-279.ATTERR.txt')

        if self.year == '2013':
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.028-066.SPD.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.028-066.TCMD.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.028-066.MTRT.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.028-066.TRQF.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.028-066.ATTERR.txt')

            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.065-099.SPD.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.065-099.TCMD.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.065-099.MTRT.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.065-099.TRQF.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.065-099.ATTERR.txt')

            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.098-125.SPD.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.098-125.TCMD.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.098-125.MTRT.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.098-125.TRQF.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.098-125.ATTERR.txt')

            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.124-135.SPD.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.124-135.TCMD.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.124-135.MTRT.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.124-135.TRQF.txt')
            self.datapath.append(self.dir_path + '/2013_028_to_2013_135/2013.124-135.ATTERR.txt')

    def read_data(self):
        if self.year == '2009':
            self._read_data_2009()
        elif self.year == '2012':
            self._read_data_2012()
        elif self.year == '2013':
            self._read_data_2013()

    def _read_data_2009(self):
        self.input_data = pd.read_csv(self.datapath[0])
        temp = pd.read_csv(self.datapath[1])
        self.input_data = pd.concat([self.input_data.iloc[:, 1:], temp.iloc[:, 1:]], axis=1)

        self.input_data = self.input_data.iloc[3:, :]  # 3 Rows to remove
        self.input_data = self.input_data.apply(pd.to_numeric)

    def _read_data_2012(self):

        self.input_data = pd.read_csv(self.datapath[0])
        temp = pd.read_csv(self.datapath[1])
        self.input_data = pd.concat([self.input_data.iloc[1:, 1:], temp.iloc[1:, 1:]], axis=1)

        temp = pd.read_csv(self.datapath[2])
        temp_1 = pd.read_csv(self.datapath[3])
        temp = pd.concat([temp.iloc[1:, 1:], temp_1.iloc[1:, 1:]], axis=1)

        self.input_data = pd.concat([self.input_data, temp])

        for i in range(3):

            temp = pd.read_csv(self.datapath[4 + i * 5], engine='python')
            temp = temp.iloc[1:, 1:]  # Removing First Row and First Column
            for j in range(1, 5):
                temp_1 = pd.read_csv(self.datapath[4 + i * 5 + j], engine='python')
                temp = pd.concat([temp, temp_1.iloc[1:, 1:]], axis=1)

            self.input_data = pd.concat([self.input_data, temp])

        self.input_data = self.input_data.apply(pd.to_numeric, errors='coerce')

    def _read_data_2013(self):

        for i in range(4):

            temp = pd.read_csv(self.datapath[i * 5], engine='python')
            temp = temp.iloc[1:, 1:]
            for j in range(1, 5):
                temp_1 = pd.read_csv(self.datapath[i * 5 + j], engine='python')
                temp = pd.concat([temp, temp_1.iloc[1:, 1:]], axis=1)

            if i == 0:
                self.input_data = temp
            else:
                self.input_data = pd.concat([self.input_data, temp])

        self.input_data = self.input_data.apply(pd.to_numeric)


if __name__ == '__main__':
    inputs = Satellite('2013')

    inputs.read_data()

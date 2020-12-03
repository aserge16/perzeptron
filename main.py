from mlp import Perzeptron
import numpy as np
import re


#TODO: Turn this into script


def get_data(file_name):
        with open(file_name, 'r') as fh:
            lines = fh.readlines()
        data = {}
        P, N, M = [int(d) for d in re.findall(r'\b\d+\b', lines[1])]
        data['P'], data['N'], data['M'] = P, N, M

        input_data, teacher_data = list(), list()
        for line in lines[2:2+P]:
            nums = [float(n) for n in re.findall(r'[-+]?\d*\.\d+|\d+', line)]
            input_data.append(np.array(nums[:N]))
            teacher_data.append(np.array(nums[N:]))

        data['input'], data['teacher'] = input_data, teacher_data
        return data

if __name__=='__main__':
    fh = 'PA-B_training_data_03.txt'
    data = get_data(fh)
    x = Perzeptron([4,4,2])
    x.train(data['input'], data['teacher'])
# Once-for-All: Train One Network and Specialize it for Efficient Deployment on Diverse Hardware Platforms
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

# APQ: Joint Search for Network Architecture, Pruning and Quantization Policy
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Hanrui Wang, Yujun Lin, Song Han
# Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

# Winner Solution for 4th Low-Power Computer Vision Challenge (LPCVC)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import pickle
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--latency', type=int,
                    default=36)
parser.add_argument('--data_split', type=str,
                    default=None)
args, _ = parser.parse_known_args()

data_split = 'train' if args.data_split == 'train' else 'train+val'
latency = args.latency

if __name__ == '__main__':
    pth_path = 'finetune/{}/{}ms/init'.format(data_split, latency)
    tfinit_path = 'finetune/{}/{}ms/tfinit'.format(data_split, latency)

    pth = torch.load(pth_path, map_location='cpu')
    new_dict = {}
    for key in pth['state_dict']:
        v = pth['state_dict'][key]
        if (key[-len('depth_conv/conv/weight'):] == 'depth_conv.conv.weight'):
            v = v.permute(2, 3, 0, 1)
        elif (key[-len('conv/weight'):] == 'conv.weight'):
            v = v.permute(2, 3, 1, 0)
        if (key == 'classifier.linear.weight'):
            v = v.permute(1, 0)
        new_dict[key.replace('.', '/')] = v.numpy()

    pickle.dump(new_dict, open(tfinit_path, 'wb'))

    tfinit_1001_path = 'finetune/{}/{}ms/1001.tfinit'.format(data_split, latency)


    def convert_name(key):
        ans = key.replace('weight', 'weights')
        ans = ans.replace('bn/weights', 'BatchNorm/gamma')
        ans = ans.replace('bn/bias', 'BatchNorm/beta')
        ans = ans.replace('bn/', 'BatchNorm/')
        ans = ans.replace('first_conv/conv', 'first_conv')
        ans = ans.replace('first_conv', 'Conv')
        ans = ans.replace('point_linear', 'project')
        ans = ans.replace('depth_conv', 'depthwise')
        ans = ans.replace('mobile_inverted_conv/', '')
        ans = ans.replace('inverted_bottleneck/', 'expand/')

        ans = ans.replace('depthwise/conv/weights', 'depthwise/depthwise_weights')
        ans = ans.replace('conv/weights', 'weights')

        ans = ans.replace('blocks/0', 'expanded_conv')
        ans = ans.replace('blocks/', 'expanded_conv_')
        for i in range(17):  # 5~21 3~19  1~17
            if ans.find(str(i + 5)) >= 0:
                ans = ans.replace(str(i + 5), str(i + 3))
                break
        ans = ans.replace('feature_mix_layer', 'Conv_1')
        ans = ans.replace('classifier/linear', 'Logits/Conv2d_1c_1x1')
        ans = ans.replace('bias', 'biases')
        return ans


    tfinit = pickle.load(open(tfinit_path, 'rb'))
    tot = 0
    newdict = {}
    for key in tfinit:
        cn = key
        tmp = tfinit[key]
        tmp2 = tmp
        if (cn.find('classifier/linear') >= 0):
            if cn.find('weight') >= 0:
                tmp = np.zeros((1, 1, tmp2.shape[0], tmp2.shape[1] + 1))
                tmp[0, 0, :, 1:] = tmp2
                tmp[0, 0, :, 0] = tmp2[:, [1]].reshape([-1])
            else:
                tmp = np.zeros((tmp2.shape[0] + 1))
                idx = np.argmin(tmp2)
                tmp[1:] = tmp2
                tmp[0] = tmp2[idx]
        newdict[cn] = tmp
        tot = tot + 1

    pickle.dump(newdict, open(tfinit_1001_path, 'wb'))

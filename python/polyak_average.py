#!/usr/bin/env python

import numpy as np
import os.path as osp
import sys
from argparse import ArgumentParser

import caffe


def main(args):
    if args.weight_files is not None:
        weight_files = args.weight_files
    else:
        weight_files = [args.weight_prefix + '_iter_{}.caffemodel'.format(it)
                        for it in args.iter_range]
    net = caffe.Net(args.model, weight_files[0], caffe.TEST)
    count = {param_name: 1 for param_name in net.params.keys()}
    for weight_file in weight_files[1:]:
        tmp = caffe.Net(args.model, weight_file, caffe.TEST)
        for param_name in np.intersect1d(net.params.keys(), tmp.params.keys()):
            count[param_name] += 1
            for w, v in zip(net.params[param_name], tmp.params[param_name]):
                w.data[...] += v.data
    for param_name in net.params:
        if count[param_name] == 0: continue
        for w in net.params[param_name]:
            w.data[...] /= count[param_name]
    net.save(args.output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model', help="Net definition prototxt")
    parser.add_argument('output', help="Path for saving the output")
    parser.add_argument('--weight-files', type=str, nargs='+',
        help="A list of caffemodels")
    parser.add_argument('--weight-prefix', help="Prefix of caffemodels")
    parser.add_argument('--iter-range',
        help="Iteration range complementary with the prefix. In the form of "
             "(begin, end, step), where begin is inclusive while end is "
             "exclusive.")
    args = parser.parse_args()
    if args.weight_files is None and \
            (args.weight_prefix is None or args.iter_range is None):
        raise ValueError("Must provider either weight files or weight prefix "
                         "and iter range.")
    if args.iter_range is not None:
        args.iter_range = eval('xrange' + args.iter_range)
    main(args)

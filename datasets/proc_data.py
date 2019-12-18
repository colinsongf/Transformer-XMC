#!/usr/bin/env python
# encoding: utf-8

import argparse
import h5py
import os
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
from tqdm import tqdm


# {"dataset": [N_trn, N_tst, data_dim, label_size]}
# WARNING: Eurlex-4K is not identical to Manik Repo. We use AttentionXML's version.
# WARNING: Wiki-500K is not identical to Manik Repo. We use AttentionXML's version.
# See Table 1 of You et al. NIPS 2019 paper (arXiv:1811.01727v3)
DATA_STATS = {"Eurlex-4K": [15449, 3865, 186104, 3956],
              "Wiki10-31K": [14146, 6616, 101938, 30938],
              "AmazonCat-13K": [1186239, 306782, 203882, 13330],
              "Amazon-670K": [490449, 153025, 135909, 670091],
              "Wiki-500K": [1779881, 769421, 2381304, 501070],
              "Amazon-3M": [1717899, 742507, 337067, 2812281]}


def list_to_csr(Y_list, K_in=None):
  rows, cols, vals = [], [], []
  for i, ys in enumerate(Y_list):
    rows += [i] * len(ys)
    cols += list(map(int, ys))
    vals += [1] * len(ys)
  N = max(rows) + 1
  if K_in is None:
    K = max(cols) + 1
  else:
    K = K_in
  Y = sp.csr_matrix((vals, (rows, cols)), shape=(N,K))
  return Y


def main(args):
  # load train.txt and test.txt (should be libsvm format)
  # datasets downloaded from Manik XMC Repo
  dataset = args.dataset
  trn_path = "./{}/train.txt".format(dataset)
  tst_path = "./{}/test.txt".format(dataset)
  assert dataset in DATA_STATS.keys(), "unknown dataset {}".format(dataset)
  assert os.path.exists(trn_path), "trn_path does not exist {}".format(trn_path)
  assert os.path.exists(tst_path), "tst_path does not exist {}".format(tst_path)

  print("loading trn_path {}".format(trn_path))
  X_trn, Y_list = load_svmlight_file(trn_path, n_features=None,
                                     zero_based="auto", multilabel=True)
  Y_trn = list_to_csr(Y_list, K_in=None)

  print("loading tst_path {}".format(tst_path))
  X_tst, Y_list = load_svmlight_file(tst_path, n_features=X_trn.shape[1],
                                     zero_based="auto", multilabel=True)
  Y_tst = list_to_csr(Y_list, K_in=Y_trn.shape[1])

  # verify data statistics
  assert X_trn.shape[0] == DATA_STATS[dataset][0], "N_trn={} != {}".format(X_trn.shape[0], DATA_STATS[dataset][0])
  assert X_tst.shape[0] == DATA_STATS[dataset][1], "N_tst={} != {}".format(X_tst.shape[0], DATA_STATS[dataset][1])
  assert X_trn.shape[1] == DATA_STATS[dataset][2], "D_trn={} != {}".format(X_trn.shape[1], DATA_STATS[dataset][2])
  assert Y_trn.shape[1] == DATA_STATS[dataset][3], "L_trn={} != {}".format(Y_trn.shape[1], DATA_STATS[dataset][3])

  # l2 normalized each row of tf-idf matrix
  X_trn = normalize(X_trn, axis=1, norm='l2')
  X_tst = normalize(X_tst, axis=1, norm='l2')

  # save as npz
  sp.save_npz("./{}/X.trn.npz".format(dataset), X_trn)
  sp.save_npz("./{}/Y.trn.npz".format(dataset), Y_trn)
  sp.save_npz("./{}/X.tst.npz".format(dataset), X_tst)
  sp.save_npz("./{}/Y.tst.npz".format(dataset), Y_tst)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='Eurlex-4K', type=str, help='dataset')
	args = parser.parse_args()
	print(args)
	main(args)

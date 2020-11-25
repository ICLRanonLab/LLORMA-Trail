import os
import random

import numpy as np

# from ..configs import *


def _make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


class DatasetManager:
    KIND_MOVIELENS_100K = 'movielens-100k'
    KIND_MOVIELENS_1M = 'movielens-1m'
    KIND_MOVIELENS_10M = 'movielens-10m'
    KIND_MOVIELENS_20M = 'movielens-20m'
    KIND_NETFLIX = 'netflix'
    ###
    KIND_PTMATRIX_500 = 'ptmatrix-basetimes-500'
    KIND_PTMATRIX_1000 = 'ptmatrix-basetimes0'
    KIND_PTMATRIX_100_basetimes20_0_0 = 'ptmatrix-basetimes-20-0-0'
    KIND_PTMATRIX_100_basetimes20_0_1 = 'ptmatrix-basetimes-20-0-1'
    KIND_PTMATRIX_100_basetimes20_0_2 = 'ptmatrix-basetimes-20-0-2'
    KIND_PTMATRIX_100_basetimes20_0_3 = 'ptmatrix-basetimes-20-0-3'
    KIND_PTMATRIX_100_basetimes20_0_4 = 'ptmatrix-basetimes-20-0-4'
    KIND_PTMATRIX_100_basetimes20_0_5 = 'ptmatrix-basetimes-20-0-5'
    KIND_PTMATRIX_100_basetimes20_0_6 = 'ptmatrix-basetimes-20-0-6'
    KIND_PTMATRIX_100_basetimes20_0_7 = 'ptmatrix-basetimes-20-0-7'
    KIND_PTMATRIX_100_basetimes20_0_8 = 'ptmatrix-basetimes-20-0-8'
    KIND_PTMATRIX_100_basetimes20_0_9 = 'ptmatrix-basetimes-20-0-9'
    KIND_PTMATRIX_100_basetimes20_0_10 = 'ptmatrix-basetimes-20-0-10'
    KIND_PTMATRIX_100_basetimes20_0_11 = 'ptmatrix-basetimes-20-0-11'
    KIND_PTMATRIX_100_basetimes20_0_12 = 'ptmatrix-basetimes-20-0-12'
    KIND_PTMATRIX_100_basetimes20_0_13 = 'ptmatrix-basetimes-20-0-13'
    KIND_PTMATRIX_100_basetimes20_0_14 = 'ptmatrix-basetimes-20-0-14'
    KIND_PTMATRIX_100_basetimes20_0_15 = 'ptmatrix-basetimes-20-0-15'
    KIND_PTMATRIX_100_basetimes20_0_16 = 'ptmatrix-basetimes-20-0-16'
    KIND_PTMATRIX_100_basetimes20_0_17 = 'ptmatrix-basetimes-20-0-17'
    KIND_PTMATRIX_100_basetimes20_0_18 = 'ptmatrix-basetimes-20-0-18'
    KIND_PTMATRIX_100_basetimes20_0_19 = 'ptmatrix-basetimes-20-0-19'
    KIND_PTMATRIX_100_basetimes20_0_20 = 'ptmatrix-basetimes-20-0-20'
    KIND_PTMATRIX_100_basetimes20_0_21 = 'ptmatrix-basetimes-20-0-21'
    KIND_PTMATRIX_100_basetimes20_0_22 = 'ptmatrix-basetimes-20-0-22'
    KIND_PTMATRIX_100_basetimes20_0_23 = 'ptmatrix-basetimes-20-0-23'
    KIND_PTMATRIX_100_basetimes20_0_24 = 'ptmatrix-basetimes-20-0-24'
    KIND_PTMATRIX_100_basetimes20_0_25 = 'ptmatrix-basetimes-20-0-25'
    KIND_PTMATRIX_100_basetimes20_0_26 = 'ptmatrix-basetimes-20-0-26'
    KIND_PTMATRIX_100_basetimes20_0_27 = 'ptmatrix-basetimes-20-0-27'
    KIND_PTMATRIX_100_basetimes20_0_28 = 'ptmatrix-basetimes-20-0-28'
    KIND_PTMATRIX_100_basetimes20_0_29 = 'ptmatrix-basetimes-20-0-29'
    KIND_PTMATRIX_100_basetimes20_0_30 = 'ptmatrix-basetimes-20-0-30'

    KIND_PTMATRIX_100_basetimes20_1_0 = 'ptmatrix-basetimes-20-1-0'
    KIND_PTMATRIX_100_basetimes20_1_1 = 'ptmatrix-basetimes-20-1-1'
    KIND_PTMATRIX_100_basetimes20_1_2 = 'ptmatrix-basetimes-20-1-2'
    KIND_PTMATRIX_100_basetimes20_1_3 = 'ptmatrix-basetimes-20-1-3'
    KIND_PTMATRIX_100_basetimes20_1_4 = 'ptmatrix-basetimes-20-1-4'
    KIND_PTMATRIX_100_basetimes20_1_5 = 'ptmatrix-basetimes-20-1-5'
    KIND_PTMATRIX_100_basetimes20_1_6 = 'ptmatrix-basetimes-20-1-6'
    KIND_PTMATRIX_100_basetimes20_1_7 = 'ptmatrix-basetimes-20-1-7'
    KIND_PTMATRIX_100_basetimes20_1_8 = 'ptmatrix-basetimes-20-1-8'
    KIND_PTMATRIX_100_basetimes20_1_9 = 'ptmatrix-basetimes-20-1-9'
    KIND_PTMATRIX_100_basetimes20_1_10 = 'ptmatrix-basetimes-20-1-10'
    KIND_PTMATRIX_100_basetimes20_1_11 = 'ptmatrix-basetimes-20-1-11'
    KIND_PTMATRIX_100_basetimes20_1_12 = 'ptmatrix-basetimes-20-1-12'
    KIND_PTMATRIX_100_basetimes20_1_13 = 'ptmatrix-basetimes-20-1-13'
    KIND_PTMATRIX_100_basetimes20_1_14 = 'ptmatrix-basetimes-20-1-14'
    KIND_PTMATRIX_100_basetimes20_1_15 = 'ptmatrix-basetimes-20-1-15'
    KIND_PTMATRIX_100_basetimes20_1_16 = 'ptmatrix-basetimes-20-1-16'
    KIND_PTMATRIX_100_basetimes20_1_17 = 'ptmatrix-basetimes-20-1-17'
    KIND_PTMATRIX_100_basetimes20_1_18 = 'ptmatrix-basetimes-20-1-18'
    KIND_PTMATRIX_100_basetimes20_1_19 = 'ptmatrix-basetimes-20-1-19'
    KIND_PTMATRIX_100_basetimes20_1_20 = 'ptmatrix-basetimes-20-1-20'
    KIND_PTMATRIX_100_basetimes20_1_21 = 'ptmatrix-basetimes-20-1-21'
    KIND_PTMATRIX_100_basetimes20_1_22 = 'ptmatrix-basetimes-20-1-22'
    KIND_PTMATRIX_100_basetimes20_1_23 = 'ptmatrix-basetimes-20-1-23'
    KIND_PTMATRIX_100_basetimes20_1_24 = 'ptmatrix-basetimes-20-1-24'
    KIND_PTMATRIX_100_basetimes20_1_25 = 'ptmatrix-basetimes-20-1-25'
    KIND_PTMATRIX_100_basetimes20_1_26 = 'ptmatrix-basetimes-20-1-26'
    KIND_PTMATRIX_100_basetimes20_1_27 = 'ptmatrix-basetimes-20-1-27'
    KIND_PTMATRIX_100_basetimes20_1_28 = 'ptmatrix-basetimes-20-1-28'
    KIND_PTMATRIX_100_basetimes20_1_29 = 'ptmatrix-basetimes-20-1-29'
    KIND_PTMATRIX_100_basetimes20_1_30 = 'ptmatrix-basetimes-20-1-30'

    KIND_PTMATRIX_100_basetimes50_0_0 = 'ptmatrix-basetimes-50-0-0'
    KIND_PTMATRIX_100_basetimes50_0_1 = 'ptmatrix-basetimes-50-0-1'
    KIND_PTMATRIX_100_basetimes50_0_2 = 'ptmatrix-basetimes-50-0-2'
    KIND_PTMATRIX_100_basetimes50_0_3 = 'ptmatrix-basetimes-50-0-3'
    KIND_PTMATRIX_100_basetimes50_0_4 = 'ptmatrix-basetimes-50-0-4'
    KIND_PTMATRIX_100_basetimes50_0_5 = 'ptmatrix-basetimes-50-0-5'
    KIND_PTMATRIX_100_basetimes50_0_6 = 'ptmatrix-basetimes-50-0-6'
    KIND_PTMATRIX_100_basetimes50_0_7 = 'ptmatrix-basetimes-50-0-7'
    KIND_PTMATRIX_100_basetimes50_0_8 = 'ptmatrix-basetimes-50-0-8'
    KIND_PTMATRIX_100_basetimes50_0_9 = 'ptmatrix-basetimes-50-0-9'
    KIND_PTMATRIX_100_basetimes50_0_10 = 'ptmatrix-basetimes-50-0-10'
    KIND_PTMATRIX_100_basetimes50_0_11 = 'ptmatrix-basetimes-50-0-11'
    KIND_PTMATRIX_100_basetimes50_0_12 = 'ptmatrix-basetimes-50-0-12'
    KIND_PTMATRIX_100_basetimes50_0_13 = 'ptmatrix-basetimes-50-0-13'
    KIND_PTMATRIX_100_basetimes50_0_14 = 'ptmatrix-basetimes-50-0-14'
    KIND_PTMATRIX_100_basetimes50_0_15 = 'ptmatrix-basetimes-50-0-15'
    KIND_PTMATRIX_100_basetimes50_0_16 = 'ptmatrix-basetimes-50-0-16'
    KIND_PTMATRIX_100_basetimes50_0_17 = 'ptmatrix-basetimes-50-0-17'
    KIND_PTMATRIX_100_basetimes50_0_18 = 'ptmatrix-basetimes-50-0-18'
    KIND_PTMATRIX_100_basetimes50_0_19 = 'ptmatrix-basetimes-50-0-19'
    KIND_PTMATRIX_100_basetimes50_0_20 = 'ptmatrix-basetimes-50-0-20'
    KIND_PTMATRIX_100_basetimes50_0_21 = 'ptmatrix-basetimes-50-0-21'
    KIND_PTMATRIX_100_basetimes50_0_22 = 'ptmatrix-basetimes-50-0-22'
    KIND_PTMATRIX_100_basetimes50_0_23 = 'ptmatrix-basetimes-50-0-23'
    KIND_PTMATRIX_100_basetimes50_0_24 = 'ptmatrix-basetimes-50-0-24'
    KIND_PTMATRIX_100_basetimes50_0_25 = 'ptmatrix-basetimes-50-0-25'
    KIND_PTMATRIX_100_basetimes50_0_26 = 'ptmatrix-basetimes-50-0-26'
    KIND_PTMATRIX_100_basetimes50_0_27 = 'ptmatrix-basetimes-50-0-27'
    KIND_PTMATRIX_100_basetimes50_0_28 = 'ptmatrix-basetimes-50-0-28'
    KIND_PTMATRIX_100_basetimes50_0_29 = 'ptmatrix-basetimes-50-0-29'
    KIND_PTMATRIX_100_basetimes50_0_30 = 'ptmatrix-basetimes-50-0-30'

    KIND_PTMATRIX_100_basetimes50_1_0 = 'ptmatrix-basetimes-50-1-0'
    KIND_PTMATRIX_100_basetimes50_1_1 = 'ptmatrix-basetimes-50-1-1'
    KIND_PTMATRIX_100_basetimes50_1_2 = 'ptmatrix-basetimes-50-1-2'
    KIND_PTMATRIX_100_basetimes50_1_3 = 'ptmatrix-basetimes-50-1-3'
    KIND_PTMATRIX_100_basetimes50_1_4 = 'ptmatrix-basetimes-50-1-4'
    KIND_PTMATRIX_100_basetimes50_1_5 = 'ptmatrix-basetimes-50-1-5'
    KIND_PTMATRIX_100_basetimes50_1_6 = 'ptmatrix-basetimes-50-1-6'
    KIND_PTMATRIX_100_basetimes50_1_7 = 'ptmatrix-basetimes-50-1-7'
    KIND_PTMATRIX_100_basetimes50_1_8 = 'ptmatrix-basetimes-50-1-8'
    KIND_PTMATRIX_100_basetimes50_1_9 = 'ptmatrix-basetimes-50-1-9'
    KIND_PTMATRIX_100_basetimes50_1_10 = 'ptmatrix-basetimes-50-1-10'
    KIND_PTMATRIX_100_basetimes50_1_11 = 'ptmatrix-basetimes-50-1-11'
    KIND_PTMATRIX_100_basetimes50_1_12 = 'ptmatrix-basetimes-50-1-12'
    KIND_PTMATRIX_100_basetimes50_1_13 = 'ptmatrix-basetimes-50-1-13'
    KIND_PTMATRIX_100_basetimes50_1_14 = 'ptmatrix-basetimes-50-1-14'
    KIND_PTMATRIX_100_basetimes50_1_15 = 'ptmatrix-basetimes-50-1-15'
    KIND_PTMATRIX_100_basetimes50_1_16 = 'ptmatrix-basetimes-50-1-16'
    KIND_PTMATRIX_100_basetimes50_1_17 = 'ptmatrix-basetimes-50-1-17'
    KIND_PTMATRIX_100_basetimes50_1_18 = 'ptmatrix-basetimes-50-1-18'
    KIND_PTMATRIX_100_basetimes50_1_19 = 'ptmatrix-basetimes-50-1-19'
    KIND_PTMATRIX_100_basetimes50_1_20 = 'ptmatrix-basetimes-50-1-20'
    KIND_PTMATRIX_100_basetimes50_1_21 = 'ptmatrix-basetimes-50-1-21'
    KIND_PTMATRIX_100_basetimes50_1_22 = 'ptmatrix-basetimes-50-1-22'
    KIND_PTMATRIX_100_basetimes50_1_23 = 'ptmatrix-basetimes-50-1-23'
    KIND_PTMATRIX_100_basetimes50_1_24 = 'ptmatrix-basetimes-50-1-24'
    KIND_PTMATRIX_100_basetimes50_1_25 = 'ptmatrix-basetimes-50-1-25'
    KIND_PTMATRIX_100_basetimes50_1_26 = 'ptmatrix-basetimes-50-1-26'
    KIND_PTMATRIX_100_basetimes50_1_27 = 'ptmatrix-basetimes-50-1-27'
    KIND_PTMATRIX_100_basetimes50_1_28 = 'ptmatrix-basetimes-50-1-28'
    KIND_PTMATRIX_100_basetimes50_1_29 = 'ptmatrix-basetimes-50-1-29'
    KIND_PTMATRIX_100_basetimes50_1_30 = 'ptmatrix-basetimes-50-1-30'

    KIND_PTMATRIX_1000_basetimes200_10_0 = 'ptmatrix-basetimes-200-10-0'

    KIND_PTMATRIX_100_err20_0_0 = 'ptmatrix-err-20-0-0'
    KIND_PTMATRIX_100_err20_0_1 = 'ptmatrix-err-20-0-1'
    KIND_PTMATRIX_100_err20_0_2 = 'ptmatrix-err-20-0-2'
    KIND_PTMATRIX_100_err20_0_3 = 'ptmatrix-err-20-0-3'
    KIND_PTMATRIX_100_err20_0_4 = 'ptmatrix-err-20-0-4'
    KIND_PTMATRIX_100_err20_0_5 = 'ptmatrix-err-20-0-5'
    KIND_PTMATRIX_100_err20_0_6 = 'ptmatrix-err-20-0-6'
    KIND_PTMATRIX_100_err20_0_7 = 'ptmatrix-err-20-0-7'
    KIND_PTMATRIX_100_err20_0_8 = 'ptmatrix-err-20-0-8'
    KIND_PTMATRIX_100_err20_0_9 = 'ptmatrix-err-20-0-9'
    KIND_PTMATRIX_100_err20_0_10 = 'ptmatrix-err-20-0-10'
    KIND_PTMATRIX_100_err20_0_11 = 'ptmatrix-err-20-0-11'
    KIND_PTMATRIX_100_err20_0_12 = 'ptmatrix-err-20-0-12'
    KIND_PTMATRIX_100_err20_0_13 = 'ptmatrix-err-20-0-13'
    KIND_PTMATRIX_100_err20_0_14 = 'ptmatrix-err-20-0-14'
    KIND_PTMATRIX_100_err20_0_15 = 'ptmatrix-err-20-0-15'
    KIND_PTMATRIX_100_err20_0_16 = 'ptmatrix-err-20-0-16'
    KIND_PTMATRIX_100_err20_0_17 = 'ptmatrix-err-20-0-17'
    KIND_PTMATRIX_100_err20_0_18 = 'ptmatrix-err-20-0-18'
    KIND_PTMATRIX_100_err20_0_19 = 'ptmatrix-err-20-0-19'
    KIND_PTMATRIX_100_err20_0_20 = 'ptmatrix-err-20-0-20'
    KIND_PTMATRIX_100_err20_0_21 = 'ptmatrix-err-20-0-21'
    KIND_PTMATRIX_100_err20_0_22 = 'ptmatrix-err-20-0-22'
    KIND_PTMATRIX_100_err20_0_23 = 'ptmatrix-err-20-0-23'
    KIND_PTMATRIX_100_err20_0_24 = 'ptmatrix-err-20-0-24'
    KIND_PTMATRIX_100_err20_0_25 = 'ptmatrix-err-20-0-25'
    KIND_PTMATRIX_100_err20_0_26 = 'ptmatrix-err-20-0-26'
    KIND_PTMATRIX_100_err20_0_27 = 'ptmatrix-err-20-0-27'
    KIND_PTMATRIX_100_err20_0_28 = 'ptmatrix-err-20-0-28'
    KIND_PTMATRIX_100_err20_0_29 = 'ptmatrix-err-20-0-29'
    KIND_PTMATRIX_100_err20_0_30 = 'ptmatrix-err-20-0-30'

    KIND_PTMATRIX_100_err20_1_0 = 'ptmatrix-err-20-1-0'
    KIND_PTMATRIX_100_err20_1_1 = 'ptmatrix-err-20-1-1'
    KIND_PTMATRIX_100_err20_1_2 = 'ptmatrix-err-20-1-2'
    KIND_PTMATRIX_100_err20_1_3 = 'ptmatrix-err-20-1-3'
    KIND_PTMATRIX_100_err20_1_4 = 'ptmatrix-err-20-1-4'
    KIND_PTMATRIX_100_err20_1_5 = 'ptmatrix-err-20-1-5'
    KIND_PTMATRIX_100_err20_1_6 = 'ptmatrix-err-20-1-6'
    KIND_PTMATRIX_100_err20_1_7 = 'ptmatrix-err-20-1-7'
    KIND_PTMATRIX_100_err20_1_8 = 'ptmatrix-err-20-1-8'
    KIND_PTMATRIX_100_err20_1_9 = 'ptmatrix-err-20-1-9'
    KIND_PTMATRIX_100_err20_1_10 = 'ptmatrix-err-20-1-10'
    KIND_PTMATRIX_100_err20_1_11 = 'ptmatrix-err-20-1-11'
    KIND_PTMATRIX_100_err20_1_12 = 'ptmatrix-err-20-1-12'
    KIND_PTMATRIX_100_err20_1_13 = 'ptmatrix-err-20-1-13'
    KIND_PTMATRIX_100_err20_1_14 = 'ptmatrix-err-20-1-14'
    KIND_PTMATRIX_100_err20_1_15 = 'ptmatrix-err-20-1-15'
    KIND_PTMATRIX_100_err20_1_16 = 'ptmatrix-err-20-1-16'
    KIND_PTMATRIX_100_err20_1_17 = 'ptmatrix-err-20-1-17'
    KIND_PTMATRIX_100_err20_1_18 = 'ptmatrix-err-20-1-18'
    KIND_PTMATRIX_100_err20_1_19 = 'ptmatrix-err-20-1-19'
    KIND_PTMATRIX_100_err20_1_20 = 'ptmatrix-err-20-1-20'
    KIND_PTMATRIX_100_err20_1_21 = 'ptmatrix-err-20-1-21'
    KIND_PTMATRIX_100_err20_1_22 = 'ptmatrix-err-20-1-22'
    KIND_PTMATRIX_100_err20_1_23 = 'ptmatrix-err-20-1-23'
    KIND_PTMATRIX_100_err20_1_24 = 'ptmatrix-err-20-1-24'
    KIND_PTMATRIX_100_err20_1_25 = 'ptmatrix-err-20-1-25'
    KIND_PTMATRIX_100_err20_1_26 = 'ptmatrix-err-20-1-26'
    KIND_PTMATRIX_100_err20_1_27 = 'ptmatrix-err-20-1-27'
    KIND_PTMATRIX_100_err20_1_28 = 'ptmatrix-err-20-1-28'
    KIND_PTMATRIX_100_err20_1_29 = 'ptmatrix-err-20-1-29'
    KIND_PTMATRIX_100_err20_1_30 = 'ptmatrix-err-20-1-30'

    KIND_PTMATRIX_100_err50_0_0 = 'ptmatrix-err-50-0-0'
    KIND_PTMATRIX_100_err50_0_1 = 'ptmatrix-err-50-0-1'
    KIND_PTMATRIX_100_err50_0_2 = 'ptmatrix-err-50-0-2'
    KIND_PTMATRIX_100_err50_0_3 = 'ptmatrix-err-50-0-3'
    KIND_PTMATRIX_100_err50_0_4 = 'ptmatrix-err-50-0-4'
    KIND_PTMATRIX_100_err50_0_5 = 'ptmatrix-err-50-0-5'
    KIND_PTMATRIX_100_err50_0_6 = 'ptmatrix-err-50-0-6'
    KIND_PTMATRIX_100_err50_0_7 = 'ptmatrix-err-50-0-7'
    KIND_PTMATRIX_100_err50_0_8 = 'ptmatrix-err-50-0-8'
    KIND_PTMATRIX_100_err50_0_9 = 'ptmatrix-err-50-0-9'
    KIND_PTMATRIX_100_err50_0_10 = 'ptmatrix-err-50-0-10'
    KIND_PTMATRIX_100_err50_0_11 = 'ptmatrix-err-50-0-11'
    KIND_PTMATRIX_100_err50_0_12 = 'ptmatrix-err-50-0-12'
    KIND_PTMATRIX_100_err50_0_13 = 'ptmatrix-err-50-0-13'
    KIND_PTMATRIX_100_err50_0_14 = 'ptmatrix-err-50-0-14'
    KIND_PTMATRIX_100_err50_0_15 = 'ptmatrix-err-50-0-15'
    KIND_PTMATRIX_100_err50_0_16 = 'ptmatrix-err-50-0-16'
    KIND_PTMATRIX_100_err50_0_17 = 'ptmatrix-err-50-0-17'
    KIND_PTMATRIX_100_err50_0_18 = 'ptmatrix-err-50-0-18'
    KIND_PTMATRIX_100_err50_0_19 = 'ptmatrix-err-50-0-19'
    KIND_PTMATRIX_100_err50_0_20 = 'ptmatrix-err-50-0-20'
    KIND_PTMATRIX_100_err50_0_21 = 'ptmatrix-err-50-0-21'
    KIND_PTMATRIX_100_err50_0_22 = 'ptmatrix-err-50-0-22'
    KIND_PTMATRIX_100_err50_0_23 = 'ptmatrix-err-50-0-23'
    KIND_PTMATRIX_100_err50_0_24 = 'ptmatrix-err-50-0-24'
    KIND_PTMATRIX_100_err50_0_25 = 'ptmatrix-err-50-0-25'
    KIND_PTMATRIX_100_err50_0_26 = 'ptmatrix-err-50-0-26'
    KIND_PTMATRIX_100_err50_0_27 = 'ptmatrix-err-50-0-27'
    KIND_PTMATRIX_100_err50_0_28 = 'ptmatrix-err-50-0-28'
    KIND_PTMATRIX_100_err50_0_29 = 'ptmatrix-err-50-0-29'
    KIND_PTMATRIX_100_err50_0_30 = 'ptmatrix-err-50-0-30'

    KIND_PTMATRIX_100_err50_1_0 = 'ptmatrix-err-50-1-0'
    KIND_PTMATRIX_100_err50_1_1 = 'ptmatrix-err-50-1-1'
    KIND_PTMATRIX_100_err50_1_2 = 'ptmatrix-err-50-1-2'
    KIND_PTMATRIX_100_err50_1_3 = 'ptmatrix-err-50-1-3'
    KIND_PTMATRIX_100_err50_1_4 = 'ptmatrix-err-50-1-4'
    KIND_PTMATRIX_100_err50_1_5 = 'ptmatrix-err-50-1-5'
    KIND_PTMATRIX_100_err50_1_6 = 'ptmatrix-err-50-1-6'
    KIND_PTMATRIX_100_err50_1_7 = 'ptmatrix-err-50-1-7'
    KIND_PTMATRIX_100_err50_1_8 = 'ptmatrix-err-50-1-8'
    KIND_PTMATRIX_100_err50_1_9 = 'ptmatrix-err-50-1-9'
    KIND_PTMATRIX_100_err50_1_10 = 'ptmatrix-err-50-1-10'
    KIND_PTMATRIX_100_err50_1_11 = 'ptmatrix-err-50-1-11'
    KIND_PTMATRIX_100_err50_1_12 = 'ptmatrix-err-50-1-12'
    KIND_PTMATRIX_100_err50_1_13 = 'ptmatrix-err-50-1-13'
    KIND_PTMATRIX_100_err50_1_14 = 'ptmatrix-err-50-1-14'
    KIND_PTMATRIX_100_err50_1_15 = 'ptmatrix-err-50-1-15'
    KIND_PTMATRIX_100_err50_1_16 = 'ptmatrix-err-50-1-16'
    KIND_PTMATRIX_100_err50_1_17 = 'ptmatrix-err-50-1-17'
    KIND_PTMATRIX_100_err50_1_18 = 'ptmatrix-err-50-1-18'
    KIND_PTMATRIX_100_err50_1_19 = 'ptmatrix-err-50-1-19'
    KIND_PTMATRIX_100_err50_1_20 = 'ptmatrix-err-50-1-20'
    KIND_PTMATRIX_100_err50_1_21 = 'ptmatrix-err-50-1-21'
    KIND_PTMATRIX_100_err50_1_22 = 'ptmatrix-err-50-1-22'
    KIND_PTMATRIX_100_err50_1_23 = 'ptmatrix-err-50-1-23'
    KIND_PTMATRIX_100_err50_1_24 = 'ptmatrix-err-50-1-24'
    KIND_PTMATRIX_100_err50_1_25 = 'ptmatrix-err-50-1-25'
    KIND_PTMATRIX_100_err50_1_26 = 'ptmatrix-err-50-1-26'
    KIND_PTMATRIX_100_err50_1_27 = 'ptmatrix-err-50-1-27'
    KIND_PTMATRIX_100_err50_1_28 = 'ptmatrix-err-50-1-28'
    KIND_PTMATRIX_100_err50_1_29 = 'ptmatrix-err-50-1-29'
    KIND_PTMATRIX_100_err50_1_30 = 'ptmatrix-err-50-1-30'

    KIND_PTMATRIX_100_size1_0_10 = 'ptmatrix-size-1-0-10'
    KIND_PTMATRIX_100_size1_0_15 = 'ptmatrix-size-1-0-15'
    KIND_PTMATRIX_100_size1_0_20 = 'ptmatrix-size-1-0-20'
    KIND_PTMATRIX_100_size1_0_25 = 'ptmatrix-size-1-0-25'
    KIND_PTMATRIX_100_size1_0_30 = 'ptmatrix-size-1-0-30'
    KIND_PTMATRIX_100_size1_0_35 = 'ptmatrix-size-1-0-35'
    KIND_PTMATRIX_100_size1_0_40 = 'ptmatrix-size-1-0-40'
    KIND_PTMATRIX_100_size1_0_45 = 'ptmatrix-size-1-0-45'
    KIND_PTMATRIX_100_size1_0_50 = 'ptmatrix-size-1-0-50'
    KIND_PTMATRIX_100_size1_1_10 = 'ptmatrix-size-1-1-10'
    KIND_PTMATRIX_100_size1_1_15 = 'ptmatrix-size-1-1-15'
    KIND_PTMATRIX_100_size1_1_20 = 'ptmatrix-size-1-1-20'
    KIND_PTMATRIX_100_size1_1_25 = 'ptmatrix-size-1-1-25'
    KIND_PTMATRIX_100_size1_1_30 = 'ptmatrix-size-1-1-30'
    KIND_PTMATRIX_100_size1_1_35 = 'ptmatrix-size-1-1-35'
    KIND_PTMATRIX_100_size1_1_40 = 'ptmatrix-size-1-1-40'
    KIND_PTMATRIX_100_size1_1_45 = 'ptmatrix-size-1-1-45'
    KIND_PTMATRIX_100_size1_1_50 = 'ptmatrix-size-1-1-50'
    KIND_PTMATRIX_100_size5_0_10 = 'ptmatrix-size-5-0-10'
    KIND_PTMATRIX_100_size5_0_15 = 'ptmatrix-size-5-0-15'
    KIND_PTMATRIX_100_size5_0_20 = 'ptmatrix-size-5-0-20'
    KIND_PTMATRIX_100_size5_0_25 = 'ptmatrix-size-5-0-25'
    KIND_PTMATRIX_100_size5_0_30 = 'ptmatrix-size-5-0-30'
    KIND_PTMATRIX_100_size5_0_35 = 'ptmatrix-size-5-0-35'
    KIND_PTMATRIX_100_size5_0_40 = 'ptmatrix-size-5-0-40'
    KIND_PTMATRIX_100_size5_0_45 = 'ptmatrix-size-5-0-45'
    KIND_PTMATRIX_100_size5_0_50 = 'ptmatrix-size-5-0-50'
    KIND_PTMATRIX_100_size5_1_10 = 'ptmatrix-size-5-1-10'
    KIND_PTMATRIX_100_size5_1_15 = 'ptmatrix-size-5-1-15'
    KIND_PTMATRIX_100_size5_1_20 = 'ptmatrix-size-5-1-20'
    KIND_PTMATRIX_100_size5_1_25 = 'ptmatrix-size-5-1-25'
    KIND_PTMATRIX_100_size5_1_30 = 'ptmatrix-size-5-1-30'
    KIND_PTMATRIX_100_size5_1_35 = 'ptmatrix-size-5-1-35'
    KIND_PTMATRIX_100_size5_1_40 = 'ptmatrix-size-5-1-40'
    KIND_PTMATRIX_100_size5_1_45 = 'ptmatrix-size-5-1-45'
    KIND_PTMATRIX_100_size5_1_50 = 'ptmatrix-size-5-1-50'

    KIND_PTMATRIX_100_testTime100_0_0 = 'ptmatrix-testTime-100-0-0'
    KIND_PTMATRIX_100_testTime500_0_0 = 'ptmatrix-testTime-500-0-0'
    KIND_PTMATRIX_100_testTime1000_0_0 = 'ptmatrix-testTime-1000-0-0'
    KIND_PTMATRIX_100_testTime2000_0_0 = 'ptmatrix-testTime-2000-0-0'
    KIND_PTMATRIX_100_testTime5000_0_0 = 'ptmatrix-testTime-5000-0-0'
    KIND_PTMATRIX_100_testTime10000_0_0 = 'ptmatrix-testTime-10000-0-0'

    KIND_PTMATRIX_100_gse72056_scRNA_maxMean = 'ptmatrix-gse-72056-scRNA-maxMean'
    KIND_PTMATRIX_100_gse103322_scRNA_maxMean = 'ptmatrix-gse-103322-scRNA-maxMean'

    KIND_OBJECTS = ( \
        (KIND_MOVIELENS_100K, 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'), \
        (KIND_MOVIELENS_1M,  'http://files.grouplens.org/datasets/movielens/ml-1m.zip'), \
        (KIND_MOVIELENS_10M, 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'), \
        (KIND_MOVIELENS_20M, 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'), \
        #(KIND_NETFLIX, None), \
        #(KIND_PTMATRIX_100_basetimes20_0_0, None)
        (KIND_PTMATRIX_100_basetimes20_0_0, None), \
        (KIND_PTMATRIX_100_basetimes20_0_1, None), \
        (KIND_PTMATRIX_100_basetimes20_0_2, None), \
        (KIND_PTMATRIX_100_basetimes20_0_3, None), \
        (KIND_PTMATRIX_100_basetimes20_0_4, None), \
        (KIND_PTMATRIX_100_basetimes20_0_5, None), \
        (KIND_PTMATRIX_100_basetimes20_0_6, None), \
        (KIND_PTMATRIX_100_basetimes20_0_7, None), \
        (KIND_PTMATRIX_100_basetimes20_0_8, None), \
        (KIND_PTMATRIX_100_basetimes20_0_9, None), \
        (KIND_PTMATRIX_100_basetimes20_0_10, None), \
        (KIND_PTMATRIX_100_basetimes20_0_11, None), \
        (KIND_PTMATRIX_100_basetimes20_0_12, None), \
        (KIND_PTMATRIX_100_basetimes20_0_13, None), \
        (KIND_PTMATRIX_100_basetimes20_0_14, None), \
        (KIND_PTMATRIX_100_basetimes20_0_15, None), \
        (KIND_PTMATRIX_100_basetimes20_0_16, None), \
        (KIND_PTMATRIX_100_basetimes20_0_17, None), \
        (KIND_PTMATRIX_100_basetimes20_0_18, None), \
        (KIND_PTMATRIX_100_basetimes20_0_19, None), \
        (KIND_PTMATRIX_100_basetimes20_0_20, None), \
        (KIND_PTMATRIX_100_basetimes20_0_21, None), \
        (KIND_PTMATRIX_100_basetimes20_0_22, None), \
        (KIND_PTMATRIX_100_basetimes20_0_23, None), \
        (KIND_PTMATRIX_100_basetimes20_0_24, None), \
        (KIND_PTMATRIX_100_basetimes20_0_25, None), \
        (KIND_PTMATRIX_100_basetimes20_0_26, None), \
        (KIND_PTMATRIX_100_basetimes20_0_27, None), \
        (KIND_PTMATRIX_100_basetimes20_0_28, None), \
        (KIND_PTMATRIX_100_basetimes20_0_29, None), \
        (KIND_PTMATRIX_100_basetimes20_0_30, None), \
        )

    def _set_kind_and_url(self, kind):
        self.kind = kind
        #print("3",self.kind)
        return True
        #for k, url in self.KIND_OBJECTS:
            #if k == kind:
                #self.url = url
                #return True
        #raise NotImplementedError()

    """
    def _download_data_if_not_exists(self):
        if not os.path.exists('data/{}'.format(self.kind)):
            os.system('wget {url} -O data/{kind}.zip'.format(
                url=self.url, kind=self.kind))
            os.system(
                'unzip data/{kind}.zip -d data/{kind}/'.format(kind=self.kind))
    """



    def __init_data(self, detail_path, delimiter, header=False):
        current_u = 0
        u_dict = {}
        current_i = 0
        i_dict = {}

        data = []
        with open('data/{}{}'.format(self.kind, detail_path), 'r') as f:
            if header:
                f.readline()

            for line in f:
                cols = line.strip().split(delimiter)
                assert len(cols) == 4
                # cols = [float(c) for c in cols]
                user_id = cols[0]
                item_id = cols[1]
                r = float(cols[2])
                t = int(cols[3])

                u = u_dict.get(user_id, None)
                if u is None:
                    u_dict[user_id] = current_u
                    u = current_u
                    current_u += 1

                i = i_dict.get(item_id, None)
                if i is None:
                    # print(current_i)
                    i_dict[item_id] = current_i
                    i = current_i
                    current_i += 1

                data.append((u, i, r, t))
            f.close()

        data = np.array(data)
        print("Shape:")
        print(data.shape)
        np.save('data/{}/data.npy'.format(self.kind), data)

    def _init_data(self):
        #print("2",self.kind)
        #print("2",self.KIND_PTMATRIX_100_basetimes20_0_0)
        #print(self.kind == self.KIND_PTMATRIX_100_basetimes20_0_0)
        if self.kind == self.KIND_MOVIELENS_100K:
            self.__init_data('/ml-100k/u.data', '\t')
        elif self.kind == self.KIND_MOVIELENS_1M:
            self.__init_data('/ml-1m/ratings.dat', '::')
        elif self.kind == self.KIND_MOVIELENS_10M:
            self.__init_data('/ml-10M100K/ratings.dat', '::')
        elif self.kind == self.KIND_MOVIELENS_20M:
            self.__init_data('/ml-20m/ratings.csv', ',', header=True)
        elif self.kind == self.KIND_PTMATRIX_500:
            self.__init_data('/ratings.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_1000:
            self.__init_data('/ratings.dat', '::')

        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_0:
            self.__init_data('/basetimes20-0-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_1:
            self.__init_data('/basetimes20-0-1.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_2:
            self.__init_data('/basetimes20-0-2.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_3:
            self.__init_data('/basetimes20-0-3.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_4:
            self.__init_data('/basetimes20-0-4.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_5:
            self.__init_data('/basetimes20-0-5.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_6:
            self.__init_data('/basetimes20-0-6.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_7:
            self.__init_data('/basetimes20-0-7.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_8:
            self.__init_data('/basetimes20-0-8.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_9:
            self.__init_data('/basetimes20-0-9.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_10:
            self.__init_data('/basetimes20-0-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_11:
            self.__init_data('/basetimes20-0-11.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_12:
            self.__init_data('/basetimes20-0-12.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_13:
            self.__init_data('/basetimes20-0-13.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_14:
            self.__init_data('/basetimes20-0-14.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_15:
            self.__init_data('/basetimes20-0-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_16:
            self.__init_data('/basetimes20-0-16.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_17:
            self.__init_data('/basetimes20-0-17.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_18:
            self.__init_data('/basetimes20-0-18.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_19:
            self.__init_data('/basetimes20-0-19.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_20:
            self.__init_data('/basetimes20-0-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_21:
            self.__init_data('/basetimes20-0-21.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_22:
            self.__init_data('/basetimes20-0-22.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_23:
            self.__init_data('/basetimes20-0-23.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_24:
            self.__init_data('/basetimes20-0-24.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_25:
            self.__init_data('/basetimes20-0-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_26:
            self.__init_data('/basetimes20-0-26.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_27:
            self.__init_data('/basetimes20-0-27.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_28:
            self.__init_data('/basetimes20-0-28.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_29:
            self.__init_data('/basetimes20-0-29.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_0_30:
            self.__init_data('/basetimes20-0-30.dat', '::')

        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_0:
            self.__init_data('/basetimes20-1-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_1:
            self.__init_data('/basetimes20-1-1.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_2:
            self.__init_data('/basetimes20-1-2.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_3:
            self.__init_data('/basetimes20-1-3.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_4:
            self.__init_data('/basetimes20-1-4.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_5:
            self.__init_data('/basetimes20-1-5.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_6:
            self.__init_data('/basetimes20-1-6.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_7:
            self.__init_data('/basetimes20-1-7.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_8:
            self.__init_data('/basetimes20-1-8.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_9:
            self.__init_data('/basetimes20-1-9.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_10:
            self.__init_data('/basetimes20-1-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_11:
            self.__init_data('/basetimes20-1-11.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_12:
            self.__init_data('/basetimes20-1-12.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_13:
            self.__init_data('/basetimes20-1-13.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_14:
            self.__init_data('/basetimes20-1-14.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_15:
            self.__init_data('/basetimes20-1-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_16:
            self.__init_data('/basetimes20-1-16.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_17:
            self.__init_data('/basetimes20-1-17.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_18:
            self.__init_data('/basetimes20-1-18.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_19:
            self.__init_data('/basetimes20-1-19.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_20:
            self.__init_data('/basetimes20-1-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_21:
            self.__init_data('/basetimes20-1-21.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_22:
            self.__init_data('/basetimes20-1-22.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_23:
            self.__init_data('/basetimes20-1-23.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_24:
            self.__init_data('/basetimes20-1-24.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_25:
            self.__init_data('/basetimes20-1-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_26:
            self.__init_data('/basetimes20-1-26.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_27:
            self.__init_data('/basetimes20-1-27.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_28:
            self.__init_data('/basetimes20-1-28.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_29:
            self.__init_data('/basetimes20-1-29.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes20_1_30:
            self.__init_data('/basetimes20-1-30.dat', '::')

        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_0:
            self.__init_data('/basetimes50-0-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_1:
            self.__init_data('/basetimes50-0-1.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_2:
            self.__init_data('/basetimes50-0-2.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_3:
            self.__init_data('/basetimes50-0-3.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_4:
            self.__init_data('/basetimes50-0-4.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_5:
            self.__init_data('/basetimes50-0-5.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_6:
            self.__init_data('/basetimes50-0-6.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_7:
            self.__init_data('/basetimes50-0-7.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_8:
            self.__init_data('/basetimes50-0-8.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_9:
            self.__init_data('/basetimes50-0-9.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_10:
            self.__init_data('/basetimes50-0-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_11:
            self.__init_data('/basetimes50-0-11.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_12:
            self.__init_data('/basetimes50-0-12.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_13:
            self.__init_data('/basetimes50-0-13.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_14:
            self.__init_data('/basetimes50-0-14.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_15:
            self.__init_data('/basetimes50-0-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_16:
            self.__init_data('/basetimes50-0-16.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_17:
            self.__init_data('/basetimes50-0-17.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_18:
            self.__init_data('/basetimes50-0-18.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_19:
            self.__init_data('/basetimes50-0-19.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_20:
            self.__init_data('/basetimes50-0-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_21:
            self.__init_data('/basetimes50-0-21.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_22:
            self.__init_data('/basetimes50-0-22.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_23:
            self.__init_data('/basetimes50-0-23.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_24:
            self.__init_data('/basetimes50-0-24.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_25:
            self.__init_data('/basetimes50-0-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_26:
            self.__init_data('/basetimes50-0-26.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_27:
            self.__init_data('/basetimes50-0-27.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_28:
            self.__init_data('/basetimes50-0-28.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_29:
            self.__init_data('/basetimes50-0-29.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_0_30:
            self.__init_data('/basetimes50-0-30.dat', '::')

        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_0:
            self.__init_data('/basetimes50-1-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_1:
            self.__init_data('/basetimes50-1-1.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_2:
            self.__init_data('/basetimes50-1-2.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_3:
            self.__init_data('/basetimes50-1-3.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_4:
            self.__init_data('/basetimes50-1-4.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_5:
            self.__init_data('/basetimes50-1-5.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_6:
            self.__init_data('/basetimes50-1-6.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_7:
            self.__init_data('/basetimes50-1-7.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_8:
            self.__init_data('/basetimes50-1-8.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_9:
            self.__init_data('/basetimes50-1-9.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_10:
            self.__init_data('/basetimes50-1-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_11:
            self.__init_data('/basetimes50-1-11.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_12:
            self.__init_data('/basetimes50-1-12.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_13:
            self.__init_data('/basetimes50-1-13.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_14:
            self.__init_data('/basetimes50-1-14.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_15:
            self.__init_data('/basetimes50-1-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_16:
            self.__init_data('/basetimes50-1-16.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_17:
            self.__init_data('/basetimes50-1-17.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_18:
            self.__init_data('/basetimes50-1-18.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_19:
            self.__init_data('/basetimes50-1-19.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_20:
            self.__init_data('/basetimes50-1-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_21:
            self.__init_data('/basetimes50-1-21.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_22:
            self.__init_data('/basetimes50-1-22.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_23:
            self.__init_data('/basetimes50-1-23.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_24:
            self.__init_data('/basetimes50-1-24.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_25:
            self.__init_data('/basetimes50-1-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_26:
            self.__init_data('/basetimes50-1-26.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_27:
            self.__init_data('/basetimes50-1-27.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_28:
            self.__init_data('/basetimes50-1-28.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_29:
            self.__init_data('/basetimes50-1-29.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_basetimes50_1_30:
            self.__init_data('/basetimes50-1-30.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_1000_basetimes200_10_0:
            self.__init_data('/basetimes200-10-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_0:
            self.__init_data('/err20-0-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_1:
            self.__init_data('/err20-0-1.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_2:
            self.__init_data('/err20-0-2.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_3:
            self.__init_data('/err20-0-3.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_4:
            self.__init_data('/err20-0-4.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_5:
            self.__init_data('/err20-0-5.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_6:
            self.__init_data('/err20-0-6.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_7:
            self.__init_data('/err20-0-7.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_8:
            self.__init_data('/err20-0-8.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_9:
            self.__init_data('/err20-0-9.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_10:
            self.__init_data('/err20-0-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_11:
            self.__init_data('/err20-0-11.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_12:
            self.__init_data('/err20-0-12.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_13:
            self.__init_data('/err20-0-13.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_14:
            self.__init_data('/err20-0-14.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_15:
            self.__init_data('/err20-0-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_16:
            self.__init_data('/err20-0-16.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_17:
            self.__init_data('/err20-0-17.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_18:
            self.__init_data('/err20-0-18.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_19:
            self.__init_data('/err20-0-19.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_20:
            self.__init_data('/err20-0-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_21:
            self.__init_data('/err20-0-21.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_22:
            self.__init_data('/err20-0-22.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_23:
            self.__init_data('/err20-0-23.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_24:
            self.__init_data('/err20-0-24.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_25:
            self.__init_data('/err20-0-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_26:
            self.__init_data('/err20-0-26.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_27:
            self.__init_data('/err20-0-27.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_28:
            self.__init_data('/err20-0-28.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_29:
            self.__init_data('/err20-0-29.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_0_30:
            self.__init_data('/err20-0-30.dat', '::')

        elif self.kind == self.KIND_PTMATRIX_100_err20_1_0:
            self.__init_data('/err20-1-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_1:
            self.__init_data('/err20-1-1.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_2:
            self.__init_data('/err20-1-2.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_3:
            self.__init_data('/err20-1-3.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_4:
            self.__init_data('/err20-1-4.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_5:
            self.__init_data('/err20-1-5.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_6:
            self.__init_data('/err20-1-6.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_7:
            self.__init_data('/err20-1-7.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_8:
            self.__init_data('/err20-1-8.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_9:
            self.__init_data('/err20-1-9.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_10:
            self.__init_data('/err20-1-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_11:
            self.__init_data('/err20-1-11.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_12:
            self.__init_data('/err20-1-12.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_13:
            self.__init_data('/err20-1-13.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_14:
            self.__init_data('/err20-1-14.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_15:
            self.__init_data('/err20-1-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_16:
            self.__init_data('/err20-1-16.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_17:
            self.__init_data('/err20-1-17.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_18:
            self.__init_data('/err20-1-18.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_19:
            self.__init_data('/err20-1-19.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_20:
            self.__init_data('/err20-1-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_21:
            self.__init_data('/err20-1-21.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_22:
            self.__init_data('/err20-1-22.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_23:
            self.__init_data('/err20-1-23.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_24:
            self.__init_data('/err20-1-24.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_25:
            self.__init_data('/err20-1-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_26:
            self.__init_data('/err20-1-26.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_27:
            self.__init_data('/err20-1-27.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_28:
            self.__init_data('/err20-1-28.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_29:
            self.__init_data('/err20-1-29.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err20_1_30:
            self.__init_data('/err20-1-30.dat', '::')

        elif self.kind == self.KIND_PTMATRIX_100_err50_0_0:
            self.__init_data('/err50-0-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_1:
            self.__init_data('/err50-0-1.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_2:
            self.__init_data('/err50-0-2.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_3:
            self.__init_data('/err50-0-3.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_4:
            self.__init_data('/err50-0-4.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_5:
            self.__init_data('/err50-0-5.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_6:
            self.__init_data('/err50-0-6.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_7:
            self.__init_data('/err50-0-7.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_8:
            self.__init_data('/err50-0-8.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_9:
            self.__init_data('/err50-0-9.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_10:
            self.__init_data('/err50-0-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_11:
            self.__init_data('/err50-0-11.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_12:
            self.__init_data('/err50-0-12.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_13:
            self.__init_data('/err50-0-13.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_14:
            self.__init_data('/err50-0-14.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_15:
            self.__init_data('/err50-0-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_16:
            self.__init_data('/err50-0-16.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_17:
            self.__init_data('/err50-0-17.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_18:
            self.__init_data('/err50-0-18.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_19:
            self.__init_data('/err50-0-19.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_20:
            self.__init_data('/err50-0-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_21:
            self.__init_data('/err50-0-21.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_22:
            self.__init_data('/err50-0-22.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_23:
            self.__init_data('/err50-0-23.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_24:
            self.__init_data('/err50-0-24.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_25:
            self.__init_data('/err50-0-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_26:
            self.__init_data('/err50-0-26.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_27:
            self.__init_data('/err50-0-27.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_28:
            self.__init_data('/err50-0-28.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_29:
            self.__init_data('/err50-0-29.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_0_30:
            self.__init_data('/err50-0-30.dat', '::')

        elif self.kind == self.KIND_PTMATRIX_100_err50_1_0:
            self.__init_data('/err50-1-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_1:
            self.__init_data('/err50-1-1.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_2:
            self.__init_data('/err50-1-2.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_3:
            self.__init_data('/err50-1-3.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_4:
            self.__init_data('/err50-1-4.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_5:
            self.__init_data('/err50-1-5.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_6:
            self.__init_data('/err50-1-6.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_7:
            self.__init_data('/err50-1-7.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_8:
            self.__init_data('/err50-1-8.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_9:
            self.__init_data('/err50-1-9.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_10:
            self.__init_data('/err50-1-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_11:
            self.__init_data('/err50-1-11.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_12:
            self.__init_data('/err50-1-12.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_13:
            self.__init_data('/err50-1-13.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_14:
            self.__init_data('/err50-1-14.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_15:
            self.__init_data('/err50-1-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_16:
            self.__init_data('/err50-1-16.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_17:
            self.__init_data('/err50-1-17.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_18:
            self.__init_data('/err50-1-18.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_19:
            self.__init_data('/err50-1-19.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_20:
            self.__init_data('/err50-1-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_21:
            self.__init_data('/err50-1-21.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_22:
            self.__init_data('/err50-1-22.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_23:
            self.__init_data('/err50-1-23.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_24:
            self.__init_data('/err50-1-24.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_25:
            self.__init_data('/err50-1-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_26:
            self.__init_data('/err50-1-26.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_27:
            self.__init_data('/err50-1-27.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_28:
            self.__init_data('/err50-1-28.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_29:
            self.__init_data('/err50-1-29.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_err50_1_30:
            self.__init_data('/err50-1-30.dat', '::')

        elif self.kind == self.KIND_PTMATRIX_100_size1_0_10:
            self.__init_data('/size1-0-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_0_15:
            self.__init_data('/size1-0-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_0_20:
            self.__init_data('/size1-0-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_0_25:
            self.__init_data('/size1-0-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_0_30:
            self.__init_data('/size1-0-30.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_0_35:
            self.__init_data('/size1-0-35.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_0_40:
            self.__init_data('/size1-0-40.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_0_45:
            self.__init_data('/size1-0-45.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_0_50:
            self.__init_data('/size1-0-50.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_1_10:
            self.__init_data('/size1-1-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_1_15:
            self.__init_data('/size1-1-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_1_20:
            self.__init_data('/size1-1-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_1_25:
            self.__init_data('/size1-1-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_1_30:
            self.__init_data('/size1-1-30.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_1_35:
            self.__init_data('/size1-1-35.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_1_40:
            self.__init_data('/size1-1-40.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_1_45:
            self.__init_data('/size1-1-45.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size1_1_50:
            self.__init_data('/size1-1-50.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_0_10:
            self.__init_data('/size5-0-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_0_15:
            self.__init_data('/size5-0-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_0_20:
            self.__init_data('/size5-0-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_0_25:
            self.__init_data('/size5-0-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_0_30:
            self.__init_data('/size5-0-30.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_0_35:
            self.__init_data('/size5-0-35.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_0_40:
            self.__init_data('/size5-0-40.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_0_45:
            self.__init_data('/size5-0-45.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_0_50:
            self.__init_data('/size5-0-50.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_1_10:
            self.__init_data('/size5-1-10.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_1_15:
            self.__init_data('/size5-1-15.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_1_20:
            self.__init_data('/size5-1-20.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_1_25:
            self.__init_data('/size5-1-25.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_1_30:
            self.__init_data('/size5-1-30.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_1_35:
            self.__init_data('/size5-1-35.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_1_40:
            self.__init_data('/size5-1-40.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_1_45:
            self.__init_data('/size5-1-45.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_size5_1_50:
            self.__init_data('/size5-1-50.dat', '::')

        elif self.kind == self.KIND_PTMATRIX_100_testTime100_0_0:
            self.__init_data('/testTime100-0-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_testTime500_0_0:
            self.__init_data('/testTime500-0-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_testTime1000_0_0:
            self.__init_data('/testTime1000-0-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_testTime2000_0_0:
            self.__init_data('/testTime2000-0-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_testTime10000_0_0:
            self.__init_data('/testTime10000-0-0.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_testTime5000_0_0:
            self.__init_data('/testTime5000-0-0.dat', '::')

        elif self.kind == self.KIND_PTMATRIX_100_gse72056_scRNA_maxMean:
            self.__init_data('/gse72056-scRNA-maxMean.dat', '::')
        elif self.kind == self.KIND_PTMATRIX_100_gse103322_scRNA_maxMean:
            self.__init_data('/gse103322-scRNA-maxMean.dat', '::')

        else:
            raise NotImplementedError()

    def _load_base_data(self):
        return np.load('data/{}/data.npy'.format(self.kind))

    def _split_data(self):
        data = self.data
        n_shot = self.n_shot
        np.random.shuffle(data)

        if self.n_shot == -1:
            # n_shot이 -1일때는 더 sparse하게 전체 레이팅을 9:1로 test train set을 나눈다.
            n_train = int(data.shape[0] * 0.1)
            n_valid = int(n_train * 0.9)

            train_data = data[:n_valid]
            valid_data = data[n_valid:n_train]
            test_data = data[n_train:]

            np.save(self._get_npy_path('train'), train_data)
            np.save(self._get_npy_path('valid'), valid_data)
            np.save(self._get_npy_path('test'), test_data)

        elif self.n_shot == 0:
            # n_shot이 0일때는 다른 알고리즘들처럼 전체 레이팅을 1:9로 test train set을 나눈다.
            n_train = int(data.shape[0] * 0.9)
            n_valid = int(n_train * 0.98)

            train_data = data[:n_valid]
            valid_data = data[n_valid:n_train]
            test_data = data[n_train:]

            np.save(self._get_npy_path('train'), train_data)
            np.save(self._get_npy_path('valid'), valid_data)
            np.save(self._get_npy_path('test'), test_data)

        else:
            # 전체 유저 중에 20%를 일단 test user로 뗍니다.
            test_user_ids = random.sample(
                list(range(self.n_user)), self.n_user // 5)

            train_data = []
            test_data = []
            count_dict = {}
            for i in range(data.shape[0]):
                row = data[i]
                user_id = int(row[0])
                if user_id in test_user_ids:
                    count = count_dict.get(user_id, 0)
                    if count < n_shot:
                        train_data.append(row)
                    else:
                        test_data.append(row)
                    count_dict[user_id] = count + 1
                else:
                    train_data.append(row)

            train_data = np.array(train_data)
            n_valid = int(train_data.shape[0] * 0.98)
            train_data, valid_data = train_data[:n_valid], train_data[n_valid:]

            np.save(self._get_npy_path('train'), train_data)
            np.save(self._get_npy_path('valid'), valid_data)

            test_data = np.array(test_data)
            np.save(self._get_npy_path('test'), test_data)

    def _get_npy_path(self, split_kind):
        return 'data/{}/shot-{}/{}.npy'.format(self.kind, self.n_shot,
                                               split_kind)

    def __init__(self, kind, n_shot=0):
        assert type(n_shot) == int and n_shot >= -1

        _make_dir_if_not_exists('data')
        self._set_kind_and_url(kind)
        #self._download_data_if_not_exists()
        self.n_shot = n_shot

        # 예쁜 형태로 정제된 npy 파일이 없으면, 정제를 수행합니다.
        if not os.path.exists('data/{}/data.npy'.format(kind)):
            self._init_data()
        self.data = self._load_base_data()

        _make_dir_if_not_exists(
            'data/{}/shot-{}'.format(self.kind, self.n_shot))

        self.n_user = int(np.max(self.data[:, 0])) + 1
        self.n_item = int(np.max(self.data[:, 1])) + 1
        self.n_row = self.n_user
        self.n_col = self.n_item

        # split된 데이터가 없으면 split합니다.
        if not os.path.exists(
                self._get_npy_path('train')) or not os.path.exists(
                    self._get_npy_path('valid')) or not os.path.exists(
                        self._get_npy_path('test')):
            self._split_data()

        self.train_data = np.load(self._get_npy_path('train'))
        self.valid_data = np.load(self._get_npy_path('valid'))
        self.test_data = np.load(self._get_npy_path('test'))

    def get_train_data(self):
        return self.train_data

    def get_valid_data(self):
        return self.valid_data

    def get_test_data(self):
        return self.test_data


# if __name__ == '__main__':
#     kind = DatasetManager.KIND_MOVIELENS_100K
#     kind = DatasetManager.KIND_MOVIELENS_1M
#     kind = DatasetManager.KIND_MOVIELENS_10M
#     kind = DatasetManager.KIND_MOVIELENS_20M
#     dataset_manager = DatasetManager(kind)

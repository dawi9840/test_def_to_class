import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import pandas as pd
import numpy as np
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf 


class csv_dataset:

    def __init__(self, dataset_csv_file):
        self.dataframe = pd.read_csv(dataset_csv_file)
        self.val_df = None
        self.train_df = None
        self.val_ds = None
        self.train_ds = None

    def df_to_datasets(self, target):
        # frac(float): 要抽出的比例, random_state：隨機的狀態.
        self.val_df = self.dataframe.sample(frac=0.2, random_state=1337)
        # drop the colum 1 of 'class'.
        self.train_df = self.dataframe.drop(self.val_df.index)

        train_df = self.train_df.copy()
        val_df = self.val_df.copy()

        train_labels = train_df.pop(target)
        val_labels = val_df.pop(target)
        
        # tf.data.Dataset.from_tensor_slices(): 可以獲取列表或數組的切片。
        self.train_ds = tf.data.Dataset.from_tensor_slices((dict(self.train_df), train_labels))
        self.val_ds = tf.data.Dataset.from_tensor_slices((dict(self.val_df), val_labels))

        # shuffle(): 用來打亂數據集中數據順序.
        # buffer_size: https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/661458/
        self.train_ds = self.train_ds.shuffle(buffer_size=len(self.train_ds))
        self.val_ds = self.val_ds.shuffle(buffer_size=len(self.val_ds))

        return self.train_ds, self.val_ds

if __name__ == '__main__':
    # Config TF to use GPU.
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)

    dataset_csv_file = './datasets/heart.csv'
    target_value = "target"

    heart_dataset = csv_dataset(dataset_csv_file)

    train_ds, val_ds = heart_dataset.df_to_datasets(target_value)

    '''# .take(n): get n datas.
    for x, y in train_ds.take(1):
        # tf.print("Input(Features):", x)
        tf.print("Target:", y)'''

    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    print(f'train_ds: \n{train_ds}')


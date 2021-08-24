import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import pandas as pd
import numpy as np
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf 

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def check_csv_contents(file):
    dataframe = pd.read_csv(file)
    # print(f'Top5 datas: \n{df.head()}')
    # print(f'Last5 datas: \n{df.tail()}')
    # print(f'shape: {df.shape}')
    return dataframe

def df_to_dataset(dataframe, target):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target)

    # 使用tf.data.Dataset.from_tensor_slices()方法，我們可以獲取列表或數組的切片。
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    # shuffle(): 用來打亂數據集中數據順序.
    # buffer_size: https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/661458/
    ds = ds.shuffle(buffer_size=len(dataframe))

    return ds

if __name__ == '__main__':

    #*************************************************#
    #               Prepare Dataset.                  #                           
    #*************************************************#

    dataset_csv_file = './datasets/heart.csv'
    df = check_csv_contents(file=dataset_csv_file)
    target_value = "target"

    # frac(float): 要抽出的比例, random_state：隨機的狀態.
    val_df = df.sample(frac=0.2, random_state=1337)
    # drop the colum 1 of 'class'.
    train_df = df.drop(val_df.index)
    # print(f'\nlen of: \ndf: {len(df)}, train_df:{len(train_df)}, val_df: {len(val_df)}')

    df_to_dataset(dataframe=df, target=target_value)

    train_ds = df_to_dataset(dataframe=train_df, target=target_value)
    val_ds = df_to_dataset(dataframe=val_df, target=target_value)

    # .take(n): get n datas.
    for x, y in train_ds.take(1):
        # tf.print("Input(Features):", x)
        tf.print("Target:", y)

    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)
    
    print(f'train_ds: \n{train_ds}')

# test_def_to_class

Modify def() to class() for practicing.

I have compared two of train_ds value when I try to print them.    

Unfortunately, they are not the same.    

I have confused about the result is different.  

Difference1:   
![Screenshot from 2021-08-24 14-20-22](https://user-images.githubusercontent.com/19554347/130567717-18cb1d4f-70d4-4aee-9da1-0e59c6aee5cc.png)

Difference2:   
![Screenshot from 2021-08-24 14-21-05](https://user-images.githubusercontent.com/19554347/130567746-3882cc19-f5ce-4304-9fde-2c26f1b38f9e.png)

**test1_origin.py**   
![Screenshot from 2021-08-24 14-42-51](https://user-images.githubusercontent.com/19554347/130569296-f4a440a4-5bc7-436e-95c5-30ab5a611a66.png)

**test2_modify.py**   
![Screenshot from 2021-08-24 14-43-05](https://user-images.githubusercontent.com/19554347/130569364-56d5cd09-1e35-44e3-92f6-74f27ebca775.png)

**Conda virtual env**

```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install pandas==1.1.3
pip install numpy
pip install tensorflow-gpu==2.6.0
conda install cudnn==8.2.0.53
pip install pydot
```

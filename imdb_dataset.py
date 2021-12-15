# 构建dataset
"""
本模块主要实现：
  1) 构建dataset类,实现批量读取文件数据
  2) 完成数据预处理
"""

import os
import pickle
import re
import zipfile
import nltk
import torch.nn
from torch.utils.data import Dataset, DataLoader
from imdb_vocab import Vocab
from tqdm import tqdm

def unzip(zip_file, data_path):

    """
    解压模块
    zip_file:指定待解压zip文件
    data_path:指定解压缩文件夹
    本模块实现将传入的待解压缩zip文件,解压到指定文件夹中
    """

    # 确定待解压文件是否为zip文件
    sign_zip = zipfile.is_zipfile(zip_file)
    # 如果是zip文件,开始解压缩
    if sign_zip:
        # 获取解压对象属性,"r"读取模式
        file = zipfile.ZipFile(zip_file, "r")
        # 获取待解压zip文件内所有的文件名和文件夹名,并保存在列表中
        # tqdm()用来显示解压进度
        name_list = tqdm(file.namelist())
        name_list.set_description("unzip  " + zip_file)
        # 根据name_list中的文件名,解压在指定文件夹中
        for name in name_list:
            file.extract(name, data_path)
    else:
        print("This is not zip")

def tokenize(sentence):
    """
    进行文本预处理
    """
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<br />", " ", sentence)  # 去掉html标签
    sentence = re.sub("|".join(fileters), " ", sentence)  # 去掉标点符号
    result = [i for i in sentence.split(" ") if len(i) > 0]  # 按空格分词

    return result

class ImdbDataset(Dataset):
    def __init__(self,train = True):

        """
        __init__()函数的作用是读取数据
        """
        # 继承
        super(ImdbDataset,self).__init__()
        # 判断待解压文件夹是否存在
        sign = os.path.exists("./data/download")
        # 如果存在,解压训练集和测试集数据在指定文件夹中
        if not sign:
            unzip("./data/train.zip", "./data/download")
            unzip("./data/test.zip", "./data/download")

        # 保存指定解压文件夹路径,方便后续指定解压训练集数据还是测试集数据
        data_path = "./data/download"  # (1)
        # 根据train参数,选择读取训练集数据还是测试集数据
        data_path += "/train" if train else "/test"  # (2)
        # 将所有文件路径保存在列表中
        self.total_path = []
        # 遍历train或test下的neg和pos数据
        for i in ["/pos", "/neg"]:  # (3)
            """
            (1) "./data/download"
            (2) "./data/download/train"
            (3) "./data/download/train/neg" 和 "./data/download/train/pos"
            (4) "./data/download/train/neg/文件名.txt"
            """
            cur_path = data_path + i
            # 遍历文件夹中的文件,将其文件路径保存在total_path列表中
            # os.path.join(path1, path2)用来将path1路径和path2路径拼接起来
            # os.listdir(path)用来将path文件路径下的所有文件名和文件夹名保存在列表中,与zipfile.namelist差别在于：os方法是对确定存在的路径,zip方法是对待解压的文件路径
            self.total_path += [os.path.join(cur_path, j) for j in os.listdir(cur_path) if j.endswith(".txt")]  # (4)

    def __getitem__(self, idx):

        """
        本模块实现根据索引idx传出对应的数据
        """

        # 根据索引读取对应文件路径
        file = self.total_path[idx]
        # 读取文件数据
        review = tokenize(open(file, "r", encoding = "utf-8").read())
        # 读取文件标签,先对文件地址以"_"分,取最后一部分也就是[-1],再按"."分,取第一个部分就是标签
        label = int(file.split("_")[-1].split(".")[0])
        label = 0 if label < 5 else 1

        return review,label

    def __len__(self):
        return len(self.total_path)

'''def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)

    return reviews, labels

def get_dataloader(train = True):
    # 为遍历测试集和训练集的数据来建立词典,声明一个dataloader方法
    train_batch_size = 512
    test_batch_size = 128
    imdb_dataset = ImdbDataset(train)
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(dataset = imdb_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)'''

# 为建立词典,需要遍历测试集和训练集的数据
if __name__ == "__main__":
    '''vb = Vocab()
    train_Loader = get_dataloader(True)
    for reviews,label in tqdm(train_Loader, total = len(train_Loader)):
        for review in reviews:
            vb.fit(review)
    test_Loader = get_dataloader(False)
    for reviews,label in tqdm(test_Loader, total = len(test_Loader)):
        for review in reviews:
            vb.fit(review)
    vb.build_vocab()
    pickle.dump(vb, open("./models/vocab.pkl", "wb"))'''



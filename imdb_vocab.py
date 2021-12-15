# 建立词典

"""
本模块主要实现
  1) 根据dataset,遍历完所有数据,通过本模块建立词典
  2) 建立word2idx,实现 word(dataset读取的一句话中的word) --> idx(字典中的索引)
  3) 建立idx2word,实现 idx --> word
流程：
  输入 --> [[word, word, ... ,word], [word, word, ... ,word], ... ,[word, word, ... ,word]]
  --> [[索引, 索引, ... ,索引], [索引, 索引, ... ,索引], ... ,[索引, 索引, ... ,索引]]  (索引为在字典中的位置)
  --> [[词向量, 词向量, ... ,词向量], [词向量, 词向量, ... ,词向量], ... ,[词向量, 词向量, ... ,词向量]]  (词向量为词嵌入矩阵中的向量元素,根据对应索引,取出对应的词向量)
  --> 送入RNN网络
"""

import pickle
from tqdm import tqdm
import imdb_dataset
from torch.utils.data import DataLoader

class Vocab:
    # 初始化参数
    UNK_TAG = "<UNK>"
    PAD_TAG = "<PAD>"
    UNK = 0
    PAD = 1
    def __init__(self):
        # 初始化词典和词频词典,词频词典用来根据词频建立dict词典
        # 未知词标注放在词典第一位,填充位标注放在词典第二位
        self.dict = {
            self.UNK_TAG : self.UNK,
            self.PAD_TAG : self.PAD
        }

        # 初始化词频词典
        self.count = {}

    def fit(self,sentence):
        # 建立词频字典
        """
        sentence格式为[word, word, word, ... ,word]
        count词频字典实现 word --> 词频
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count = 1, max_count = None, max_word = None):
        # 建立字典dict
        """
        min_count: 过滤dict词典中出现次数小于min_count的word
        max_count: 过滤dict词典中出现次数大于于min_count的word
        max_word: 控制dict词典的长度
        """
        if min_count is not None:
            self.count = {word : count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word : count for word, count in self.count.items() if count <= max_count}
        if max_word is not None:
            # 同时进行排序
            self.count = dict(sorted(self.count.items(), key = lambda x:x[-1], reverse = True)[:max_word])

        # 根据词频顺序,建立dict词典,用来实现 word(key) --> idx(value)
        for word in self.count:
            self.dict[word] = len(self.dict)

        # 翻转词典,用来实现 idx(key) --> word(value)
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def word2idx(self,sentence,max_len = None):
        # word(sentence中的word) --> idx(dict词典中的idx)

        # 首先判断读取的sentence是否为max_len,若小于则填充<PAD>,将所有句子统一长度
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence += [self.PAD_TAG] * (max_len - len(sentence))

        # 确保为max_len后,实现word --> idx
        return [self.dict.get(word, 0) for word in sentence]

    def idx2word(self,num_list):
        # idx(给定索引列表) --> word(inverse_dict词典中的word值)

        return [self.inverse_dict.get(i, "<UNK>") for i in num_list]

    def __len__(self):
        return len(self.dict)


# 测试代码
def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)

    return reviews, labels


def get_dataloader(train=True):
    # 设置参数
    # train_batch_size = 512
    # test_batch_size = 128
    # batch_size = train_batch_size if train else test_batch_size
    # 根据train参数实例化Dataset类
    dataset = imdb_dataset.ImdbDataset(train)
    imdb_dataloader = DataLoader(dataset=dataset, batch_size=200, shuffle=True, collate_fn=collate_fn)
    # 返回数据迭代器
    return imdb_dataloader


if __name__ == "__main__":
    Vb = Vocab()
    train_loader = get_dataloader(True)
    for review, labels in tqdm(train_loader, total=len(train_loader)):
        for sentence in review:
            Vb.fit(sentence)
    test_loader = get_dataloader(False)
    for review, labels in tqdm(test_loader, total=len(test_loader)):
        for sentence in review:
            Vb.fit(sentence)
    # 遍历完成,建立词典
    Vb.build_vocab()
    print(len(Vb))
    # 保存词典实例类
    # pickle.dump(待保存实例类文件名,指定保存为允许写入的文件路径)
    pickle.dump(Vb, open("./models/vocab.pkl", "wb"))

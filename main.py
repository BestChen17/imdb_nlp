# 从构建词典、Dataset实例,到最后构建LSTM模型和训练 -------- 2021.12.10

"""
声明模块
"""
import torch
import re
import os
import zipfile
import pickle

import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

"""
构建词典
"""
class Vocab():
    """
    构建词典是为了将输入的[[word,word,...,word],[word,word,...,word],...,[word,word,...,word]]词序列转化为
    [[idx,idx,...,idx],[idx,idx,...,idx],...,[idx,idx,...,idx]]的索引序列
    从而在输入LSTM模型时,可以直接按索引在词嵌入矩阵中抽取对应的词向量实现 word --> idx --> 词向量
    即[[word,word,...,word],[word,word,...,word],...,[word,word,...,word]]  ----->
    [[词向量,词向量,...,词向量],[词向量,词向量,...,词向量],...,[词向量,词向量,...,词向量]]
    """
    # 声明未知词位和填充位
    UNK_TAG = "<UNK>"  # 未知词位
    PAD_TAG = "<PAD>"  # 填充位
    UNK = 0  # 对应词典索引
    PAD = 1
    def __init__(self):
        # 初始化word2idx词典和词频词典
        self.dict = {
            self.UNK_TAG : self.UNK,
            self.PAD_TAG : self.PAD
        }

        self.count = {}

    def fit(self,sentence):
        # 根据后续dataloader不断遍历的数据,得到数据中出现的所有词的词频,以此来填充词频词典
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self,min_count = 1,max_count = None,max_word = None):
        # 根据词频词典的词频大小顺序建立word2idx词典
        """
        :param min_count: 用来过滤count词典中词频小于min_count的词
        :param max_count: 用来过滤count词典中词频大于max_count的词
        :param max_word: 用来控制count和dict词典长度
        :return:
        """
        if min_count is not None:
            self.count = {word : count for word,count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word : count for word,count in self.count.items() if count <= max_count}
        if max_word is not None:
            self.count = dict(sorted(self.count.items(),key = lambda x:x[-1],reverse = True)[:max_word])

        # 根据过滤后的词频词典count构建word2idx词典dict,dict中的索引值等于其键在词频词典count中的位置
        for word in self.count:
            self.dict[word] = len(self.dict)

        # 翻转dict词典,用于实现idx2word
        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def word2idx(self,sentence,max_len = 10):
        # 判断输入数据长度是否需要进行填充
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence += [self.PAD_TAG] * (max_len - len(sentence))

        return [self.dict.get(word, 0) for word in sentence]

    def idx2word(self,list):
        return [self.inverse_dict.get(i,"<UNK>") for i in list]

# 词典类实例化
Vb = Vocab( )

"""
构建dataset类
"""

# 解压模块
def unzip(zip_path,down_path):
    """
    :param zip_path: 指定待解压zip文件
    :param down_path: 指定解压文件路径
    :return:
    """
    # 判断待解压文件是否为zip文件
    # zipfile.is_zipfile(path)判断path路径的文件是否为zip文件,返回bool值
    sign = zipfile.is_zipfile(zip_path)
    if sign:
        # 获取zip文件对象属性
        file = zipfile.ZipFile(zip_path,"r")
        # 获取待解压zip文件内所有文件夹和文件的name并保存在列表中
        path_list = file.namelist()
        # 根据zip文件内的文件名解压对应文件在指定解压路径下
        # 对象属性名.extract(zip内文件名,指定解压路径)
        for name in path_list:
            file.extract(name,down_path)

# 文本预处理模块
def tokenizer(sentences):
    # 设置待清洗标点符号集合
    # read()的数据是字符串
    clean = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    # 数据小写化
    sentences = sentences.lower()
    # 去html标签
    # re.sub(待分离的字符串,替换的字符串,原数据)
    sentences = re.sub(r"<br />"," ",sentences)
    # 去除标点符号
    sentences = re.sub("|".join(clean)," ",sentences)
    # 分词
    result = [word for word in sentences.split(" ") if len(word) > 0]
    # 返回结果
    return result

class ImdbDataset(Dataset):
    def __init__(self,train = True):
        """
        :param train: 通过train参数控制构建训练集还是测试集的数据迭代器DataLoader
        """
        # 继承抽象类Dataset
        super(ImdbDataset,self).__init__()
        # 判断指定解压文件夹是否存在
        # os.path.exists(path)判断path路径下的文件或文件夹是否存在,返回bool值
        sign = os.path.exists("./data/download")
        if sign:
            unzip("./data/train.zip","./data/download")
            unzip("./data/test.zip","./data/download")

        # 获取指定解压路径,便于后续路径拼接
        down_path = "./data/download"
        # 根据train参数选择路径
        down_path += "/train" if train else "/test"

        # 遍历指定路径下的每个数据,通过拼接得到全部的路径
        self.total_path = []
        for i in ["/neg","/pos"]:
            file_path = down_path + i
            # 通过os.listdir()遍历解压后文件夹内的全部数据名,并通过拼接存入路径列表,最后得到全部数据的路径
            # os.path.join(A,B)将A和B的路径拼接
            # os.listdir(文件夹路径),将指定文件夹路径内的所有文件名存在列表中
            self.total_path += [os.path.join(file_path,j) for j in os.listdir(file_path) if j.endswith(".txt")]

    def __getitem__(self, idx):
        # 根据索引传出数据
        # 根据索引取出对应文件路径
        file = self.total_path[idx]
        # 读取文件路径下的数据内容,同时进行数据预处理
        review = tokenizer(open(file,"r").read())
        # 获取情感标签
        label = file.split("_")[-1].split(".")[0]
        label = 0 if label < 5 else 1

        return review,label

    def __len__(self):
        return len(self.total_path)

# 构建数据迭代器DataLoader,遍历训练集和测试集中的所有数据,建立词典

def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)

    return reviews, labels

def get_dataloader(train = True):
    # 设置参数
    train_batch_size = 512
    test_batch_size = 128
    batch_size = train_batch_size if train else test_batch_size
    # 根据train参数实例化Dataset类
    imdb_dataset = ImdbDataset(train)
    imdb_dataloader = DataLoader(dataset = imdb_dataset,batch_size = batch_size,shuffle = True,collate_fn = collate_fn)
    # 返回数据迭代器
    return imdb_dataloader

train_loader = get_dataloader(True)
for review,labels in tqdm(train_loader,total = len(train_loader)):
    Vb.fit(review)
test_loader = get_dataloader(False)
for review,labels in tqdm(test_loader,total = len(test_loader)):
    Vb.fit(review)
# 遍历完成,建立词典
Vb.build_vocab()
# 保存词典实例类
# pickle.dump(待保存实例类文件名,指定保存为允许写入的文件路径)
pickle.dump(Vb,open("./models/vocab.pkl","wb"))

"""
构建模型类
"""

# 设置参数
train_batch_size = 512
test_batch_size = 128
# 读取词典实例类
vocab_model = pickle.load(open("./models/vocab.pkl","rb"))

class ImdbModel(torch.nn.Module):
    def __init__(self):
        # 构建Embedding层
        # num_embedding为词典长度,即词典中有多少词,embedding_dim为词向量拓展维度,padding_idx为用什么填充,这里选择词典类里的PAD位
        self.embedding = torch.nn.Embedding(num_embeddings = len(vocab_model),embedding_dim = 200,padding_idx = vocab_model.PAD)
        # 构建LSTM层
        # batch_first使得输出为[batch_size,seq_lens,hidden_size]
        # bidirectional为双向LSTM
        self.lstm = torch.nn.LSTM(input_size = 200,hidden_size = 64,num_layers = 2,batch_first = True,
                                  bidirectional = True,dropout = 0.5)
        # 构建全连接层fc1
        self.fc1 = torch.nn.Linear(64 * 2, 64)
        # 构建全连接层fc2
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self,input):
        # 得到词嵌入输出
        input_embed = self.embedding(input)
        # 输入LSTM
        output,(h_n,c_n) = self.lstm(input_embed)
        # 由于本模型为双向LSTM,需要对前后向的h_n进行拼接
        # h_n保存了每层前后向的最后一个神经元的输出,由于模型为2层双向LSTM,取最后一层即第2层的h_n
        lstm_out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]],dim = -1)
        # 输入全连接层fc1
        fc1_out = self.fc1(lstm_out)
        # 对fc1_out进行relu非线性处理
        fc1_relu = F.relu(fc1_out)
        # 输入全连接层fc2
        fc2_out = self.fc2(fc1_relu)
        # 对fc2_out进行softmax处理,得到概率
        out = F.log_softmax(fc2_out)
        # 返回输出
        return out

"""
训练和测试
"""

# GPU加速模块
def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train(models,eopch):
    # 构建优化器
    optimizer = torch.optim.Adam(models.parameters())
    # 构建损失函数
    criterion = torch.nn.nll_loss()
    # 获取数据迭代器
    train_loader = get_dataloader(True)
    # 设置迭代次数
    for i in range(eopch):
        ba = tqdm(train_loader, total = len(train_loader))
        for idx,(data,target) in enumerate(ba):
            # 梯度清零
            optimizer.zero_grad()
            # 使用GPU加速
            data.to(device())
            target.to(device())
            # 计算模型输出
            output = models(data)
            # 计算损失
            loss = criterion(output,target)
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 进度显示
            ba.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))

def test(models):
    # 初始化损失
    test_loss = 0
    # 初始化预测正确数
    correct = 0
    # 构建损失函数
    criterion = torch.nn.nll_loss()
    # 构建数据迭代器
    test_loader = get_dataloader(False)
    # 在不建立梯度关系下测试
    with torch.no_grad():
        # batch迭代
        for data,target in enumerate(test_loader):
            # 使用GPU加速
            data.to(device())
            target.to(device())
            # 计算模型输出
            output = models(data)
            # 叠加损失
            test_loss += criterion(output, target, reduction='sum').item()
            # 获取最大值的位置,[batch_size,1]
            pred = output.data.max(1, keepdim=True)[1]
            # 获取预测正确数
            correct += pred.eq(target.data.view_as(pred)).sum()
    # 遍历完所有数据后,得出误差率
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    imdb_model = ImdbModel().to(device())
    train(imdb_model, 6)
    test(imdb_model)
import torch
import pickle
import imdb_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from imdb_vocab import Vocab
from tqdm import tqdm

# 设置参数
train_batch_size = 512
test_batch_size = 128
sequence_max_len = 100
# 读取词典实例类
vocab_model = pickle.load(open("./models/vocab.pkl","rb"))

def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    将一个batch里的word序列,全转换为idx序列
    """
    reviews, labels = zip(*batch)
    reviews = torch.LongTensor([vocab_model.word2idx(i, max_len=sequence_max_len) for i in reviews])
    labels = torch.LongTensor(labels)
    return reviews, labels

def get_dataloader(train = True):
    dataset = imdb_dataset.ImdbDataset(train)
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

class ImdbModel(torch.nn.Module):
    def __init__(self):
        # 继承
        super(ImdbModel,self).__init__()
        # 构建Embedding层
        # num_embedding为词典长度,即词典中有多少词,embedding_dim为词向量拓展维度,padding_idx为用什么填充,这里选择词典类里的PAD位
        self.embedding = torch.nn.Embedding(num_embeddings = len(vocab_model),embedding_dim = 200,padding_idx = vocab_model.PAD).to()
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
        out = F.log_softmax(fc2_out, dim = -1)
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
    # nll_loss()在torch.nn.functional下,NLLLoss在torch.nn下
    #criterion = torch.nn.functional.nll_loss()
    # 获取数据迭代器
    train_loader = get_dataloader(True)
    # 设置迭代次数
    for i in range(eopch):
        ba = tqdm(train_loader, total = len(train_loader))
        for idx,(data,target) in enumerate(ba):
            # 梯度清零
            optimizer.zero_grad()
            # 使用GPU加速
            data = data.to(device())
            target = target.to(device())
            # 计算模型输出
            output = models(data)
            # 计算损失
            loss = F.nll_loss(output,target)
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
    # 停止使用dropout
    models.eval()
    # 构建损失函数
    # nll_loss()在torch.nn.functional下,NLLLoss在torch.nn下
    #criterion = torch.nn.functional.nll_loss()
    # 构建数据迭代器
    test_loader = get_dataloader(False)
    # 在不建立梯度关系下测试
    with torch.no_grad():
        # batch迭代
        for data,target in tqdm(test_loader):
            # 使用GPU加速
            data = data.to(device())
            target = target.to(device())
            # 计算模型输出
            output = models(data)
            # 叠加损失
            test_loss += F.nll_loss(output, target, reduction='sum').item()
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
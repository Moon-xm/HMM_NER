#! /usr/bin/env python
# -*-coding:utf-8-*-
# Author: Ming Chen
# create date: 2019-11-27 10:36:59
# description: 使用HMM进行命名实体识别NER
# 包括人名PER,地名LOC机构名ORG,其他O


import numpy as np
from tqdm import tqdm


class HMM_model:
    def __init__(self):
        self.n_tag = 7  # 表示所有标签个数
        self.n_char = 65535  # 所有字符的Unicode编码个数
        self.epsilon = 1e-100  # 无穷小量
        self.tag2idx = {'B-PER': 0,
                        'I-PER': 1,
                        'B-LOC': 2,
                        'I-LOC': 3,
                        'B-ORG': 4,
                        'I-ORG': 5,
                        'O': 6}
        self.idx2tag = dict(zip(self.tag2idx.values(), self.tag2idx.keys()))
        self.A = np.zeros((self.n_tag, self.n_tag))  # 转移概率矩阵,shape:7*7
        self.B = np.zeros((self.n_tag, self.n_char))  # 发射概率矩阵,shape:7*字的个数
        self.pi = np.zeros(self.n_tag)  # 初始隐状态概率,shape：4

    def train(self, corpus_path):
        """
        函数说明： 训练HMM模型, 得到模型参数pi,A,B

        Parameter：
        ----------
            corpus_path - 语料库的位置
        Return:
        -------
            None
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2019-11-27 13:42:50
        """
        with open(corpus_path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()
        print('开始训练数据：')
        for i in tqdm(range(len(lines))):
            if len(lines[i]) == 1:
                continue
            else:
                cur_char, cur_tag = lines[i].split()
                self.B[self.tag2idx[cur_tag]][ord(cur_char)] += 1
                if len(lines[i - 1]) == 1:
                    self.pi[self.tag2idx[cur_tag]] += 1
                    continue
                pre_char, pre_tag = lines[i - 1].split()
                self.A[self.tag2idx[pre_tag]][self.tag2idx[cur_tag]] += 1
        self.pi[self.pi == 0] = self.epsilon  # 防止数据下溢,对数据进行对数归一化
        self.pi = np.log(self.pi) - np.log(np.sum(self.pi))
        self.A[self.A == 0] = self.epsilon
        self.A = np.log(self.A) - np.log(np.sum(self.A, axis=1, keepdims=True))
        self.B[self.B == 0] = self.epsilon
        self.B = np.log(self.B) - np.log(np.sum(self.B, axis=1, keepdims=True))
        np.savetxt('pi', self.pi)
        np.savetxt('A', self.A)
        np.savetxt('B', self.B)
        print('训练完毕！')

    def viterbi(self, Obs):
        """
        函数说明： 使用viterbi算法进行解码

        Parameter：
        ----------
            Obs - 要解码的文本string
        Return:
        -------
            path - 最可能的隐状态路径
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2019-11-27 16:52:42
        """
        T = len(Obs)
        delta = np.zeros((T, self.n_tag))  # shape: 观测文本数量*7
        psi = np.zeros((T, self.n_tag))  # shape: 观测文本数量*7
        delta[0] = self.pi[:] + self.B[:, ord(Obs[0])]  # 初始化
        for i in range(1, T):
            temp = delta[i - 1].reshape(self.n_tag, -1) + self.A  # 这里运用到了矩阵的广播算法
            delta[i] = np.max(temp, axis=0)
            delta[i] = delta[i, :] + self.B[:, ord(Obs[i])]
            psi[i] = np.argmax(temp, axis=0)
        path = np.zeros(T)
        path[T - 1] = np.argmax(delta[T - 1])
        for i in range(T - 2, -1, -1):  # 回溯
            path[i] = int(psi[i + 1][int(path[i + 1])])
        return path

    def predict(self, Obs):
        """
        函数说明： 将文本进行命名实体识别

        Parameter：
        ----------
            Obs - 要识别的文本
        Return:
        -------
            None
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2019-11-27 20:53:23
        """
        T = len(Obs)
        path = self.viterbi(Obs)
        for i in range(T):
            print(Obs[i]+self.idx2tag[path[i]]+'_|', end='')


def main():
    model = HMM_model()
    model.train('corpus/BIO_train.txt')
    s = '林徽因什么理由拒绝了徐志摩而选择梁思成为终身伴侣？' \
        '谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分'
    model.predict(s)


if __name__ == '__main__':
    main()

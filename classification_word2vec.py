# -*- coding: utf-8 -*-
"""
@Time    :2018/6/27 16:22
@Author  :weicm
#@Software: PyCharm
效果：使用机器学习并word2vec完成分析
"""
from sklearn.naive_bayes import BernoulliNB  # s-l贝叶斯分类器
from sklearn.neighbors import KNeighborsClassifier  # s-lKNN分类器
from sklearn.svm import SVC  # s-l支持向量机分类器
from sklearn import cross_validation  # 划分训练集和测试集
from nltk.corpus import PlaintextCorpusReader
from gensim.models import word2vec
from numpy import *
import numpy as np
import time

# 加载自己的语料库
print('加载语料库...')
corpus_root_neg = r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_6000\neg_pre6000"
corpus_root_pos = r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_6000\pos_pre6000"

neg = PlaintextCorpusReader(corpus_root_neg, '.*')
pos = PlaintextCorpusReader(corpus_root_pos, '.*')

documents_neg = [(list(neg.words(fileid)), 0) for fileid in neg.fileids()]
documents_pos = [(list(pos.words(fileid)), 1) for fileid in pos.fileids()]
documents_neg.extend(documents_pos)
documents = documents_neg
random.shuffle(documents)

# 加载词向量训练语料
sentences = word2vec.Text8Corpus(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_6000\merge\6000.txt")
# 训练自己的模型但是准确率不高  得利用大文本数据训练
# model = word2vec.Word2Vec(sentences, size=150, min_count=1)


# 增量训练
# print('加载用微信数据训练完成的预料')
model = word2vec.Word2Vec.load(r"E:\dissertation_weicm_data\weicm\weicm_lw\wx\word2vec_wx")


# print('增量训练...')
# model.train(sentences)

# 增量训练效果示例
# mod =  word2vec.Word2Vec.load(r"D:\weicm\weicm_lw\10Gvec\60\Word60.model")
# mod.similarity(u"不错", u"好")
# sentence=[[u'不错', u'好', u'好'], [u'不错', u'好', u'不错']]
# mod.train(sentence)
# mod.similarity(u"不错", u"好")

# 词向量提取函数
# a=model['房间']  # 示例
def document_vecfea(a):
    # a 为一条评论
    # a=documents[1][0]
    vecfea = {}
    for word in a:
        # 如果词汇在模型内，则输出词向量；否则，置为零向量
        try:
            vecfea[word] = model[word]  # dict　
        except KeyError:
            continue
        # vecfea[word] = zeros(256)
    b = list(vecfea.values())  # dict
    # print(b[0])
    # n * 256维的矩阵
    c = np.array(b)
    # print(c)
    # 将句子中全体词向量的平均值算作其特征值，d 为这条评论的句向量
    d = c.mean(axis=0)
    # print(d)
    return d


print('提取词向量...')
vecfea = [(document_vecfea(d1), c1) for (d1, c1) in documents]

x = [d for (d, c) in vecfea]  # 可供scikit-learn训练的输入
y = [c for (d, c) in vecfea]  # 可供scikit-learn训练的输出（标签）

clf0 = KNeighborsClassifier()
clf1 = BernoulliNB()
clf2 = SVC()


# 交叉验证
start0 = time.clock()
scores0 = cross_validation.cross_val_score(clf0, x, y, cv=3)
KNN_F1 = cross_validation.cross_val_score(clf0, x, y, cv=3, scoring='f1_macro')
KNN_P = cross_validation.cross_val_score(clf0, x, y, cv=3, scoring='precision')
KNN_R = cross_validation.cross_val_score(clf0, x, y, cv=3, scoring='recall')
elapsed0 = (time.clock() - start0) / 3
KNN_Score = scores0.mean()
print('KNNtime:', elapsed0)
print('Acc:\n KNN:{0:.3f} '.format(KNN_Score, ))
print('F1:\n KNN:{0:.3f} '.format(KNN_F1.mean(), ))
print('P:\n KNN:{0:.3f} '.format(KNN_P.mean(), ))
print('P:\n KNN:{0:.3f} '.format(KNN_R.mean(), ))
print(20*"="+'KNN交叉验证结束'+20*"=")
print("\n")
print("\n")

print(20 * "=" + 'NB交叉验证开始' + 20 * "=")
start1 = time.clock()
scores1 = cross_validation.cross_val_score(clf1, x, y, cv=3)
NB_F1 = cross_validation.cross_val_score(clf1, x, y, cv=3, scoring='f1_macro')
NB_P = cross_validation.cross_val_score(clf1, x, y, cv=3, scoring='precision')
NB_R = cross_validation.cross_val_score(clf1, x, y, cv=3, scoring='recall')
elapsed1 = (time.clock() - start1) / 3
NB_Score = scores1.mean()
print('NBtime:', elapsed1)
print('Acc:\n NB:{0:.3f} '.format(NB_Score, ))
print('F1:\n NB:{0:.3f} '.format(NB_F1.mean(), ))
print('P:\n NB:{0:.3f} '.format(NB_P.mean(), ))
print('P:\n NB:{0:.3f} '.format(NB_R.mean(), ))
print(20 * "=" + 'NB交叉验证结束' + 20 * "=")
print("\n")
print("\n")
print("\n")
print(20 * "=" + 'SVM交叉验证开始' + 20 * "=")
start2 = time.clock()
scores2 = cross_validation.cross_val_score(clf2, x, y, cv=3)
SVM_F1 = cross_validation.cross_val_score(clf2, x, y, cv=3, scoring='f1_macro')
SVM_P = cross_validation.cross_val_score(clf2, x, y, cv=3, scoring='precision')
SVM_R = cross_validation.cross_val_score(clf2, x, y, cv=3, scoring='recall')
elapsed2 = (time.clock() - start2) / 3
SVM_Score = scores2.mean()
print('SVMtime:', elapsed2)
print('Acc:\n SVM:{0:.3f} '.format(SVM_Score, ))
print('F1:\n SVM:{0:.3f} '.format(SVM_F1.mean(), ))
print('P:\n SVM:{0:.3f} '.format(SVM_P.mean(), ))
print('R:\n SVM:{0:.3f} '.format(SVM_R.mean(), ))
print(20 * "=" + 'SVM交叉验证结束' + 20 * "=")

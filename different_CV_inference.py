# _*_coding:utf-8 _*_
"""
@Time    :2018/6/19 17:17
@Author  :weicm
#@Software: PyCharm
#效果：用三种机器学习方法并one-hot完成情感分析

"""

import nltk
import random
from nltk.corpus import PlaintextCorpusReader  # 加载自定义语料库
from sklearn.naive_bayes import BernoulliNB  # s-l贝叶斯分类器
from sklearn.neighbors import KNeighborsClassifier  # s-lKNN分类器
from sklearn.svm import SVC  # s-l支持向量机分类器
from sklearn import cross_validation  # 导入交叉验证模块
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import time

# 加载语料库 将变量corpus_root的值设置为自己的语料的文件夹目录
print("=" * 20 + '语料库加载中' + 20 * "=")
# 分词后合并文件夹
corpus_root_reviews = r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_6000\merge\6000"
# 负面分词文件夹
corpus_root_neg = r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_6000\neg_pre6000"
# 正面分词文件夹
corpus_root_pos = r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_6000\pos_pre6000"

# PlaintextCorpusReader 初始化函数的第二个参数可以是需要加载的文件，使用正则表达式
reviews = PlaintextCorpusReader(corpus_root_reviews, '.*')
neg = PlaintextCorpusReader(corpus_root_neg, '.*')
pos = PlaintextCorpusReader(corpus_root_pos, '.*')

documents_neg = [(list(neg.words(fileid)), 0)
                 for fileid in neg.fileids()]
# print(type(documents_neg))
# print(documents_neg[0])
# <class 'list'>
# (['标准间', '太', '差', '房间', '还', '不如', '星', '而且', '设施', '非常', '陈旧', '建议', '酒店', '把', '老', '标准间', '从', '新', '改善'], 0)
documents_pos = [(list(pos.words(fileid)), 1)
                 for fileid in pos.fileids()]
documents_neg.extend(documents_pos)
documents = documents_neg
# 将他们随机打乱
random.shuffle(documents)

# print(documents[0])
# 随机显示正负面评价分词结果

# 统计词频信息
all_words = nltk.FreqDist(w for w in reviews.words())
# print(type(all_words))
# <class 'nltk.probability.FreqDist'>
# 选取特征词超过3000的所有词
# words = [word for word in all_words]
# 打印所有词
word_features = [word for (word, freq) in all_words.most_common(1000)]
# for (word, freq) in all_words.most_common(3000):
#     print(word+":"+str(freq))
# 酒店:8137 是:7453 我:6220 房间:5639 很:4936
# print(len(word_features))
# 3000


# 定义特征提取函数
def document_features(document):
    # 去重
    # (['标准间', '太', '差', '房间', '还', '不如', '星', '而且', '设施', '非常', '陈旧', '建议', '酒店', '把', '老', '标准间', '从', '新', '改善'], 0)
    document_words = set(document)
    features = {}
    # word_features数据形式酒店:8137 是:7453 我:6220 房间:5639 很:4936
    for word in word_features:
        # (word in document_words) 返回的是true or false
        features[word] = (word in document_words)
        # print(features)
        # 将在特征集合里面的词赋值为true，不在的赋值为false
        # {'这': False, '设施': False, '差': False, '只有': False, '环境': False, '比较': False, '还有': False}
    # print(len(features))
    return features


# 特征提取、语料转化为 one-hot
print("*"*20+'转化 one-hot '+"*"*20)
# documents文件样式
# (['标准间', '太', '差', '房间', '还', '不如', '星', '而且', '设施', '非常', '陈旧', '建议', '酒店', '把', '老', '标准间', '从', '新', '改善'], 0)
# document_features(d)返回值形式{'这': False, '设施': False, '差': False, '只有': False, '环境': False, '比较': False, '还有': False}
# 只取其值组成one-hot向量
featuresets = [(list(document_features(d).values()), c) for (d, c) in documents]
# print(featuresets[0])
# ([False, False, False, False, False, False, False, True, False, False, False, False, False],1)

# 使用scikit-learn模块分类
# 可供scikit-learn训练的输入(有监督)
# ([False, False, False, False, False, False, False, True, False, False, False, False, False],1)

# [False, False, False, False, False, False, False, True, False, False, False, False, False] ...... one_hot向量
x = [d for (d, c) in featuresets]
# 可供scikit-learn训练的输出（标签）
# 1 0 0 1 0 1 0 1 0 1 ........  对应情绪标签
y = [c for (d, c) in featuresets]

# K近领
clf0 = KNeighborsClassifier()
# 贝叶斯
clf1 = BernoulliNB()
# 支持向量机
clf2 = SVC()
scores_KNN = []
scores_NB = []
scores_SVM = []
cv = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
for CV in cv:
    # 交叉验证 准确率
    print(20*"="+'KNN交叉验证开始'+20*"=")
    # 定义开始时间 Python time clock() 函数以浮点数计算的秒数返回当前的CPU时间。用来衡量不同程序的耗时，比time.time()更有用。
    start0 = time.clock()
    # 交叉验证
    scores0 = cross_validation.cross_val_score(clf0, x, y, cv=CV)
    # 打印scores0
    # print(scores0)
    # [0.663  0.7155 0.7215]
    elapsed0 = (time.clock() - start0) / 3
    score_mean = scores0.mean()
    scores_KNN.append(score_mean)
    print('KNNtime:', elapsed0)
    print('scores0=',scores0)
    print('Acc:\n KNN:{0:.3f} '.format(score_mean, ))
    print(20*"="+'KNN交叉验证结束'+20*"=")

    print(20 * "=" + 'NB交叉验证开始' + 20 * "=")
    start1 = time.clock()
    scores1 = cross_validation.cross_val_score(clf1, x, y, cv=CV)
    elapsed1 = (time.clock() - start1) / 3
    score_mean1 = scores1.mean()
    scores_NB.append(score_mean1)
    print('NBtime:', scores1)
    print('scores0=', scores0)
    print('Acc:\n NB:{0:.3f} '.format(score_mean1, ))
    print(20 * "=" + 'NB交叉验证结束' + 20 * "=")

    print(20 * "=" + 'SVM交叉验证开始' + 20 * "=")
    start2 = time.clock()
    scores2 = cross_validation.cross_val_score(clf2, x, y, cv=CV)
    elapsed2 = (time.clock() - start2) / 3
    score_mean2 = scores2.mean()
    scores_SVM.append(score_mean2)
    print('SVMtime:', elapsed2)
    print('scores0=', scores2)
    print('Acc:\n SVM:{0:.3f} '.format(score_mean2, ))
    print(20 * "=" + 'SVM交叉验证结束' + 20 * "=")

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
# 绘图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(cv, scores_KNN, label="KNN Score", marker="*")
ax.plot(cv, scores_NB, label="NB Score", marker="o")
ax.plot(cv, scores_SVM, label="SVM Score", marker="+")
ax.set_xlabel("CV")
ax.set_ylabel("score")
ax.set_title(u"不同算法之间的比较", fontproperties=font_set)
ax.set_ylim(0, 1.05)
plt.xlim(1, 20)
ax.legend(framealpha=0.5,loc="best")
plt.show()


# 训练结果
# KNNtime: 52.224403716428725
# scores0= [0.674 0.733 0.703] 打印的是三次分别的分数
# Acc:
#  KNN:0.703


# # 将预测的三次的平均准去率打印出来
# print('Acc:\n KNN:{0:.3f} \n NB:{1:.3f} \n SVM:{2:.3f} \n'
#       .format(scores0.mean(), scores1.mean(), scores2.mean()))

'''

# 交叉验证 F
print('KNN交叉验证...')
scores0 = cross_validation.cross_val_score(clf0, x , y , cv=3, scoring='f1_macro')
print('NB交叉验证...')
scores1 = cross_validation.cross_val_score(clf1, x , y , cv=3, scoring='f1_macro')
print('SVM交叉验证...')
scores2 = cross_validation.cross_val_score(clf2, x , y , cv=3, scoring='f1_macro')
print ( 'Fscore:\n KNN:{0:.3f} \n NB:{1:.3f} \n SVM:{2:.3f} \n'
       .format(scores0.mean(),scores1.mean(),scores2.mean()))


# 交叉验证 Precision
print('KNN交叉验证...')
scores0 = cross_validation.cross_val_score(clf0, x , y , cv=3, scoring='precision')
print('NB交叉验证...')
scores1 = cross_validation.cross_val_score(clf1, x , y , cv=3, scoring='precision')
print('SVM交叉验证...')
scores2 = cross_validation.cross_val_score(clf2, x , y , cv=3, scoring='precision')
print('RF交叉验证...')
scores3 = cross_validation.cross_val_score(clf3, x , y , cv=3, scoring='precision')
print ( 'Precision:\n KNN:{0:.3f} \n NB:{1:.3f} \n SVM:{2:.3f} \n RF:{3:.3f} \n'
       .format(scores0.mean(),scores1.mean(),scores2.mean(),scores3.mean()))


# 交叉验证 Recall
print('KNN交叉验证...')
scores0 = cross_validation.cross_val_score(clf0, x , y , cv=3, scoring='recall')
print('NB交叉验证...')
scores1 = cross_validation.cross_val_score(clf1, x , y , cv=3, scoring='recall')
print('SVM交叉验证...')
scores2 = cross_validation.cross_val_score(clf2, x , y , cv=3, scoring='recall')
print('RF交叉验证...')
scores3 = cross_validation.cross_val_score(clf3, x , y , cv=3, scoring='recall')
print ( 'Recall:\n KNN:{0:.3f} \n NB:{1:.3f} \n SVM:{2:.3f} \n RF:{3:.3f} \n'
       .format(scores0.mean(),scores1.mean(),scores2.mean(),scores3.mean()))

'''

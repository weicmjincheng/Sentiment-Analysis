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
documents_pos = [(list(pos.words(fileid)), 1)
                 for fileid in pos.fileids()]
documents_neg.extend(documents_pos)
documents = documents_neg
random.shuffle(documents)
all_words = nltk.FreqDist(w for w in reviews.words())
feature_numbers = [1000]
scores_KNN = []
scores_NB = []
scores_SVM = []
for feature_number in feature_numbers:
    word_features = [word for (word, freq) in all_words.most_common(feature_number)]
    def document_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features[word] = (word in document_words)
        return features
    featuresets = [(list(document_features(d).values()), c) for (d, c) in documents]
    x = [d for (d, c) in featuresets]
    y = [c for (d, c) in featuresets]

    # K近领
    clf0 = KNeighborsClassifier()
    # 贝叶斯
    clf1 = BernoulliNB()
    # 支持向量机
    clf2 = SVC()

    # 交叉验证 准确率
    print(20*"="+'KNN交叉验证开始'+20*"=")
    # 定义开始时间 Python time clock() 函数以浮点数计算的秒数返回当前的CPU时间。用来衡量不同程序的耗时，比time.time()更有用。
    start0 = time.clock()
    # 交叉验证
    scores0 = cross_validation.cross_val_score(clf0, x, y, cv=4)
    elapsed0 = (time.clock() - start0) / 3
    score_mean = scores0.mean()
    scores_KNN.append(score_mean)
    print('KNNtime:', elapsed0)
    print('scores0=',scores0)
    print('Acc:\n KNN:{0:.3f} '.format(score_mean, ))
    print(20*"="+'KNN交叉验证结束'+20*"=")

    print(20 * "=" + 'NB交叉验证开始' + 20 * "=")
    start1 = time.clock()
    scores1 = cross_validation.cross_val_score(clf1, x, y, cv=4)
    elapsed1 = (time.clock() - start1) / 3
    score_mean1 = scores1.mean()
    scores_NB.append(score_mean1)
    print('NBtime:', scores1)
    print('scores0=', scores0)
    print('Acc:\n NB:{0:.3f} '.format(score_mean1, ))
    print(20 * "=" + 'NB交叉验证结束' + 20 * "=")

    print(20 * "=" + 'SVM交叉验证开始' + 20 * "=")
    start2 = time.clock()
    scores2 = cross_validation.cross_val_score(clf2, x, y, cv=4)
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
ax.plot(feature_numbers, scores_KNN, label="KNN Score", marker="*")
ax.plot(feature_numbers, scores_NB, label="NB Score", marker="o")
ax.plot(feature_numbers, scores_SVM, label="SVM Score", marker="+")
ax.set_xlabel("CV")
ax.set_ylabel("score")
ax.set_title(u"不同算法之间的比较", fontproperties=font_set)
ax.set_ylim(0, 1.05)
plt.xlim(100, 6100)
ax.legend(framealpha=0.5,loc="best")
plt.show()
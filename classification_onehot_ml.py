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
import matplotlib
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
# print(words)    500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000


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

print("\n")
print("\n")
# 交叉验证 准确率
print(20*"="+'KNN交叉验证开始'+20*"=")
# 定义开始时间 Python time clock() 函数以浮点数计算的秒数返回当前的CPU时间。用来衡量不同程序的耗时，比time.time()更有用。
start0 = time.clock()




# 交叉验证
scores0 = cross_validation.cross_val_score(clf0, x, y, cv=4)
KNN_F1 = cross_validation.cross_val_score(clf0, x, y, cv=4, scoring='f1_macro')
KNN_P = cross_validation.cross_val_score(clf0, x, y, cv=4, scoring='precision')
KNN_R = cross_validation.cross_val_score(clf0, x, y, cv=4, scoring='recall')
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
scores1 = cross_validation.cross_val_score(clf1, x, y, cv=4)
NB_F1 = cross_validation.cross_val_score(clf1, x, y, cv=4, scoring='f1_macro')
NB_P = cross_validation.cross_val_score(clf1, x, y, cv=4, scoring='precision')
NB_R = cross_validation.cross_val_score(clf1, x, y, cv=4, scoring='recall')
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
scores2 = cross_validation.cross_val_score(clf2, x, y, cv=4)
SVM_F1 = cross_validation.cross_val_score(clf2, x, y, cv=4, scoring='f1_macro')
SVM_P = cross_validation.cross_val_score(clf2, x, y, cv=4, scoring='precision')
SVM_R = cross_validation.cross_val_score(clf2, x, y, cv=4, scoring='recall')
elapsed2 = (time.clock() - start2) / 3
SVM_Score = scores2.mean()
print('SVMtime:', elapsed2)
print('Acc:\n SVM:{0:.3f} '.format(SVM_Score, ))
print('F1:\n SVM:{0:.3f} '.format(SVM_F1.mean(), ))
print('P:\n SVM:{0:.3f} '.format(SVM_P.mean(), ))
print('R:\n SVM:{0:.3f} '.format(SVM_R.mean(), ))
print(20 * "=" + 'SVM交叉验证结束' + 20 * "=")

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)


# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

label_list = ['ACC', 'F1', 'Precision', 'recall', 'time']    # 横坐标刻度显示值
num_list1 = [0.710, 0.700, 0.657, 0.886, 53.285]      # 纵坐标值1
num_list2 = [0.737, 0.732, 0.686, 0.872, 58.865]      # 纵坐标值2
num_list3 = [0.737, 0.732, 0.686, 0.872, 58.865]
x = range(len(num_list1))
"""
绘制条形图
left:长条形中点横坐标
height:长条形高度
width:长条形宽度，默认值0.8
label:为后面设置legend准备
"""
rects1 = plt.bar(left=x, height=num_list1, width=0.4, alpha=0.8, color='red', label="KNN")
rects2 = plt.bar(left=[i + 0.4 for i in x], height=num_list2, width=0.4, color='green', label="NB")
rects2 = plt.bar(left=[i + 0.4 for i in x], height=num_list3, width=0.4, color='blue', label="SVM")
plt.ylim(0, 1)     # y轴取值范围
plt.ylabel("综合比较")
"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([index + 0.2 for index in x], label_list)
plt.xlabel("不同参考数值比较")
plt.title("算法比较")
plt.legend()     # 设置题注
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
plt.show()

# 绘图
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(feature_numbers, scores_KNN, label="KNN Score", marker="*")
# ax.plot(feature_numbers, scores_NB, label="NB Score", marker="o")
# ax.plot(feature_numbers, scores_SVM, label="SVM Score", marker="+")
# ax.set_xlabel("CV")
# ax.set_ylabel("score")
# ax.set_title(u"不同算法之间的比较", fontproperties=font_set)
# ax.set_ylim(0, 1.05)
# plt.xlim(100, 6100)
# ax.legend(framealpha=0.5,loc="best")
# plt.show()


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


"""
计算AUC值的方法
from sklearn import cross_validation,metrics
from sklearn import svm

train_data,train_target = load(filename)#自定义加载数据函数，返回的是训练数据的数据项和标签项
train_x,test_x,train_y,test_y = cross_validation.train_test_split(train_data,train_target,test_size=0.2,random_state=27)#把训练集按0.2的比例划分为训练集和验证集
#start svm
clf = svm.SVC(C=5.0)
clf.fit(train_x,train_y)
predict_prob_y = clf.predict_proba(test_x)#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
#end svm ,start metrics
test_auc = metrics.roc_auc_score(test_y,prodict_prob_y)#验证集上的auc值
print test_auc
"""










"""
====================语料库加载中====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 6.052016666788571
scores0= [0.733  0.7215 0.719 ]
Acc:
 KNN:0.724
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.734 0.714 0.735]
scores0= [0.733  0.7215 0.719 ]
Acc:
 KNN:0.728
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 8.192259061773905
scores0= [0.8525 0.8425 0.857 ]
Acc:
 KNN:0.851
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 12.028563208722515
scores0= [0.7065 0.7285 0.721 ]
Acc:
 KNN:0.719
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.7365 0.729  0.744 ]
scores0= [0.7065 0.7285 0.721 ]
Acc:
 KNN:0.737
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 18.271733669464783
scores0= [0.8515 0.84   0.859 ]
Acc:
 KNN:0.850
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 18.04355892987319
scores0= [0.67  0.715 0.703]
Acc:
 KNN:0.696
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.742  0.729  0.7545]
scores0= [0.67  0.715 0.703]
Acc:
 KNN:0.742
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 29.208179713296783
scores0= [0.8415 0.8275 0.848 ]
Acc:
 KNN:0.839
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 24.017447871587382
scores0= [0.683  0.7155 0.6925]
Acc:
 KNN:0.697
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.744  0.7355 0.76  ]
scores0= [0.683  0.7155 0.6925]
Acc:
 KNN:0.747
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 40.66499592432492
scores0= [0.834  0.8235 0.837 ]
Acc:
 KNN:0.832
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 29.975240050354728
scores0= [0.6935 0.7005 0.6935]
Acc:
 KNN:0.696
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.747  0.732  0.7615]
scores0= [0.6935 0.7005 0.6935]
Acc:
 KNN:0.747
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 51.86973898612806
scores0= [0.8115 0.804  0.8185]
Acc:
 KNN:0.811
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 35.48586898837925
scores0= [0.68   0.6905 0.699 ]
Acc:
 KNN:0.690
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.7495 0.734  0.762 ]
scores0= [0.68   0.6905 0.699 ]
Acc:
 KNN:0.748
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 63.8199141629506
scores0= [0.8015 0.79   0.798 ]
Acc:
 KNN:0.796
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 41.45260741618457
scores0= [0.676  0.6835 0.691 ]
Acc:
 KNN:0.683
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.7535 0.7295 0.7585]
scores0= [0.676  0.6835 0.691 ]
Acc:
 KNN:0.747
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 75.3682655666001
scores0= [0.776 0.772 0.775]
Acc:
 KNN:0.774
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 47.24935646593864
scores0= [0.6925 0.6735 0.707 ]
Acc:
 KNN:0.691
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.749  0.7305 0.7565]
scores0= [0.6925 0.6735 0.707 ]
Acc:
 KNN:0.745
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 87.41836630038415
scores0= [0.7565 0.755  0.7505]
Acc:
 KNN:0.754
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 53.1451572609188
scores0= [0.6915 0.6715 0.7055]
Acc:
 KNN:0.690
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.7515 0.733  0.756 ]
scores0= [0.6915 0.6715 0.7055]
Acc:
 KNN:0.747
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 99.37288509847046
scores0= [0.7345 0.736  0.736 ]
Acc:
 KNN:0.736
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 58.92177767700347
scores0= [0.688 0.662 0.707]
Acc:
 KNN:0.686
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.7495 0.7315 0.7575]
scores0= [0.688 0.662 0.707]
Acc:
 KNN:0.746
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 112.00071086600171
scores0= [0.7205 0.7225 0.722 ]
Acc:
 KNN:0.722
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 65.89326501802968
scores0= [0.688  0.663  0.7035]
Acc:
 KNN:0.685
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.747  0.7315 0.758 ]
scores0= [0.688  0.663  0.7035]
Acc:
 KNN:0.745
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 126.31678641755889
scores0= [0.7025 0.7105 0.707 ]
Acc:
 KNN:0.707
====================SVM交叉验证结束====================
********************转化 one-hot ********************
====================KNN交叉验证开始====================
KNNtime: 71.15368545865779
scores0= [0.6845 0.661  0.703 ]
Acc:
 KNN:0.683
====================KNN交叉验证结束====================
====================NB交叉验证开始====================
KNNtime: [0.746  0.7315 0.758 ]
scores0= [0.6845 0.661  0.703 ]
Acc:
 KNN:0.745
====================NB交叉验证结束====================
====================SVM交叉验证开始====================
KNNtime: 141.49807309558688
scores0= [0.688 0.697 0.698]
Acc:
 KNN:0.694
====================SVM交叉验证结束====================

"""
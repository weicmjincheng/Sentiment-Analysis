# _*_coding:utf-8 _*_
"""
@Time    :2018/6/19 16:22
@Author  :weicm
#@Software: PyCharm
#效果：分词  去停用词   文本预处理   合并分正负面的评论数据   合并正负面数据

"""

import re
import glob
import jieba


def jieba_stopwords():
    # 加载字典
    jieba.load_userdict('dict.txt')
    # 读入停用词表
    stopwords = [line.strip() for line in open('my_stop_word.txt', 'r', encoding='utf-8').readlines()]

    j = 0;
    # 读入负面情绪文档
    for file in glob.glob(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_2000\neg\*.txt"):
        with open(file, "r+", encoding='utf-8') as f1:
            lines = f1.readlines()
            lines2 = ''.join(lines)
            lines3 = lines2.replace('\n', '')
            lines4 = ''.join(re.findall(u'[\u4e00-\u9fa5]+', lines3))
            seg_list = jieba.cut(lines4)
            seg_list2 = [word for word in seg_list if not word in stopwords]
            seg_list3 = ' '.join(seg_list2)
            # 将分完词的数据保存在neg_jieba文件夹中
            f2 = open(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_2000\neg_jieba\%d.txt" % j, 'w',
                      encoding='utf-8')
            f2.write(seg_list3)
            f2.close()
            j = j + 1;

    i = 0;
    # 读入正面情绪文档
    for file in glob.glob(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_2000\pos\*.txt"):
        with open(file, "r+", encoding='utf-8') as f1:
            lines = f1.readlines()
            lines2 = ''.join(lines)
            lines3 = lines2.replace('\n', '')
            lines4 = ''.join(re.findall(u'[\u4e00-\u9fa5]+', lines3))
            seg_list = jieba.cut(lines4)
            seg_list2 = [word for word in seg_list if not word in stopwords]
            seg_list3 = ' '.join(seg_list2)
            # 分词后结果保存到pos_jieba
            f2 = open(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_2000\pos_jieba\%d.txt" % i, 'w',
                      encoding='utf-8')
            f2.write(seg_list3)
            f2.close()
            i = i + 1;


def merged_pos_neg():
    jieba_stopwords()
    # 将正面情绪所有文档保存在pos.txt单个文档中
    f = open(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_2000\pos.txt", 'w', encoding='utf-8')
    for file in glob.glob(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_2000\pos_jieba\*.txt"):
        with open(file, "r+", encoding='utf-8') as f1:
            lines = f1.readlines()
            lines1 = ''.join(lines)
            f.write(lines1 + '\n')
    f.close()

    # 将负面情绪所有文档保存在neg.txt单个文档中
    f2 = open(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_2000\neg.txt", 'w', encoding='utf-8')
    for file in glob.glob(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_2000\neg_jieba\*.txt"):
        with open(file, "r+", encoding='utf-8') as f3:
            lines2 = f3.readlines()
            lines3 = ''.join(lines2)
            f2.write(lines3 + '\n')
    f2.close()

    # 将正负面
    f4 = open(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_2000\merged.txt", 'w', encoding='utf-8')
    for file in glob.glob(r"E:\dissertation_weicm_data\weicm\weicm_lw\ChnSentiCorp_htl_ba_2000\*.txt"):
        with open(file, "r+", encoding='utf-8') as f5:
            lines4 = f5.readlines()
            lines5 = ''.join(lines4)
            f4.write(lines5 + '\n')
    f4.close()


if __name__ == "__main__":
    merged_pos_neg()


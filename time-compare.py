# _*_coding:utf-8 _*_
"""
@Time    :2018/7/16 11:39
@Author  :weicm
#@Software: PyCharm
"""
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

label_list = ['      Time']    # 横坐标刻度显示值
num_list1 = [53.285]      # 纵坐标值1
num_list2 = [58.865]      # 纵坐标值2
num_list3 = [116.029]
x = range(len(num_list1))
"""
绘制条形图
left:长条形中点横坐标
height:长条形高度
width:长条形宽度，默认值0.8
label:为后面设置legend准备
"""
rects1 = plt.bar(left=x, height=num_list1, width=0.1, alpha=0.8, color='red', label="KNN")
rects2 = plt.bar(left=[i + 0.3 for i in x], height=num_list2, width=0.1, color='green', label="NB")
rects3 = plt.bar(left=[i + 0.6 for i in x], height=num_list3, width=0.1, color='blue', label="SVM")
plt.ylim(0, 150)     # y轴取值范围
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
for rect in rects3:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
plt.show()

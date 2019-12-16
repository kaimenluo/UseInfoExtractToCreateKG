import jieba
import jieba.posseg as pseg
import re
import datetime
import os

f_v_s_path = "../data/feature_vector.npy"
is_exist_f_v = os.path.exists(f_v_s_path)

dict_entity_name_unify = {}

# 从输入的“公司名”中提取主体
def main_extract(input_str,stop_word,d_4_delete,d_city_province):
    # 开始分词并处理
    seg = pseg.cut(input_str)

    # for word, flag in seg:
    #     print('%s %s' % (word, flag))

    seg_lst = remove_word(seg,stop_word,d_4_delete)
    seg_lst = city_prov_ahead(seg_lst, d_city_province)
    # return seg_lst

    result = ''.join(seg_lst)

    if result != input_str:
        if result not in dict_entity_name_unify:
            dict_entity_name_unify[result] = ""
        dict_entity_name_unify[result] = dict_entity_name_unify[result] + "|" + input_str

    return result


# TODO：实现公司名称中地名提前
def city_prov_ahead(seg, d_city_province):
    city_prov_lst = []
    # TODO ...
    for word in seg:
        if word in d_city_province:
            city_prov_lst.append(word)
    seg_lst = [word for word in seg if word not in city_prov_lst]
    return city_prov_lst + seg_lst


# TODO：替换特殊符号
def remove_word(seg, stop_word, d_4_delete):
    # TODO ...
    filter_stop_word = [ word  for word, flag in seg if word not in stop_word]
    seg_lst = [word for word in filter_stop_word if word not in d_4_delete]
    return seg_lst

# 初始化，加载词典
def my_initial():
    fr1 = open(r"../data/dict/co_City_Dim.txt", encoding='utf-8') #城市名
    fr2 = open(r"../data/dict/co_Province_Dim.txt", encoding='utf-8') #省份名
    fr3 = open(r"../data/dict/company_business_scope.txt", encoding='utf-8') # 公司业务范围
    fr4 = open(r"../data/dict/company_suffix.txt", encoding='utf-8') #公司后缀
    #城市名
    lines1 = fr1.readlines()
    d_4_delete = []
    d_city_province = [re.sub(r'(\r|\n)*','',line) for line in lines1] # 将换行符和tab转换成空字符串
    #省份名
    lines2 = fr2.readlines()
    l2_tmp = [re.sub(r'(\r|\n)*','',line) for line in lines2]
    d_city_province.extend(l2_tmp)
    #公司后缀
    lines3 = fr3.readlines()
    l3_tmp = [re.sub(r'(\r|\n)*','',line) for line in lines3]
    lines4 = fr4.readlines()
    l4_tmp = [re.sub(r'(\r|\n)*','',line) for line in lines4]
    d_4_delete.extend(l4_tmp)
    #get stop_word
    fr = open(r'../data/dict/stopwords.txt', encoding='utf-8')
    stop_word = fr.readlines()
    stop_word_after = [re.sub(r'(\r|\n)*','',stop_word[i]) for i in range(len(stop_word))]

    stop_word_after[-1] = stop_word[-1]

    stop_word = stop_word_after
    return d_4_delete,stop_word,d_city_province

# TODO：测试实体统一用例
d_4_delete,stop_word,d_city_province = my_initial()
company_name = "河北银行股份有限公司"
company_name = main_extract(company_name,stop_word,d_4_delete,d_city_province)
# company_name = ''.join(lst)  # 对公司名提取主体部分，将包含相同主体部分的公司统一为一个实体
print(company_name)

"""
步骤 2：实体识别
"""

# 处理test数据，利用开源工具进行实体识别和并使用实体统一函数存储实体

import fool
import pandas as pd
from copy import copy
from tqdm import tqdm, trange

test_data = pd.read_csv('../data/info_extract/test_data.csv', encoding='gb2312', header=0)

# test_data = test_data[0:10]

test_data['ner'] = None
# ner_id = 1001
ner_id = 1000
ner_dict_new = {}  # 存储所有实体
ner_dict_reverse_new = {}  # 存储所有实体

for i in trange(len(test_data)):
    sentence = copy(test_data.iloc[i, 1])
    # TODO：调用fool进行实体识别，得到words和ners结果
    # TODO ...

    words, ners = fool.analysis(sentence)
    ners[0].sort(key=lambda x: x[0], reverse=True)
    for start, end, ner_type, ner_name in ners[0]:
        if ner_type == 'company' or ner_type == 'person':
            # TODO：调用实体统一函数，存储统一后的实体
            # 并自增ner_id
            # TODO ...

            company_main_name = main_extract(ner_name, stop_word, d_4_delete, d_city_province)
            # company_main_name = ''.join(lst)  # 对公司名提取主体部分，将包含相同主体部分的公司统一为一个实体

            if company_main_name not in ner_dict_new:
                # ner_id 从 1001开始
                ner_id += 1
                ner_dict_new[company_main_name] = ner_id


            # 在句子中用编号替换实体名
            sentence = sentence[:start] + ' ner_' + str(ner_dict_new[company_main_name]) + '_ ' + sentence[end - 1:]
    test_data.iloc[i, -1] = sentence

X_test = test_data[['ner']]


# 处理train数据，利用开源工具进行实体识别和并使用实体统一函数存储实体
train_data = pd.read_csv('../data/info_extract/train_data.csv', encoding='gb2312', header=0)

# train_data = train_data[0:10]

train_data['ner'] = None

# if is_exist_f_v == False:

for i in trange(len(train_data)):
    # 判断正负样本
    if train_data.iloc[i, :]['member1'] == '0' and train_data.iloc[i, :]['member2'] == '0':
        sentence = copy(train_data.iloc[i, 1])
        # TODO：调用fool进行实体识别，得到words和ners结果
        # TODO ...
        words, ners = fool.analysis(sentence)
        ners[0].sort(key=lambda x: x[0], reverse=True)
        for start, end, ner_type, ner_name in ners[0]:
            if ner_type == 'company' or ner_type == 'person':
                # TODO：调用实体统一函数，存储统一后的实体
                # 并自增ner_id
                # TODO ...

                company_main_name = main_extract(ner_name, stop_word, d_4_delete, d_city_province)
                # company_main_name = ''.join(lst)  # 对公司名提取主体部分，将包含相同主体部分的公司统一为一个实体

                if company_main_name not in ner_dict_new:
                    ner_id += 1
                    ner_dict_new[company_main_name] = ner_id


                # 在句子中用编号替换实体名
                sentence = sentence[:start] + ' ner_' + str(ner_dict_new[company_main_name]) + '_ ' + sentence[end - 1:]
        train_data.iloc[i, -1] = sentence
    else:
        # 将训练集中正样本已经标注的实体也使用编码替换
        sentence = copy(train_data.iloc[i, :]['sentence'])
        for company_main_name in [train_data.iloc[i, :]['member1'], train_data.iloc[i, :]['member2']]:
            # TODO：调用实体统一函数，存储统一后的实体
            # 并自增ner_id
            # TODO ...

            company_main_name_new = main_extract(company_main_name, stop_word, d_4_delete, d_city_province)
            # company_main_name_new = ''.join(lst)  # 对公司名提取主体部分，将包含相同主体部分的公司统一为一个实体

            if company_main_name_new not in ner_dict_new:
                ner_id += 1
                ner_dict_new[company_main_name_new] = ner_id

            # 在句子中用编号替换实体名
            sentence = re.sub(company_main_name, ' ner_%s_ ' % (str(ner_dict_new[company_main_name_new])), sentence)
        train_data.iloc[i, -1] = sentence

ner_dict_reverse_new = {id:name for name, id in ner_dict_new.items()}

y = train_data.loc[:, ['tag']]
train_num = len(train_data)
X_train = train_data[['ner']]

# 将train和test放在一起提取特征
X = pd.concat([X_train, X_test])

"""
步骤 3：关系抽取

目标：借助句法分析工具，和实体识别的结果，以及文本特征，基于训练数据抽取关系，并存储进图数据库。

本次要求抽取股权交易关系，关系为无向边，不要求判断投资方和被投资方，只要求得到双方是否存在交易关系。

模板建立可以使用“正则表达式”、“实体间距离”、“实体上下文”、“依存句法”等。

答案提交在submit目录中，命名为info_extract_submit.csv和info_extract_entity.csv。

    info_extract_entity.csv格式为：第一列是实体编号，第二列是实体名（实体统一的多个实体名用“|”分隔）
    info_extract_submit.csv格式为：第一列是关系中实体1的编号，第二列为关系中实体2的编号。

示例：

    info_extract_entity.csv

实体编号 	实体名
1001 	小王
1002 	A化工厂


    info_extract_submit.csv

实体1 	实体2
1001 	1003
1002 	1001

"""

"""
练习3：提取文本tf-idf特征

去除停用词，并转换成tfidf向量。
"""
# code
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from pyltp import Segmentor


# 实体符号加入分词词典
with open('../data/user_dict.txt', 'w', encoding='utf-8') as fw:
    for v in ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']:
        fw.write( v + '号企业 ni\n')
fw.close()

# 初始化实例
segmentor = Segmentor()


# 加载模型，加载自定义词典
import os
LTP_DATA_DIR = 'D:\myLTP\ltp_data_v3.4.0'
# LTP_DATA_DIR = '/Users/Badrain/Downloads/ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

# print(cws_model_path)

# segmentor.load_with_lexicon('/Users/Badrain/Downloads/ltp_data_v3.4.0/cws.model', '../data/user_dict.txt')
segmentor.load_with_lexicon(cws_model_path, '../data/user_dict.txt')

# 加载停用词
fr = open(r'../data/dict/stopwords.txt', encoding='utf-8')
stop_word = fr.readlines()
stop_word = [re.sub(r'(\r|\n)*','',stop_word[i]) for i in range(len(stop_word))]


# 分词
# f = lambda x: ' '.join([word for word in segmentor.segment(x) if word not in stop_word and not re.findall(r'ner\_\d\d\d\d\_', word)])
f = lambda x: ' '.join([word for word in segmentor.segment(re.sub(r'ner\_\d\d\d\d\_','',x)) if word not in stop_word])

corpus=X['ner'].map(f).tolist()


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()  # 定一个tf-idf的vectorizer
X_tfidf = vectorizer.fit_transform(corpus).toarray()  # 结果存放在X矩阵

# print("hello ziqing")

"""
练习4：提取句法特征

除了词语层面的句向量特征，我们还可以从句法入手，提取一些句法分析的特征。

参考特征：

1、企业实体间距离

2、企业实体间句法距离

3、企业实体分别和关键触发词的距离

4、实体的依存关系类别

"""

# -*- coding: utf-8 -*-
from pyltp import Parser
from pyltp import Segmentor
from pyltp import Postagger
import networkx as nx
import pylab
import re
import matplotlib.pyplot as plt
from pylab import mpl
from graphviz import Digraph
import numpy as np

postagger = Postagger() # 初始化实例

pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
postagger.load_with_lexicon(pos_model_path, '../data/user_dict.txt')  # 加载模型
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon(cws_model_path, '../data/user_dict.txt')  # 加载模型

SEN_TAGS = ["SBV","VOB","IOB","FOB","DBL","ATT","ADV","CMP","COO","POB","LAD","RAD","IS","HED"]

# parser = Parser()
# parse_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
# parser.load(parse_model_path)

def parse(s, isGraph = False):
    """
    对语句进行句法分析，并返回句法结果
    """
    tmp_ner_dict = {}
    num_lst = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']

    # print(s)

    # 将公司代码替换为特殊称谓，保证分词词性正确
    for i, ner in enumerate(list(set(re.findall(r'(ner\_\d\d\d\d\_)', s)))):
        try:
            tmp_ner_dict[num_lst[i] + '号企业'] = ner
        except IndexError:
            # TODO：定义错误情况的输出
            # TODO ...
            num_lst.append(str(i))
            tmp_ner_dict[num_lst[i] + '号企业'] = ner

        s = s.replace(ner, num_lst[i] + '号企业')

    # print(tmp_ner_dict)

    words = segmentor.segment(s)
    tags = postagger.postag(words)
    parser = Parser()  # 初始化实例

    parse_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
    # parser.load('/Users/Badrain/Downloads/ltp_data_v3.4.0/parser.model')  # 加载模型
    parser.load(parse_model_path)

    arcs = parser.parse(words, tags)  # 句法分析
    arcs_lst = list(map(list, zip(*[[arc.head, arc.relation] for arc in arcs])))

    # 句法分析结果输出
    parse_result = pd.DataFrame([[a, b, c, d] for a, b, c, d in zip(list(words), list(tags), arcs_lst[0], arcs_lst[1])],
                                index=range(1, len(words) + 1))
    parser.release()  # 释放模型

    result = []

    # 实体的依存关系类别
    rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    relation = [arc.relation for arc in arcs]  # 提取依存关系
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
    # for i in range(len(words)):
    #     print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')

    company_list = list(tmp_ner_dict.keys())

    #
    # last_entity = company_list[-1]
    # lw = list(words)
    # last_entity_index = lw.index(last_entity)
    # if last_entity_index != -1:
    #     entity_sentence_type = parse_result.iloc[last_entity_index, -1]
    #     if entity_sentence_type in relation_list:
    #         result.append(relation_list.index(entity_sentence_type))
    #     else:
    #         result.append(-1)
    # else:
    #     result.append(-1)

    str_enti_1 = "一号企业"
    str_enti_2 = "二号企业"
    l_w = list(words)
    is_two_company = str_enti_1 in l_w and str_enti_2 in l_w
    if is_two_company:
        second_entity_index = l_w.index(str_enti_2)
        entity_sentence_type = parse_result.iloc[second_entity_index, -1]
        if entity_sentence_type in SEN_TAGS:
            result.append(SEN_TAGS.index(entity_sentence_type))
        else:
            result.append(-1)
    else:
        result.append(-1)

    if isGraph:
        g = Digraph('测试图片')
        g.node(name='Root')
        for word in words:
            g.node(name=word, fontname="SimHei")

        for i in range(len(words)):
            if relation[i] not in ['HED']:
                g.edge(words[i], heads[i], label=relation[i], fontname="SimHei")
            else:
                if heads[i] == 'Root':
                    g.edge(words[i], 'Root', label=relation[i], fontname="SimHei")
                else:
                    g.edge(heads[i], 'Root', label=relation[i], fontname="SimHei")
        g.view()

    # 企业实体间句法距离
    distance_e_jufa = 0
    if is_two_company:
        distance_e_jufa = shortest_path(parse_result, list(words), str_enti_1, str_enti_2, isGraph=False)
    result.append(distance_e_jufa)

    # 企业实体间距离
    distance_entity = 0
    if is_two_company:
        distance_entity = np.abs(l_w.index(str_enti_1) - l_w.index(str_enti_2))
    result.append(distance_entity)

    # 企业实体分别和关键触发词的距离
    key_words = ["收购", "竞拍", "转让", "扩张", "并购", "注资", "整合", "并入", "竞购", "竞买", "支付", "收购价", "收购价格", "承购", "购得", "购进",
                 "购入", "买进", "买入", "赎买", "购销", "议购", "函购", "函售", "抛售", "售卖", "销售", "转售"]
    # TODO：*根据关键词和对应句法关系提取特征（如没有思路可以不完成）
    # TODO ...

    k_w = None
    for w in words:
        if w in key_words:
            k_w = w
            break

    dis_key_e_1 = -1
    dis_key_e_2 = -1

    if k_w != None and is_two_company:
        k_w = str(k_w)
        # print("k_w", k_w)

        l_w = list(words)
        # dis_key_e_1  = shortest_path(parse_result, l_w, str_enti_1, k_w)
        # dis_key_e_2 = shortest_path(parse_result, l_w, str_enti_2, k_w)

        dis_key_e_1 = np.abs(l_w.index(str_enti_1) - l_w.index(k_w))
        dis_key_e_2 = np.abs(l_w.index(str_enti_2) - l_w.index(k_w))

    result.append(dis_key_e_1)
    result.append(dis_key_e_2)

    return result

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体


def shortest_path(arcs_ret, words, source, target, isGraph = False):
    """
    求出两个词最短依存句法路径，不存在路径返回-1
    arcs_ret：句法分析结果
    source：实体1
    target：实体2
    """
    # G = nx.DiGraph()
    G = nx.Graph()

    # 为这个网络添加节点...
    for i in list(arcs_ret.index):
        G.add_node(i)

    # TODO：在网络中添加带权中的边...（注意，我们需要的是无向边）
    # TODO ...

    for i in range(len(arcs_ret)):
        head = arcs_ret.iloc[i, -2]
        index = i + 1 # 从1开始
        G.add_edge(index, head)

    if isGraph:
        nx.draw(G, with_labels=True)
        plt.savefig("undirected_graph_2.png")
        plt.close()

    try:
        # TODO：利用nx包中shortest_path_length方法实现最短距离提取
        # TODO ...

        source_index = words.index(source) + 1 #从1开始
        target_index = words.index(target) + 1 #从1开始
        distance = nx.shortest_path_length(G, source=source_index, target=target_index)
        # print("'%s'与'%s'在依存句法分析图中的最短距离为:  %s" % (source, target, distance))

        return distance
    except:
        return -1


def shortest_path_1(words, heads, source, target):
    """
    求出两个词最短依存句法路径，不存在路径返回-1
    arcs_ret：句法分析结果
    source：实体1
    target：实体2
    """
    # G = nx.DiGraph()
    G = nx.Graph()

    # 为这个网络添加节点...
    # 添加节点
    for word in words:
        G.add_node(word)

    # TODO：在网络中添加带权中的边...（注意，我们需要的是无向边）
    # TODO ...
    G.add_node('Root')

    # 添加边
    for i in range(len(words)):
        G.add_edge(words[i], heads[i])

    nx.draw(G, with_labels=True)
    plt.savefig("undirected_graph_1.png")
    plt.close()

    try:
        # TODO：利用nx包中shortest_path_length方法实现最短距离提取
        # TODO ...

        distance = nx.shortest_path_length(G, source=source, target=target)
        print("'%s'与'%s'在依存句法分析图中的最短距离为:  %s" % (source, target, distance))
        return distance

    except:
        return -1


corpus_1 = X['ner'].tolist()

# test_data_1 = corpus_1[0]
# result = parse(test_data_1, isGraph = False)
# print(result)

len_train_data = len(train_data)
def get_feature(s):
    """
    汇总上述函数汇总句法分析特征与TFIDF特征
    """
    # TODO：汇总上述函数汇总句法分析特征与TFIDF特征
    # TODO ...
    sen_feature = []
    len_s = len(s)
    for i in trange(len_s):
        f_e = parse(s[i], isGraph = False)
        sen_feature.append(f_e)

    sen_feature = np.array(sen_feature)

    features = np.concatenate((X_tfidf,  sen_feature), axis= 1)

    return features


features = []
if not is_exist_f_v:
    features = get_feature(corpus_1)
    np.save(f_v_s_path, features)
else:
    features = np.load(f_v_s_path)

features_train = features[:len_train_data, :]

# print(features)
"""
利用已经提取好的tfidf特征以及parse特征，建立分类器进行分类任务。
"""

# 建立分类器进行分类
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from seqeval.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB

seed = 2019

# y = np.array(y.values, dtype=np.float)
y = np.array(y.values)
y = y.reshape(-1)
Xtrain, Xtest, ytrain, ytest = train_test_split(features_train,  y, test_size = 0.2, random_state = seed)

def logistic_class(Xtrain, Xtest, ytrain, ytest):
    cross_validator = KFold(n_splits=10, shuffle=True, random_state = seed)

    lr = LogisticRegression(penalty = "l1", solver='liblinear')

    # params = {"penalty":["l1","l2"],
    #              "C":[0.1,1.0,10.0,20.0,30.0,100.0]}

    params = {"C":[0.1,1.0,10.0,15.0,20.0,30.0,40.0,50.0]}

    grid = GridSearchCV(estimator=lr, param_grid = params, cv=cross_validator)
    grid.fit(Xtrain, ytrain)
    print("最优参数为：",grid.best_params_)
    model = grid.best_estimator_
    y_pred = model.predict(Xtest)

    y_test = [str(value) for value in ytest]
    y_pred = [str(value) for value in y_pred]

    proba_value = model.predict_proba(Xtest)
    p = proba_value[:,1]
    print("Logistic=========== ROC-AUC score: %.3f" % roc_auc_score(y_test, p))

    report = classification_report(y_pred=y_pred,y_true=y_test)
    print(report)

    cnf_matrix = confusion_matrix(y_test,y_pred)
    Recall = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]) #召回率
    Accuary = (cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[1,1]+cnf_matrix[1,0]+cnf_matrix[0,1]) #准确率
    Precision = cnf_matrix[1,1]/(cnf_matrix[0,1]+cnf_matrix[1,1]) #精准率
    F1_Score = 2 * Precision * Recall / (Precision + Recall )


    return model

# logistic_class(Xtrain, Xtest, ytrain, ytest)

print("\n")

def NativeBayes_class(Xtrain, Xtest, ytrain, ytest):
    cross_validator = KFold(n_splits=10, shuffle=True, random_state=seed)

    nb = BernoulliNB(alpha=0.0001)

    params = {"alpha": [0.00001, 0.0001,0.001,0.005,0.01,0.1,1]}

    grid = GridSearchCV(estimator=nb, param_grid=params, cv=cross_validator)
    grid.fit(Xtrain, ytrain)
    print("最优参数为：", grid.best_params_)
    model = grid.best_estimator_
    y_pred = model.predict(Xtest)

    y_test = [str(value) for value in ytest]
    y_pred = [str(value) for value in y_pred]

    proba_value = model.predict_proba(Xtest)
    p = proba_value[:, 1]
    print("贝叶斯=========== ROC-AUC score: %.3f" % roc_auc_score(y_test, p))

    report = classification_report(y_pred=y_pred, y_true=y_test)
    print(report)

    cnf_matrix = confusion_matrix(y_test, y_pred)
    Recall = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])  # 召回率
    Accuary = (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (
                cnf_matrix[0, 0] + cnf_matrix[1, 1] + cnf_matrix[1, 0] + cnf_matrix[0, 1])  # 准确率
    Precision = cnf_matrix[1, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1])  # 精准率
    F1_Score = 2 * Precision * Recall / (Precision + Recall)

    print("Recall", Recall)
    print("Accuary", Accuary)
    print("Precision", Precision)

    return model

# nb_model = NativeBayes_class(Xtrain, Xtest, ytrain, ytest)
s_model = logistic_class(Xtrain, Xtest, ytrain, ytest)

features_test = features[len_train_data:, :]
y_pred_test = s_model.predict(features_test)

l_X_test_ner = X_test.values.tolist()

entity_dict = {}
relation_list = []

for i, label in enumerate(y_pred_test):
    if label == 1:
        cur_ner_content = str(l_X_test_ner[i])

        ner_list = list(set(re.findall(r'(ner\_\d\d\d\d\_)',cur_ner_content)))
        if len(ner_list) == 2:
            # print(ner_list)
            r_e_l = []
            for i, ner in enumerate(ner_list):
                split_list = str.split(ner, "_")
                if len(split_list) == 3:
                    ner_id = int(split_list[1])

                    if ner_id in ner_dict_reverse_new:
                        if ner_id not in entity_dict:

                            company_main_name = ner_dict_reverse_new[ner_id]

                            if company_main_name in dict_entity_name_unify:
                                entity_dict[ner_id] = company_main_name + dict_entity_name_unify[company_main_name]
                            else:
                                entity_dict[ner_id] = company_main_name

                        r_e_l.append(ner_id)
            if len(r_e_l) == 2:
                relation_list.append(r_e_l)

# print(entity_dict)
# print(relation_list)
entity_list = [[item[0], item[1]] for item in entity_dict.items()]
# print(entity_list)
pd_enti = pd.DataFrame(np.array(entity_list), columns=['实体编号','实体名'])
pd_enti.to_csv("../submit/info_extract_entity.csv",index=0, encoding='utf_8_sig')

pd_re = pd.DataFrame(np.array(relation_list), columns=['实体1','实体2'])
pd_re.to_csv("../submit/info_extract_submit.csv",index=0,encoding='utf_8_sig')


"""
练习6：操作图数据库

对关系最好的描述就是用图，那这里就需要使用图数据库，目前最常用的图数据库是noe4j，通过cypher语句就可以操作图数据库的增删改查。可以参考“https://cuiqingcai.com/4778.html”。

本次作业我们使用neo4j作为图数据库，neo4j需要java环境，请先配置好环境。

将我们提出的实体关系插入图数据库，并查询某节点的3层投资关系，即三个节点组成的路径（如果有的话）。如果无法找到3层投资关系，请查询出任意指定节点的投资路径。

关于neo4j查询多深度关系节点
https://www.cnblogs.com/sea520/p/11940400.html 

"""


from py2neo import Node, Relationship, Graph

graph = Graph(
    "http://localhost:7474",
    username="neo4j",
    # password="person"
    password = "111111"
)

#清空所有数据对象
graph.delete_all()

for v in relation_list:
    a = Node('Company', name=str(v[0]))
    b = Node('Company', name=str(v[1]))

    # 本次不区分投资方和被投资方，无向图
    r = Relationship(a, 'INVEST', b)
    s = a | b | r
    graph.create(s)
    r = Relationship(b, 'INVEST', a)
    s = a | b | r
    graph.create(s)

"""
match data=(na:Company{name:1001})-[*3]->(nb:Company) return data
"""

import random

result_2 = []
result_3 = []
for value in entity_list:
    ner_id = value[0]
    str_sql_3 = "match data=(na:Company{{name:'{0}'}})-[:INVEST]->(nb:Company)-[:INVEST]->(nc:Company) where na.name <> nc.name return data".format(str(ner_id))
    result_3 = graph.run(str_sql_3).data()
    if len(result_3) > 0:
        break

if len(result_3) > 0:
    print("step1")
    print(result_3)
else:
    print("step2")
    random_index = random.randint(0, len(entity_list) - 1)
    random_ner_id = entity_list[random_index][0]
    str_sql_2 = "match data=(na:Company{{name:'{0}'}})-[*2]->(nb:Company) return data".format(str(random_ner_id))
    result_2 = graph.run(str_sql_2).data()
    print(result_2)

"""
步骤4：实体消歧

解决了实体识别和关系的提取，我们已经完成了一大截，但是我们提取的实体究竟对应知识库中哪个实体呢？下图中，光是“苹果”就对应了13个同名实体。 <img src="../image/baike2.png", width=340, heigth=480>

在这个问题上，实体消歧旨在解决文本中广泛存在的名称歧义问题，将句中识别的实体与知识库中实体进行匹配，解决实体歧义问题。

练习7：

匹配test_data.csv中前25条样本中的人物实体对应的百度百科URL（此部分样本中所有人名均可在百度百科中链接到）。

利用scrapy、beautifulsoup、request等python包对百度百科进行爬虫，判断是否具有一词多义的情况，如果有的话，选择最佳实体进行匹配。

使用URL为‘https://baike.baidu.com/item/’+人名 可以访问百度百科该人名的词条，此处需要根据爬取到的网页识别该词条是否对应多个实体，如下图： <img src="../image/baike1.png", width=440, heigth=480> 如果该词条有对应多个实体，请返回正确匹配的实体URL，例如该示例网页中的‘https://baike.baidu.com/item/陆永/20793929’。

    提交文件：entity_disambiguation_submit.csv
    提交格式：第一列为实体id（与info_extract_submit.csv中id保持一致），第二列为对应URL。
    示例：

实体编号 	URL
1001 	https://baike.baidu.com/item/陆永/20793929
1002 	https://baike.baidu.com/item/王芳/567232

"""

import pandas as pd

# 找出test_data.csv中前25条样本所有的人物名称，以及人物所在文档的上下文内容
test_data = pd.read_csv('../data/info_extract/test_data.csv', encoding = 'gb2312', header=0)


# 观察上下文的窗口大小
window = 20

list_person_content = {}

f = lambda x: ' '.join([word for word in segmentor.segment(x)])
# corpus = X_test_new['ner'].map(f).tolist()
corpus= test_data['sentence'].map(f).tolist()
vectorizer = TfidfVectorizer()  # 定一个tf-idf的vectorizer
X_tfidf = vectorizer.fit_transform(corpus).toarray()  # 结果存放在X矩阵

# 遍历前25条样本
for i in range(25):
    sentence = str(copy(test_data.iloc[i, 1]))
    len_sen = len(sentence)
    words, ners = fool.analysis(sentence)
    ners[0].sort(key=lambda x: x[0], reverse=True)
    for start, end, ner_type, ner_name in ners[0]:
        if ner_type == 'person':
            # TODO：提取实体的上下文
            # print("ner_name", ner_name)
            # print("ner_id", ner_dict_new[ner_name])
            start_index = max(0, start - window)
            end_index = min(len_sen - 1, end - 1 + window)
            left_str = sentence[start_index:start]
            right_str = sentence[end - 1:end_index]

            left_str = ' '.join([word for word in segmentor.segment(left_str)])
            right_str = ' '.join([word for word in segmentor.segment(right_str)])
            new_str = left_str + " " +right_str

            # new_str = left_str + right_str
            # new_str =  ' '.join([word for word in segmentor.segment(new_str)])

            content_vec = vectorizer.transform([new_str])

            ner_id = ner_dict_new[ner_name]
            if ner_id not in list_person_content:
                list_person_content[ner_id] = content_vec


# 利用爬虫得到每个人物名称对应的URL
# TODO：找到每个人物实体的词条内容。

from requests_html import HTMLSession
from requests_html import HTML
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import jieba


list_company_names = [company for value in entity_list for company in str.split(value[1], "|")]
# for value in list_company_names:
#     jieba.add_word(value)

list_person_url = []
url_prefix = "https://baike.baidu.com/item/"
url_error = "https://baike.baidu.com/error.html"

l_p_items = list(list_person_content.items())
len_items = len(l_p_items)

# print(l_p_items)

def get_para_vector(para_elems):
    str_res = ""
    for p_e in para_elems:
        str_res += re.sub(r'(\r|\n)*', '', p_e.text)
    str_res = ' '.join([word for word in jieba.cut(str_res)])
    content_vec = vectorizer.transform([str_res])
    content_vec = content_vec.toarray()[0]
    return content_vec

for index in trange(len_items):
    value = l_p_items[index]

    person_id = value[0]
    vector_entity = csr_matrix(value[1])

    person_name = ner_dict_reverse_new[person_id]

    session = HTMLSession()
    url = url_prefix + person_name
    response = session.get(url)

    url_list = []
    if response.url != url_error:
        para_elems = response.html.find('.para')
        content_vec = get_para_vector(para_elems)
        url_list.append([response.url, content_vec])

        banks = response.html.find('.polysemantList-wrapper')

        if len(banks) > 0:
            banks_child = banks[0]
            persion_links = list(banks_child.absolute_links)
            for link in persion_links:
                r_link = session.get(link)

                if r_link.url == url_error:
                    continue

                para_elems = r_link.html.find('.para')
                content_vec = get_para_vector(para_elems)
                url_list.append([r_link.url, content_vec])

        # max_cos_simi = -1
        # max_index = 0
        # for index, url_item in enumerate(url_list):
        #     cos_simi = cosine_similarity(vector_entity, csr_matrix(url_item[1]))[0]
        #     if cos_simi> max_cos_simi:
        #         max_cos_simi = cos_simi
        #         max_index = index
        #
        # list_person_url.append([person_id, person_name, url_list[max_index][0]])

        vectorizer_list = [item[1] for item in url_list]
        vectorizer_list = csr_matrix(vectorizer_list)
        result = list(cosine_similarity(value[1], vectorizer_list)[0])
        max_index = result.index(max(result))
        list_person_url.append([person_id, person_name, url_list[max_index][0]])

print(list_person_url)
# pd.DataFrame(list_person_url).to_csv('../submit/entity_disambiguation_submit.csv', index=False)

pd_re = pd.DataFrame(np.array(list_person_url), columns=['实体编号','名字','url'])
pd_re.to_csv("../submit/entity_disambiguation_submit.csv",index=0,encoding='utf_8_sig')

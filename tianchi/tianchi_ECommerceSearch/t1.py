from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models, similarities
import pandas as pd
import jieba
import numpy as np
data = pd.read_csv('D:/code/tianchi_ECommerceSearch/data/corpus.tsv', sep='\t', header=None)
data_test = data[1][0:10000]

# 分别使用jieba的搜索引擎模式、全模式、精确模式的api进行分词；使用gensim库的tf和sklearn的tfidf，进行编码，最后使用gensim计算了相似度，然后根据关键词提取出前五个商品名称


# print(data_test)
# tfidf编码
# tfidf = TfidfVectorizer().fit_transform(data_test)
# print(tfidf)
# jieba分词
test1 = jieba.cut(data[1][0], cut_all=True)
print("全模式: " + " | ".join(test1))

test2 = jieba.cut(data[1][0], cut_all=False)
print("精确模式: " + " | ".join(test2))

test3= jieba.cut_for_search(data[1][0])
print("搜索引擎模式:" + " | ".join(test3))

# 将data_test中的数据进行遍历后分词
base_items = [[i for i in jieba.lcut(item)] for item in data_test]
# print(base_items)
# 词典
dictionary = corpora.Dictionary(base_items)

corpus = [dictionary.doc2bow(item) for item in base_items]

tf = models.TfidfModel(corpus)

num_features = len(dictionary.token2id.keys())

index = similarities.MatrixSimilarity(tf[corpus], num_features=num_features)

test_text = "保温杯"
test_words = [word for word in jieba.cut(test_text)]
# print(test_words)

new_vec = dictionary.doc2bow(test_words)
sims = index[tf[new_vec]]
print(list(sims))
unsorted_sims = list(sims)
sorted_sims = list(sims)
sorted_sims.sort(reverse=True)
print(sorted_sims[0:5])

for i in range(0, 5):
    max_index = list(sims).index(sorted_sims[0:5][i])
    print(sorted_sims[0:5][i], max_index, data_test[max_index])

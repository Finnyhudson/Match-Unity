import json
import random
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import jionlp as jio
from TopTexRank import TopTexRank


word2vec_model = KeyedVectors.load('./model/word2vec.model')
data_enhance = TopTexRank(word2vec_model)
print("加载完成！")

def load_json_dataset(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as json_file:
        dataset = json.load(json_file)
    return dataset


def enhance_data(data):
    raw_data = []
    solo_src_data = []
    solo_tgt_data = []
    dual_data = []
    for item in tqdm(data, desc='Enhancing Data'):

        label = item['label']
        content1 = item['content1']
        doc1_title = item['doc1_title'].replace(" ", "")
        content2 = item['content2']
        doc2_title = item['doc2_title'].replace(" ", "")

        # 添加原始样本
        raw_data.append(
            {'label': label, 'content1': content1, 'doc1_title': doc1_title, 'content2': content2, 'doc2_title': doc2_title})

        # 判断两个文本的长度，根据不同情况进行增强
        if len(content1) > 1024 and len(content2) < 1024:
            modified_src = data_enhance(content1, 1024, topic_theta=0)
            modified_tgt = content2
            solo_src_data.append(
                {'label': label, 'content1': modified_src, 'doc1_title': doc1_title, 'content2': modified_tgt,
                 'doc2_title': doc2_title})
        elif len(content2) > 1024 and len(content1) < 1024:
            modified_src = content1
            modified_tgt = data_enhance(content2, 1024, topic_theta=0)
            solo_tgt_data.append(
                {'label': label, 'content1': modified_src, 'doc1_title': doc1_title, 'content2': modified_tgt,
                 'doc2_title': doc2_title})
        elif len(content1) > 1024 and len(content2) > 1024:
            modified_src = data_enhance(content1, 1024, topic_theta=0.02)
            modified_tgt = data_enhance(content2, 1024, topic_theta=0.02)
            dual_data.append(
                {'label': label, 'content1': modified_src, 'doc1_title': doc1_title, 'content2': content2,
                 'doc2_title': doc2_title})
            dual_data.append(
                {'label': label, 'content1': content1, 'doc1_title': doc1_title, 'content2': modified_tgt,
                 'doc2_title': doc2_title})

    return raw_data, solo_src_data, solo_tgt_data, dual_data


datasets = load_json_dataset('CNSE_title_and_key_clean.json')
train, test_val = train_test_split(datasets, test_size=0.4, random_state=0)
del test_val
raw_data, solo_src_data, solo_tgt_data, dual_data = enhance_data(train)

save_path1 = 'CNSE_title_and_key_clean_enhanced_solo_src.json'
with open(save_path1, 'w', encoding='utf-8') as json_file:
    json.dump(solo_src_data, json_file, ensure_ascii=False)

save_path1 = 'CNSE_title_and_key_clean_enhanced_solo_tgt.json'
with open(save_path1, 'w', encoding='utf-8') as json_file:
    json.dump(solo_tgt_data, json_file, ensure_ascii=False)

save_path1 = 'CNSS_title_and_key_clean_enhanced_dual.json'
with open(save_path1, 'w', encoding='utf-8') as json_file:
    json.dump(dual_data, json_file, ensure_ascii=False)

print(1)

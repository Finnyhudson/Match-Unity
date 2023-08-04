import json
import jionlp

from tqdm import tqdm

def clean_content(text):
    text = text.replace(" ", "")
    cleaned_text = jionlp.clean_text(text)
    return cleaned_text

def clean_json_content(json_data):
    cleaned_data = []
    for item in tqdm(json_data, desc="Cleaning JSON"):
        cleaned_item = {
            "label": item["label"],
            "doc1_keywords": item["doc1_keywords"],
            "content1": clean_content(item["content1"]),
            "doc1_title": item["doc1_title"],
            "doc2_keywords": item["doc2_keywords"],
            "content2": clean_content(item["content2"]),
            "doc2_title": item["doc2_title"],
        }
        cleaned_data.append(cleaned_item)
    return cleaned_data


# 读取JSON文件
with open('CNSE_title_and_key.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 清洗内容并替换原始文本
cleaned_json_data = clean_json_content(json_data)

# 将更新后的数据写入新的JSON文件
with open('CNSE_title_and_key_clean.json', 'w', encoding='utf-8') as file:
    json.dump(cleaned_json_data, file, ensure_ascii=False)

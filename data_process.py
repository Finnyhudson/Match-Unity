import json


def split_records(dataset):
    data = []

    with open(dataset, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                label, doc_id1, doc_id2, title1, title2, content1, content2, keywords1, keywords2, main_keywords1, main_keywords2, ner_keywords1, ner_keywords2, ner1, ner2, category1, category2, time1, time2 = line.split(
                    '|')

                record = {
                    'label': label,
                    'doc1_keywords': keywords1,
                    'doc1_title': title1,
                    "content1": content1,
                    'doc2_keywords': keywords2,
                    'doc2_title': title2,
                    "content2": content2,
                }

                data.append(record)

    return data


def save_as_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


data = split_records("same_event_doc_pair.txt")[1:]
# 保存数据为 JSON 文件
output_file = 'CNSE_title_and_key.json'
save_as_json(data, output_file)

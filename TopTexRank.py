# -*- coding=utf-8 -*-

import os
import json
from gensim.models import KeyedVectors
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import jiojio
from jionlp import logging
from jionlp.rule import clean_text
from jionlp.rule import check_any_chinese_char
from jionlp.gadget import split_sentence
from jionlp.dictionary import stopwords_loader
from jionlp.dictionary import idf_loader

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class TopTexRank(object):
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model
        self.key_to_index = word2vec_model.key_to_index
        self.unk_topic_prominence_value = 0.

    def _prepare(self):
        self.pos_name = set(sorted(list(jiojio.pos_types()['model_type'].keys())))
        # self.pos_name = set(['a', 'ad', 'an', 'c', 'd', 'f', 'm', 'n', 'nr', 'nr1', 'nrf', 'ns', 'nt',
        #                      'nz', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'vd', 'vi', 'w', 'wx', 'x'])

        self.strict_pos_name = ['a', 'n', 'nr', 'ns', 'nt', 'nx', 'nz',
                                'ad', 'an', 'vn', 'vd', 'vx']
        jiojio.init(pos_rule=True, pos=True)

        # Load IDF and calculate its average value for OOV words.
        self.idf_dict = idf_loader()
        self.median_idf = sorted(self.idf_dict.values())[len(self.idf_dict) // 2]

        # Read the stop words file.
        self.stop_words = stopwords_loader()

        # Load LDA Model Parameters
        self._lda_prob_matrix()

    def _lda_prob_matrix(self):
        """ Read probability distribution file related to the LDA model and compute the probability distribution for unknown words """
        # Read the p(topic|word) probability distribution file. Due to the large size of the LDA model,
        # it's not convenient to load and calculate the probability p(topic|word). Hence, we haven't considered
        # the p(topic|doc) probability, which may result in less accuracy. However, considering that the default LDA model
        # has topic_num == 100, the convergence of the LDA model on the predicted documents has minimal impact on the results
        # (the larger the topic_num, the less the impact).

        dict_dir_path = os.path.join(os.path.dirname(os.path.dirname(DIR_PATH)), 'dictionary')

        with open(os.path.join(dict_dir_path, 'topic_word_weight.json'),
                  'r', encoding='utf8') as f:
            self.topic_word_weight = json.load(f)
        self.word_num = len(self.topic_word_weight)

        # Read the p(word|topic) probability distribution file
        with open(os.path.join(dict_dir_path, 'word_topic_weight.json'),
                  'r', encoding='utf8') as f:
            self.word_topic_weight = json.load(f)
        self.topic_num = len(self.word_topic_weight)

        self._topic_prominence()  # Pre-compute topic prominence

    def __call__(self, text, summary_length=200, tr_theta=5, lead_3_weight=1.5,
                 topic_theta=0.2, allow_topic_weight=True):

        if type(text) is not str:
            raise ValueError('type of `text` should only be str')
        try:
            if self.unk_topic_prominence_value == 0.:
                self._prepare()

            if len(text) <= summary_length:
                return text

            # step 0: clean text
            text = clean_text(text)

            # step 1: Sentence segmentation and sentence-by-sentence noise cleaning
            sentences_list = split_sentence(text)

            # step 2: Tokenization and Part-of-Speech Tagging
            sentences_segs_dict = dict()
            for idx, sen in enumerate(sentences_list):
                if not check_any_chinese_char(sen):  # If there are no Chinese characters, then skip.
                    continue

                sen_segs = jiojio.cut(sen)
                sentences_segs_dict.update({sen: [idx, sen_segs, 0]})

            # step 3: Calculate the Weight for Each Sentence
            textrank_weight = self._textrank_weight(sentences_segs_dict)

            textrank_weight_list = []
            for sen, sen_segs in sentences_segs_dict.items():
                # Lda weight
                if allow_topic_weight:
                    topic_weight = 0.0
                    for item in sen_segs[1]:
                        topic_weight += self.topic_prominence_dict.get(
                            item[0], self.unk_topic_prominence_value)
                    topic_weight = topic_weight / len(sen_segs[1])
                else:
                    topic_weight = 0.0

                # TextRank weight
                tr_weight = textrank_weight[sen]

                sen_weight = topic_weight * topic_theta + tr_weight * tr_theta
                textrank_weight_list.append([topic_weight * topic_theta, tr_weight])

                # Sentence length exceeds the limit, weight reduction
                if len(sen) < 15 or len(sen) > 70:
                    sen_weight = 0.7 * sen_weight

                # LEAD-3 weight
                if sen_segs[0] < 3:
                    sen_weight *= lead_3_weight

                sen_segs[2] = sen_weight

            # step 4: Recalculate the weights according to the MMR algorithm and filter out undesired sentences
            sentences_info_list = sorted(sentences_segs_dict.items(),
                                         key=lambda item: item[1][2], reverse=True)

            mmr_list = list()
            for sentence_info in sentences_info_list:
                # Calculate the similarity with existing sentences
                sim_ratio = self._mmr_similarity(sentence_info, mmr_list)
                sentence_info[1][2] = (1 - sim_ratio) * sentence_info[1][2]
                mmr_list.append(sentence_info)

            # step 5: Sort the sentences by importance and select a certain number of sentences as the summary
            if len(sentences_info_list) == 1:
                return sentences_info_list[0][0]
            total_length = 0
            summary_list = list()
            for idx, item in enumerate(sentences_info_list):
                if len(item[0]) + total_length > summary_length:
                    if idx == 0:
                        return item[0]
                    else:
                        # Sort by sequence number
                        summary_list = sorted(
                            summary_list, key=lambda item: item[1][0])
                        summary = ''.join([item[0] for item in summary_list])
                        return summary
                else:
                    summary_list.append(item)
                    total_length += len(item[0])
                    if idx == len(sentences_info_list) - 1:
                        summary_list = sorted(
                            summary_list, key=lambda item: item[1][0])
                        summary = ''.join([item[0] for item in summary_list])
                        return summary

            return text[:summary_length]
        except Exception as e:
            logging.error('the text is illegal. \n{}'.format(e))
            return ''

    def _textrank_weight(self, sentences_segs_dict):
        """Calculate the TextRank weight of sentences"""
        G = nx.Graph()
        for sen, sen_segs in sentences_segs_dict.items():
            G.add_node(sen)
        for sen1, sen_segs1 in sentences_segs_dict.items():
            for sen2, sen_segs2 in sentences_segs_dict.items():
                if sen1 != sen2:
                    similarity = self._calculate_similarity(sen_segs1[1], sen_segs2[1])
                    G.add_edge(sen1, sen2, weight=similarity)

        # Calculate the TextRank score of each sentence
        textrank_scores = nx.pagerank(G)

        # Map the TextRank scores to each sentence
        textrank_weight = {}
        for sen, score in textrank_scores.items():
            textrank_weight[sen] = score

        return textrank_weight

    def _calculate_similarity(self, segs1, segs2):
        """Calculate the similarity between two sentences (based on Word2Vec and cosine similarity)"""
        # Check if word vectors are empty
        vector1 = [self.word2vec_model[word] for word, _ in segs1 if word in self.key_to_index]
        vector2 = [self.word2vec_model[word] for word, _ in segs2 if word in self.key_to_index]

        # Check if word vectors are empty
        if not vector1 or not vector2:
            return 0.0

        # Calculate the similarity between sentences (cosine similarity)
        similarity = cosine_similarity([np.mean(vector1, axis=0)], [np.mean(vector2, axis=0)])[0][0]

        # Return sentence similarity
        return similarity

    def _mmr_similarity(self, sentence_info, mmr_list):
        """Calculate the similarity of each sentence with previous sentences"""
        sim_ratio = 0.0
        notional_info = set([item[0] for item in sentence_info[1][1]
                             if item[1] in self.strict_pos_name])
        if len(notional_info) == 0:
            return 1.0
        for sen_info in mmr_list:
            no_info = set([item[0] for item in sen_info[1][1]
                           if item[1] in self.strict_pos_name])
            common_part = notional_info & no_info
            if sim_ratio < len(common_part) / len(notional_info):
                sim_ratio = len(common_part) / len(notional_info)
        return sim_ratio

    def _topic_prominence(self):
        """Calculate the salience of each word"""
        init_prob_distribution = np.array([self.topic_num for i in range(self.topic_num)])

        topic_prominence_dict = dict()
        for word in self.topic_word_weight:
            conditional_prob_list = list()
            for i in range(self.topic_num):
                if str(i) in self.topic_word_weight[word]:
                    conditional_prob_list.append(self.topic_word_weight[word][str(i)])
                else:
                    conditional_prob_list.append(1e-5)
            conditional_prob = np.array(conditional_prob_list)

            tmp_dot_log_res = np.log2(np.multiply(conditional_prob, init_prob_distribution))
            kl_div_sum = np.dot(conditional_prob, tmp_dot_log_res)  # kl divergence
            topic_prominence_dict.update({word: float(kl_div_sum)})

        tmp_list = [i[1] for i in tuple(topic_prominence_dict.items())]
        max_prominence = max(tmp_list)
        min_prominence = min(tmp_list)
        for k, v in topic_prominence_dict.items():
            topic_prominence_dict[k] = (v - min_prominence) / (max_prominence - min_prominence)

        self.topic_prominence_dict = topic_prominence_dict

        # Calculate the salience of unknown words. Since stop words have already been filtered out, there's no need to consider their salience here.
        tmp_prominence_list = [item[1] for item in self.topic_prominence_dict.items()]
        self.unk_topic_prominence_value = sum(tmp_prominence_list) / (2 * len(tmp_prominence_list))


if __name__ == '__main__':
    title = '全面解析拜登外交政策，至关重要的“对华三条”'
    text = '''腾讯体育11月16日，在2016-17赛季CBA常规赛第7轮的一场焦点战中，广东队在主场末节发力，以100-93逆转击败卫冕冠军四川队，终结对手3连胜的同时也继续在主场保持不败。技术统计本场比赛四川队在篮板球上以48-41占优，但35次3分出手仅命中10个，助攻数上也以12-20落后。广东队斯隆得到29分10次助攻，布泽尔得到21分13个篮板，易建联末节砍下13分，全场得到24分10个篮板5次抢断。四川队方面，哈达迪得到25分10个篮板，约什3分球18中7，罚球11中6得到全场最高的41分19个篮板和5次封盖，刘炜11分5次助攻。比赛焦点本场之后，两队CBA历史共交锋8次。广东队以6胜2负占据上风。约什-史密斯本场状态神勇砍下全场最高的41分。布泽尔也延续了上一场比赛复苏的状态，本场砍下21分13个篮板。比赛回放首节比赛开场后双方比分紧咬先后4次战平，本节还有5分17秒约什-史密斯篮筐正面出手命中了个人在CBA的首个3分球，紧接着他防下易建联的内线进攻后又助攻孟达投中顶弧的3分球，四川队打出6-0后以15-11反超。斯隆马上突破还以一球，广东队还以一波8-2后以19-17反超。周鹏侧翼过掉徐韬后的抛投被约什直接扇掉，约什在内线的护框功力凸显，赵睿此后的上篮也明显受到他的干扰没能碰到篮圈。约什此后再次3分出手偏出，首节比赛结束广东队在主场以21-20暂时领先。第二节比赛开始后双方均遣出双外援。布泽尔率先投中两球，不过约什此后又命中一记3分，四川队以29-25反超。约什连续肆意出手导致球队陷入被动，虽然在进攻端出手不合理但他随后又直接冒掉了布泽尔的上篮，双方陷入混战比分上也一直胶着。哈达迪面对布泽尔正面飚中3分，朱芳雨连续在左侧底角接球出手终于命中3分，随后斯隆也投中追身3分，广东队以42-38略为扩大领先。广东队控制最后一攻但斯隆被张春军断掉，刘炜最后时刻压哨的上篮得手。上半场比赛结束广东队取得48-44的微弱领先。哈达迪、约什和斯隆半场比赛各自得到15分。下半场比赛开始后约什命中了个人本场第3个3分球，双方分差再次回到1分的差距。本节还有9分20秒，刘炜在一次突破中被赵睿踢到脚后跟扭伤了脚踝，只能被哈达迪和张春军搀扶着离场。本节还有7分12秒王汝恒给约什送出空接妙传，约什起飞的暴扣被周鹏犯规，约什两罚不中，但此后约什再度命中3分双方战成57平。本节还有2分54秒赵睿反击中加速上篮被补防的约什飞身盖掉。布泽尔手感颇佳连续在罚球线附近跳投得手。前3节比赛结束广东队以69-66暂时领先。末节比赛刘炜回到场上，约什突然爆发他连续飚中两记3分球个人接连得到10分，四川队以一波12-0开局，一举取得78-69的领先。哈达迪出场后广东队开始疯狂追分，易建联的手感也彻底打开，广东队回敬一波疯狂的23-6之后以94-86反超。约什重新回到场上也未能挽回劣势，最终广东队在主场以100-93击败四川队取得主场3连胜。双方首发名单广东队首发阵容：易建联、周鹏、任俊飞、斯隆、赵睿四川队首发阵容：孟达、刘炜、张春军、蔡晨、哈达迪下轮对阵11月18日常规赛第8轮，广东主场对阵新疆，四川客场对阵深圳。'''

    word2vec_model = KeyedVectors.load('./model/word2vec.model')
    print("加载完成！")
    cse_obj = TopTexRank(word2vec_model)
    summary = cse_obj(text, topic_theta=0.02, summary_length=400)
    print('summary_0.2topic: ', summary)
    summary = cse_obj(text, topic_theta=0, summary_length=400)
    print('summary_no_topic: ', summary)
    summary = cse_obj(text, topic_theta=0.5, summary_length=400)
    print('summary_0.5topic: ', summary)

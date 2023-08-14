# Match-Unity
Code for Long-form Text Matching with  Knowledge Complementarity

## Requirement
To run the code successfully, you will need (just install the most recent version):
  Pytorch
  Graph-tool
  Jionlp

## How to use
1.Start by downloading the original datasets CNSE and CNSS from the repository: https://github.com/BangLiu/ArticlePairMatching.
2.Utilize data_process.py to generate JSON files and employ data_clean.py for text refinement.
3.Enhance your dataset using sup_data_advancing.py, which employs TopicTexRank to create augmented data.
4.Train an interactive matching model for long-form titles using bert_title.py.
5.Employ sentence_longformer.py to train a representational matching model tailored for lengthy contents.
6.Combine the results from steps 4 and 5 using bert_longformer.py to establish the Match-Unity model.

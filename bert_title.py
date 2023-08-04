import datetime
import json
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
import gensim
import jieba
import jionlp
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import BertTokenizer
import torch.optim.lr_scheduler as lr_scheduler
from tqdm.auto import tqdm
from pathlib import Path
import transformers
import logging

logger = logging.getLogger(__name__)

SOURCE_FIELD_NAME = 'src'
TARGET_FIELD_NAME = 'tgt'
LABEL_FIELD_NAME = 'label'


def set_seed():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


class TextMatchingDataset(Dataset):
    def __init__(self, data, bert_tokenizer):
        self.data = data
        self.bert_tokenizer = bert_tokenizer
        all_labels = list(set([e[LABEL_FIELD_NAME] for e in self.data]))
        self.label_to_idx = {e: i for i, e in enumerate(sorted(all_labels))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]

        # 获取src文本的主题词
        # src_key_words = instance['doc1_keywords']
        src_title = instance['doc1_title'].replace(" ", "")

        # 获取tgt文本的主题词
        # tgt_key_words = instance['doc2_keywords']
        tgt_title = instance['doc2_title'].replace(" ", "")

        bert_tokens = self.bert_tokenizer.encode_plus(src_title, tgt_title,
                                                      add_special_tokens=True, truncation=True, max_length=256,
                                                      padding='max_length')

        label = self.label_to_idx[instance[LABEL_FIELD_NAME]]

        bert_tensor_inputs = {
            "token_ids": torch.tensor(bert_tokens["input_ids"]),
            "mask": torch.tensor(bert_tokens["attention_mask"]),
            "token_type_ids": torch.tensor(bert_tokens["token_type_ids"]),
        }

        inputs = bert_tensor_inputs, label
        return inputs


class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.model = transformers.BertModel.from_pretrained("bert_pretrain")
        for i, layer in enumerate(self.model.encoder.layer):
            if i < 5:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, inputs):
        input_ids = inputs['bert_token_ids']
        attention_mask = inputs['bert_mask']
        token_type_ids = inputs['bert_type_ids']

        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 取CLS位置的向量

        return cls_output


class BERTSemanticSimilarity(nn.Module):
    def __init__(self, hidden_dim):
        super(BERTSemanticSimilarity, self).__init__()
        self.bert = BERTModel().cuda()
        #self.MLP = NonLinearClassifier(hidden_dim).cuda()
        self.fc = nn.Linear(hidden_dim, 2)

    def load_model_params(self, model_path):
        model_state_dict = torch.load(model_path)
        self.bert.load_state_dict(model_state_dict['model_state_dict']['bert'])
        self.fc.load_state_dict(model_state_dict['model_state_dict']['fc'])


    def forward(self, bert_dict):
        bert_output = self.bert(bert_dict)
        outputs = self.fc(bert_output)
        return outputs


class NonLinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=2, p=0.2):
        super(NonLinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.Dropout(p=p),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, n_label),
        )

    def forward(self, x):
        return self.net(x)

def load_json_dataset(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as json_file:
        dataset1 = json.load(json_file)
    return dataset1


if __name__ == '__main__':
    import logging

    transformers.logging.set_verbosity_error()
    # 获取当前日期和时间
    now_time = datetime.datetime.now().strftime('%m%d')

    # 设置日志文件名
    log_file = "logs/bert_training_{}.log".format(now_time)
    logging.basicConfig(filename=log_file, level=logging.INFO)

    batch_size = 64
    num_epochs = 20
    set_seed()
    #longformer_tokenizer = BertTokenizer.from_pretrained("../longformer_pretrain_models")
    bert_tokenizer = BertTokenizer.from_pretrained("bert_pretrain")

    CNSE = load_json_dataset("data/CNSE_title_and_key_clean.json")
   # CNSE_enhanced = load_json_dataset("../data/already_key_words_and_title/CNSE_title_and_key_clean_enhanced_final.json")
    train, test_val = train_test_split(CNSE, test_size=0.4, random_state=0)
    valid, test = train_test_split(test_val, test_size=0.5, random_state=0)
    
    del test_val
    train_data = TextMatchingDataset(train, bert_tokenizer)
    valid_data = TextMatchingDataset(valid, bert_tokenizer)
    test_data = TextMatchingDataset(test, bert_tokenizer)

    dataloaders = {'train': DataLoader(train_data, batch_size=batch_size,pin_memory=True,
                                       shuffle=True, num_workers=4, drop_last=True), 
                   'validation': DataLoader(valid_data, batch_size=batch_size, pin_memory=True,
                                            shuffle=True, num_workers=4, drop_last=True),
                   'test': DataLoader(test_data, batch_size=batch_size,pin_memory=True,
                                      shuffle=True, num_workers=4, drop_last=True),
                   }

    model = BERTSemanticSimilarity(hidden_dim=768).cuda()
    # model.load_model_params('model/07.20/bert_title_CNSS_epoch_1-valid_acc0.895-test_acc0.895')

    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7, eps=1e-8)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-7)
    #total_steps = (len(train_data) // batch_size) * num_epochs
    # scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=2e-5,total_steps=total_steps)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    scaler = GradScaler()
    loss_func = nn.CrossEntropyLoss()
    now_time = datetime.datetime.now().strftime('%m.%d')
    model_out = Path('model/' + now_time)

    if not model_out.exists():
        os.mkdir(model_out)

    best_test_acc = 0.89
    for epoch in range(num_epochs):
        model.train()
        train_bar = tqdm(dataloaders['train'])
        n = 0
        train_correct = 0
        train_error = 0

        for i, train_sample in enumerate(train_bar):
            bert_dict = {}
            bert_tokens = train_sample[0]
            labels = train_sample[1].cuda()

            bert_dict['bert_token_ids'] = bert_tokens['token_ids'].cuda()
            bert_dict['bert_mask'] = bert_tokens['mask'].cuda()
            bert_dict['bert_type_ids'] = bert_tokens['token_type_ids'].cuda()

            optimizer.zero_grad()

            with autocast():
                output = model(bert_dict)
                loss = loss_func(output, labels)

            train_correct += (output.argmax(1) == labels).sum().item()
            train_error += (output.argmax(1) != labels).sum().item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            

            # 更新tqdm中的显示信息
            train_bar.set_postfix({'Acc': train_correct / (train_error + train_correct), 'loss': loss.item()})
            train_bar.update()

            if (i + 1) % 200 == 0:
                log_message = 'epoch : {} batch : {},Train Acc: {},  loss : {}'.format(epoch + 1, i, train_correct / (
                        train_error + train_correct), loss.item())
                tqdm.write(log_message)
                logging.info(log_message)


        model.eval()
        valid_bar = tqdm(dataloaders['validation'])
        dev_predictions = []
        dev_targets = []

        for valid_sample in valid_bar:
            with torch.no_grad():
                bert_dict = {}
                bert_tokens = valid_sample[0]
                labels = valid_sample[1].cuda()

                bert_dict['bert_token_ids'] = bert_tokens['token_ids'].cuda()
                bert_dict['bert_mask'] = bert_tokens['mask'].cuda()
                bert_dict['bert_type_ids'] = bert_tokens['token_type_ids'].cuda()

                output = model(bert_dict)
                loss = loss_func(output, labels)
                dev_predictions.extend(output.argmax(1).cpu().numpy())
                dev_targets.extend(labels.cpu().numpy())

        dev_acc = accuracy_score(dev_targets, dev_predictions)
        dev_f1 = f1_score(dev_targets, dev_predictions)
        tqdm.write("Epoch: {}, Validation Acc: {}, Validation F1: {}".format(epoch + 1, dev_acc, dev_f1))
        logging.info("Epoch: {}, Validation Acc: {}, Validation F1: {}".format(epoch + 1, dev_acc, dev_f1))
        scheduler.step(dev_acc)

        model.eval()
        test_bar = tqdm(dataloaders['test'])
        test_predictions = []
        test_targets = []
        
        for test_sample in test_bar:
            with torch.no_grad():
                bert_dict = {}
                bert_tokens = test_sample[0]
                labels = test_sample[1].cuda()

                bert_dict['bert_token_ids'] = bert_tokens['token_ids'].cuda()
                bert_dict['bert_mask'] = bert_tokens['mask'].cuda()
                bert_dict['bert_type_ids'] = bert_tokens['token_type_ids'].cuda()

                output = model(bert_dict)
                loss = loss_func(output, labels)
                test_predictions.extend(output.argmax(1).cpu().numpy())
                test_targets.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_targets, test_predictions)
        test_f1 = f1_score(test_targets, test_predictions)
        tqdm.write("Epoch: {}, Test Acc: {}, Test F1: {}".format(epoch + 1, test_acc, test_f1))
        logging.info("Epoch: {}, Test Acc: {}, Test F1: {}".format(epoch + 1, test_acc, test_f1))

        model_path = model_out / "bert_title_CNSS_epoch_{0}-valid_acc{1:.3f}-test_acc{2:.3f}".format(epoch + 1, dev_acc,
                                                                                                  test_acc)
        if dev_acc > best_test_acc:
            best_test_acc = dev_acc
            model_state_dict = {
                'bert': model.bert.state_dict(),
                'fc': model.fc.state_dict(),
            }
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            logging.info("Model saved at: {}".format(model_path))


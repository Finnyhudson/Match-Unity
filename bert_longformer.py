import datetime
import json
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import BertTokenizer
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


class TextMatchingDataset(Dataset):
    def __init__(self, data, longformer_tokenizer, bert_tokenizer, seqlen):
        self.data = data
        self.seqlen = seqlen
        self.longformer_tokenizer = longformer_tokenizer
        self.bert_tokenizer = bert_tokenizer
        all_labels = list(set([e[LABEL_FIELD_NAME] for e in self.data]))
        self.label_to_idx = {e: i for i, e in enumerate(sorted(all_labels))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        # 获取src文本的主题词

        src_title = instance['doc1_title'].replace(" ", "")
        src_tokens = self.longformer_tokenizer(instance['content1'], max_length=self.seqlen, padding="max_length",
                                               truncation=True)

        src_tokens_tensor = torch.tensor(src_tokens["input_ids"])
        src_mask_tensor = torch.tensor(src_tokens["attention_mask"])

        # 获取tgt文本的主题词
        tgt_title = instance['doc2_title'].replace(" ", "")
        tgt_tokens = self.longformer_tokenizer(instance['content2'], max_length=self.seqlen, padding="max_length",
                                               truncation=True)

        bert_tokens = self.bert_tokenizer.encode_plus(src_title, tgt_title,
                                                      add_special_tokens=True, truncation=True, max_length=256,
                                                      padding='max_length')

        tgt_tokens_tensor = torch.tensor(tgt_tokens["input_ids"])
        tgt_mask_tensor = torch.tensor(tgt_tokens["attention_mask"])

        label = self.label_to_idx[instance[LABEL_FIELD_NAME]]

        longformer_tensor_inputs = {
            "src_token_ids": src_tokens_tensor,
            "src_mask": src_mask_tensor,
            "tgt_token_ids": tgt_tokens_tensor,
            "tgt_mask": tgt_mask_tensor,

        }
        bert_tensor_inputs = {
            "token_ids": torch.tensor(bert_tokens["input_ids"]),
            "mask": torch.tensor(bert_tokens["attention_mask"]),
            "token_type_ids": torch.tensor(bert_tokens["token_type_ids"]),

        }

        inputs = longformer_tensor_inputs, bert_tensor_inputs, label
        return inputs


class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.model = transformers.BertModel.from_pretrained("../bert_pretrain_models")

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
        # self.fc = nn.Linear(hidden_dim, 2)

    def load_b_model_params(self, b_model_path):
        b_model_state_dict = torch.load(b_model_path)
        self.bert.load_state_dict(b_model_state_dict['model_state_dict']['bert'])
        # self.fc.load_state_dict(b_model_state_dict['model_state_dict']['fc'])
        # self.prediction.load_state_dict(b_model_state_dict['model_state_dict']['prediction'])

    def forward(self, bert_dict):
        bert_output = self.bert(bert_dict)
        # outputs = self.fc(bert_output)
        return bert_output


class Longformer(nn.Module):
    def __init__(self, config_path):
        super(Longformer, self).__init__()
        self.config_path = config_path
        config = transformers.LongformerConfig.from_pretrained(config_path)
        config.attention_mode = 'sliding_chunks_no_overlap'
        config.attention_window = 170
        config.num_labels = 2
        config.gradient_checkpointing = True
        self.model_config = config
        self.model = transformers.LongformerModel.from_pretrained(config_path, config=config)

        for i, layer in enumerate(self.model.encoder.layer):
            if i < 5:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, inputs):
        src_input_ids = inputs['src_token_ids']
        src_attention_mask = inputs['src_mask']
        tgt_input_ids = inputs['tgt_token_ids']
        tgt_attention_mask = inputs['tgt_mask']

        src_global_attention_mask = torch.zeros(
            src_input_ids.shape, dtype=torch.long, device=src_input_ids.device)

        src_global_attention_mask[:, 0] = 1  # global attention for the first token

        tgt_global_attention_mask = torch.zeros(
            tgt_input_ids.shape, dtype=torch.long, device=tgt_input_ids.device)
        tgt_global_attention_mask[:, 0] = 1  # global attention for the first token

        # use Bert inner Pooler
        src_outputs = self.model(src_input_ids, attention_mask=src_attention_mask,
                                 global_attention_mask=src_global_attention_mask)[0][:, 0, :]
        tgt_outputs = self.model(tgt_input_ids, attention_mask=tgt_attention_mask,
                                 global_attention_mask=tgt_global_attention_mask)[0][:, 0, :]

        return src_outputs, tgt_outputs


class LongformerSemanticSimilarity(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super(LongformerSemanticSimilarity, self).__init__()
        self.embedding = Longformer('../longformer_pretrain_models').cuda()

        self.bert = BERTSemanticSimilarity(768).cuda()
        # self.bert.load_b_model_params('../model/bert_title_CNSE_epoch_22-valid_acc0.852-test_acc0.847')
        self.bert.load_b_model_params('../model/bert_title_CNSS_epoch_28-valid_acc0.905-test_acc0.904')
        # # 冻结BERT模型的参数，不进行更新
        for param in self.bert.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(hidden_dim * 4, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_dim * 3, hidden_dim)
        self.prediction = nn.Linear(hidden_dim, 2)

    def load_b_model_params(self, b_model_path):
        b_model_state_dict = torch.load(b_model_path)
        self.embedding.load_state_dict(b_model_state_dict['model_state_dict']['embedding'])
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.fc.load_state_dict(b_model_state_dict['model_state_dict']['fc'])
        self.fc.requires_grad = False  # 冻结 fc 层
        # complementarity_dict = torch.load('../model/bert_content_CNSE_epoch_3-valid_acc0.874-test_acc0.870')
        # self.dense.load_state_dict(complementarity_dict['model_state_dict']['dense'])
        # self.prediction.load_state_dict(complementarity_dict['model_state_dict']['prediction'])

        # self.prediction.load_state_dict(b_model_state_dict['model_state_dict']['prediction'])

    def forward(self, tensor_inputs):
        src_outputs, tgt_outputs = self.embedding(tensor_inputs)
        combined_output = torch.cat(
            [src_outputs, tgt_outputs, torch.abs(src_outputs - tgt_outputs),
             src_outputs * tgt_outputs],
            dim=1)
        combined_output = self.fc(combined_output)
        combined_output = self.dropout(combined_output)

        title_outputs = self.bert(bert_dict)
        combined_output = self.dense(
            torch.cat([title_outputs, combined_output, torch.abs(combined_output-title_outputs)], dim=1))
        #combined_output = self.dense(torch.cat([title_outputs, combined_output, title_outputs*combined_output], dim=1))

        combined_output = self.prediction(combined_output)

        return title_outputs, combined_output


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
    log_file = "../logs/training_{}.log".format(now_time)
    logging.basicConfig(filename=log_file, level=logging.INFO)

    seqlen = 1024
    batch_size = 32
    num_epochs = 10
    set_seed()
    longformer_tokenizer = BertTokenizer.from_pretrained("../longformer_pretrain_models")
    bert_tokenizer = BertTokenizer.from_pretrained("../bert_pretrain_models")

    dataset = load_json_dataset("../data/already_key_words_and_title/CNSS_title_and_key_clean.json")
    dataset2 = load_json_dataset("../data/already_key_words_and_title/CNSS_title_and_key_clean_enhanced_solo_src.json")
    dataset3 = load_json_dataset("../data/already_key_words_and_title/CNSS_title_and_key_clean_enhanced_solo_tgt.json")
    dataset4 = load_json_dataset("../data/already_key_words_and_title/CNSS_title_and_key_clean_enhanced_dual.json")

    train, test_val = train_test_split(dataset, test_size=0.4, random_state=0)
    valid, test = train_test_split(test_val, test_size=0.5, random_state=0)
    del test_val

    train_data = TextMatchingDataset(train, longformer_tokenizer,bert_tokenizer, 1024)
    enhanced_data1 = TextMatchingDataset(dataset2, longformer_tokenizer, bert_tokenizer, 1024)
    enhanced_data2 = TextMatchingDataset(dataset3, longformer_tokenizer, bert_tokenizer, 1024)
    enhanced_data3 = TextMatchingDataset(dataset4, longformer_tokenizer, bert_tokenizer, 1024)
    valid_data = TextMatchingDataset(valid, longformer_tokenizer, bert_tokenizer, 1024)
    test_data = TextMatchingDataset(test, longformer_tokenizer, bert_tokenizer, 1024)

    dataloaders = {'train': DataLoader(train_data+enhanced_data1+enhanced_data2+enhanced_data3, batch_size=batch_size, pin_memory=True,
                                       shuffle=True, num_workers=0, drop_last=True),
                   'validation': DataLoader(valid_data, batch_size=batch_size, pin_memory=True,
                                            shuffle=True, num_workers=0, drop_last=True),
                   'test': DataLoader(test_data, batch_size=batch_size, pin_memory=True,
                                      shuffle=True, num_workers=0, drop_last=True),
                   }

    model = LongformerSemanticSimilarity(hidden_dim=768).cuda()
    # b_model_path = '../model/longformer_CNSE_epoch_9-valid_acc0.8328125'
    b_model_path = '../model/longformer_CNSS_enhanced_epoch_18-valid_acc0.8978365384615384'
    model.load_b_model_params(b_model_path)

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_param = list(model.named_parameters())
    # optimizer_grouped_parameters = [
    #     {'params': [param for name, param in optimizer_param if
    #               not any((name in no_decay_name) for no_decay_name in no_decay)], 'weight_decay': 0.01},
    #    {'params': [param for name, param in optimizer_param if
    #                 any((name in no_decay_name) for no_decay_name in no_decay)], 'weight_decay': 0.0}
    # ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, eps=1e-8)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()
    now_time = datetime.datetime.now().strftime('%m.%d')
    model_out = Path('../model/' + now_time)
    if not model_out.exists():
        os.mkdir(model_out)

    loss1 = nn.CrossEntropyLoss()
    # 在训练循环中使用tqdm

    best_test_acc = 0.93
    for epoch in range(num_epochs):
        model.train()
        train_bar = tqdm(dataloaders['train'], desc='Epoch {}/{}'.format(epoch + 1, num_epochs), leave=False)

        train_correct = 0
        train_error = 0
        total_loss = 0

        for i, train_sample in enumerate(train_bar):
            data_dict = {}
            bert_dict = {}
            data = train_sample[0]
            bert_tokens = train_sample[1]
            labels = train_sample[2].cuda()

            data_dict['src_token_ids'] = data['src_token_ids'].cuda()
            data_dict['src_mask'] = data['src_mask'].cuda()
            data_dict['tgt_token_ids'] = data['tgt_token_ids'].cuda()
            data_dict['tgt_mask'] = data['tgt_mask'].cuda()
            bert_dict['bert_token_ids'] = bert_tokens['token_ids'].cuda()
            bert_dict['bert_mask'] = bert_tokens['mask'].cuda()
            bert_dict['bert_type_ids'] = bert_tokens['token_type_ids'].cuda()

            optimizer.zero_grad()

            with autocast():
                title_outputs, content_outputs = model(data_dict)
                loss = loss1(content_outputs, labels)
                # loss = loss2(content_outputs, title_outputs)
                # loss = distillation(content_outputs, labels, title_outputs, temp=5.0, alpha=0.7)

                total_loss += loss.item()

            train_correct += (content_outputs.argmax(1) == labels).sum().item()
            train_error += (content_outputs.argmax(1) != labels).sum().item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 更新tqdm中的显示信息
            train_bar.set_postfix({'Acc': train_correct / (train_error + train_correct), 'total_loss': total_loss})
            train_bar.update()

            if (i + 1) % 100 == 0:
                log_message = 'epoch : {} batch : {},Train Acc: {},  loss : {}'.format(epoch + 1, i, train_correct / (
                        train_error + train_correct), loss.item())
                tqdm.write(log_message)
                logging.info(log_message)
                lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
                tqdm.write('lr: {}'.format(lr))

        scheduler.step()
        model.eval()
        valid_bar = tqdm(dataloaders['validation'])
        dev_predictions = []
        dev_targets = []

        for valid_sample in valid_bar:
            with torch.no_grad():
                data_dict = {}
                bert_dict = {}
                data = valid_sample[0]
                bert_tokens = valid_sample[1]
                labels = valid_sample[2].cuda()

                data_dict['src_token_ids'] = data['src_token_ids'].cuda()
                data_dict['src_mask'] = data['src_mask'].cuda()
                data_dict['tgt_token_ids'] = data['tgt_token_ids'].cuda()
                data_dict['tgt_mask'] = data['tgt_mask'].cuda()
                bert_dict['bert_token_ids'] = bert_tokens['token_ids'].cuda()
                bert_dict['bert_mask'] = bert_tokens['mask'].cuda()
                bert_dict['bert_type_ids'] = bert_tokens['token_type_ids'].cuda()

                title_outputs, content_outputs = model(data_dict)
                dev_predictions.extend(content_outputs.argmax(1).cpu().numpy())
                dev_targets.extend(labels.cpu().numpy())


        dev_acc = accuracy_score(dev_targets, dev_predictions)
        dev_f1 = f1_score(dev_targets, dev_predictions)
        tqdm.write("Epoch: {}, Validation Acc: {}, Validation F1: {}".format(epoch + 1, dev_acc, dev_f1))
        logging.info("Epoch: {}, Validation Acc: {}, Validation F1: {}".format(epoch + 1, dev_acc, dev_f1))


        model.eval()
        test_bar = tqdm(dataloaders['test'])
        test_predictions = []
        test_targets = []

        for test_sample in test_bar:
            with torch.no_grad():
                data_dict = {}
                bert_dict = {}
                data = test_sample[0]
                bert_tokens = test_sample[1]
                labels = test_sample[2].cuda()

                data_dict['src_token_ids'] = data['src_token_ids'].cuda()
                data_dict['src_mask'] = data['src_mask'].cuda()
                data_dict['tgt_token_ids'] = data['tgt_token_ids'].cuda()
                data_dict['tgt_mask'] = data['tgt_mask'].cuda()
                bert_dict['bert_token_ids'] = bert_tokens['token_ids'].cuda()
                bert_dict['bert_mask'] = bert_tokens['mask'].cuda()
                bert_dict['bert_type_ids'] = bert_tokens['token_type_ids'].cuda()

                title_outputs, content_outputs = model(data_dict)
                test_predictions.extend(content_outputs.argmax(1).cpu().numpy())
                test_targets.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_targets, test_predictions)
        test_f1 = f1_score(test_targets, test_predictions)
        tqdm.write("Epoch: {}, Test Acc: {}, Test F1: {}".format(epoch + 1, test_acc, test_f1))
        logging.info("Epoch: {}, Test Acc: {}, Test F1: {}".format(epoch + 1, test_acc, test_f1))

        model_path = model_out / "bert_content_epoch_{0}-valid_acc{1:.3f}-test_acc{2:.3f}".format(epoch + 1, dev_acc,
                                                                                                  test_acc)
        if dev_acc > best_test_acc:
            best_test_acc = dev_acc
            model_state_dict = {
                'embedding': model.embedding.state_dict(),
                'fc': model.fc.state_dict(),
                'dense': model.dense.state_dict(),
                'prediction': model.prediction.state_dict()
            }

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, model_path)

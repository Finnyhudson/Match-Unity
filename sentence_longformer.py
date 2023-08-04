import datetime
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    def __init__(self, data, longformer_tokenizer, seqlen):
        self.data = data
        self.seqlen = seqlen
        self.longformer_tokenizer = longformer_tokenizer
        all_labels = list(set([e[LABEL_FIELD_NAME] for e in self.data]))
        self.label_to_idx = {e: i for i, e in enumerate(sorted(all_labels))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        # 获取src文本的主题词
        src_tokens = self.longformer_tokenizer(instance['content1'], max_length=self.seqlen, padding="max_length",
                                               truncation=True)

        src_tokens_tensor = torch.tensor(src_tokens["input_ids"])
        src_mask_tensor = torch.tensor(src_tokens["attention_mask"])

        # 获取tgt文本的主题词
        tgt_tokens = self.longformer_tokenizer(instance['content2'], max_length=self.seqlen, padding="max_length",
                                               truncation=True)

        tgt_tokens_tensor = torch.tensor(tgt_tokens["input_ids"])
        tgt_mask_tensor = torch.tensor(tgt_tokens["attention_mask"])

        label = self.label_to_idx[instance[LABEL_FIELD_NAME]]

        longformer_tensor_inputs = {
            "src_token_ids": src_tokens_tensor,
            "src_mask": src_mask_tensor,
            "tgt_token_ids": tgt_tokens_tensor,
            "tgt_mask": tgt_mask_tensor,

        }

        inputs = longformer_tensor_inputs, label
        return inputs


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
        # for param in self.model.embeddings.parameters():
        #     param.requires_grad = False

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
        self.fc = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.prediction = nn.Linear(hidden_dim, 2)

    def load_b_model_params(self, b_model_path):
        b_model_state_dict = torch.load(b_model_path)
        self.embedding.load_state_dict(b_model_state_dict['model_state_dict']['embedding'])
        # for param in self.embedding.parameters():
        #     param.requires_grad = False

        self.fc.load_state_dict(b_model_state_dict['model_state_dict']['fc'])
        # self.fc.requires_grad = False  # 冻结 fc 层
        self.prediction.load_state_dict(b_model_state_dict['model_state_dict']['prediction'])
        # self.prediction.requires_grad = False  # 冻结 fc 层


    def forward(self, tensor_inputs):
        src_outputs, tgt_outputs = self.embedding(tensor_inputs)
        combined_output = torch.cat(
            [src_outputs, tgt_outputs, torch.abs(src_outputs - tgt_outputs), src_outputs * tgt_outputs], dim=1)
        outputs = self.prediction(self.dropout(self.fc(combined_output)))
        return outputs


def load_json_dataset(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as json_file:
        dataset1 = json.load(json_file)
    return dataset1

def main():
    batch_size = 64
    num_epochs = 20
    set_seed()
    tokenizer = BertTokenizer.from_pretrained("../longformer_pretrain_models")

    dataset = load_json_dataset("../data/already_key_words_and_title/CNSE_title_and_key_clean.json")
    dataset2 = load_json_dataset("../data/already_key_words_and_title/CNSE_title_and_key_clean_enhanced_train_1.json")
    # dataset3 = load_json_dataset("../data/already_key_words_and_title/SS")
    train, test_val = train_test_split(dataset, test_size=0.4, random_state=0)
    valid, test = train_test_split(test_val, test_size=0.5, random_state=0)
    del test_val, train

    train_data = TextMatchingDataset(dataset2, tokenizer, 1024)
    valid_data = TextMatchingDataset(valid, tokenizer, 1024)
    test_data = TextMatchingDataset(test, tokenizer, 1024)

    dataloaders = {'train': DataLoader(train_data, batch_size=batch_size, pin_memory=True,
                                       shuffle=True, num_workers=0, drop_last=True),
                   'validation': DataLoader(valid_data, batch_size=batch_size, pin_memory=True,
                                            shuffle=True, num_workers=0, drop_last=True),
                   'test': DataLoader(test_data, batch_size=batch_size, pin_memory=True,
                                      shuffle=True, num_workers=0, drop_last=True),
                   }

    model = LongformerSemanticSimilarity(hidden_dim=768).cuda()
    # model.load_b_model_params('../model/longformer_CNSS_epoch_18-valid_acc0.888755980861244')

    # 定义优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    scaler = GradScaler()
    loss_func = nn.CrossEntropyLoss()
    now_time = datetime.datetime.now().strftime('%m.%d')
    model_out = Path('../model/' + now_time)
    if not model_out.exists():
        os.mkdir(model_out)

    best_valid_acc = 0.83
    for epoch in range(num_epochs):
        model.train()
        train_bar = tqdm(dataloaders['train'])
        train_correct = 0
        train_error = 0

        for i, train_sample in enumerate(train_bar):
            data_dict = {}
            data = train_sample[0]
            labels = train_sample[1].cuda()

            data_dict['src_token_ids'] = data['src_token_ids'].cuda()
            data_dict['src_mask'] = data['src_mask'].cuda()
            data_dict['tgt_token_ids'] = data['tgt_token_ids'].cuda()
            data_dict['tgt_mask'] = data['tgt_mask'].cuda()

            optimizer.zero_grad()
            with autocast():
                output = model(data_dict)
                loss = loss_func(output, labels)

            train_correct += (output.argmax(1) == labels).sum().item()
            train_error += (output.argmax(1) != labels).sum().item()

            # 使用scaler进行反向传播和优化器更新
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            scaler.step(optimizer)
            scaler.update()

            # 更新tqdm中的显示信息
            train_bar.set_postfix({'Acc': train_correct / (train_error + train_correct), 'loss': loss.item()})
            train_bar.update()

            if (i + 1) % 300 == 0:
                tqdm.write('epoch : {} batch : {},Train Acc: {},  loss : {}'.format(epoch + 1, i, train_correct / (
                        train_error + train_correct), loss.item()))


        scheduler.step()
        lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        tqdm.write('lr: {}'.format(lr))

        if epoch > 5:
            model.eval()
            valid_bar = tqdm(dataloaders['validation'])
            dev_predictions = []
            dev_targets = []

            for valid_sample in valid_bar:
                with torch.no_grad():
                    data_dict = {}
                    data = valid_sample[0]
                    labels = valid_sample[1].cuda()

                    data_dict['src_token_ids'] = data['src_token_ids'].cuda()
                    data_dict['src_mask'] = data['src_mask'].cuda()
                    data_dict['tgt_token_ids'] = data['tgt_token_ids'].cuda()
                    data_dict['tgt_mask'] = data['tgt_mask'].cuda()

                    output = model(data_dict)
                    dev_predictions.extend(output.argmax(1).cpu().numpy())
                    dev_targets.extend(labels.cpu().numpy())

            dev_acc = accuracy_score(dev_targets, dev_predictions)
            dev_f1 = f1_score(dev_targets, dev_predictions)
            tqdm.write("Epoch: {}, Validation Acc: {} Validation F1: {}".format(epoch + 1, dev_acc, dev_f1))
            logging.info("Epoch: {}, Validation Acc: {}, Validation F1: {}".format(epoch + 1, dev_acc, dev_f1))
            model_path = model_out / "longformer_CNSE_enhanced_epoch_{0}-valid_acc{1}".format(epoch + 1, dev_acc)

            model.eval()
            test_bar = tqdm(dataloaders['validation'])
            test_predictions = []
            test_targets = []

            for test_sample in test_bar:
                with torch.no_grad():
                    data_dict = {}
                    data = test_sample[0]
                    labels = test_sample[1].cuda()

                    data_dict['src_token_ids'] = data['src_token_ids'].cuda()
                    data_dict['src_mask'] = data['src_mask'].cuda()
                    data_dict['tgt_token_ids'] = data['tgt_token_ids'].cuda()
                    data_dict['tgt_mask'] = data['tgt_mask'].cuda()

                    output = model(data_dict)
                    test_predictions.extend(output.argmax(1).cpu().numpy())
                    test_targets.extend(labels.cpu().numpy())

            test_acc = accuracy_score(test_targets, test_predictions)
            test_f1 = f1_score(test_targets, test_predictions)
            tqdm.write("Epoch: {}, Test Acc: {} Test F1: {}".format(epoch + 1, test_acc, test_f1))
            logging.info("Epoch: {}, Test Acc: {}, Test F1: {}".format(epoch + 1, test_acc, test_f1))

            if dev_acc > best_valid_acc:
                best_valid_acc = dev_acc
                model_state_dict = {
                    'embedding': model.embedding.state_dict(),
                    'fc': model.fc.state_dict(),
                    'prediction': model.prediction.state_dict()
                }

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, model_path)


if __name__ == '__main__':
    main()

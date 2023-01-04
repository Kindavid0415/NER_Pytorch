import os
import logging
import torch
import json
from torch.utils.data import Dataset, DataLoader


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class NERDataSet(Dataset):
    def __init__(self, path, is_overwrite=False):
        super(NERDataSet, self).__init__()
        file_dir = os.path.dirname(path)
        file = os.path.basename(path)
        file_name, _ = os.path.splitext(file)
        cached_path = os.path.join(file_dir, "cached_{}".format(file_name)).replace('\\', '/')
        if os.path.exists(cached_path) and not is_overwrite:
            logger.info("发现缓存文件{}，直接加载".format(cached_path))
            self.data_set = torch.load(cached_path)
        else:
            self.data_set = self.load_data(path=path)
            logger.info("数据预处理完成，缓存文件写入{}".format(cached_path))
            torch.save(self.data_set, cached_path)

    @staticmethod
    def load_data(path):
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        # 1.加载数据
        instance = self.data_set[idx]
        text = [102] + instance['text'] + [103]  # [STA] + text + [END]
        label = [0] + instance['label'] + [0]
        return text, label


def ner_collate_fn(batch):
    max_len = max(len(text) for text, label in batch)
    texts, labels, masks = [], [], []
    for text, label in batch:
      text_len = len(text)
      if text_len < max_len:
        pad_len = max_len - text_len
        text += [1] * pad_len  # [PAD]
        label += [0] * pad_len
        mask = [1] * text_len + [0] * pad_len
        texts.append(text)
        labels.append(label)
        masks.append(mask)
      else:
        mask = [1] * text_len
        texts.append(text)
        labels.append(label)
        masks.append(mask)
    return torch.LongTensor(texts), torch.LongTensor(labels), torch.tensor(masks, dtype=torch.uint8)


if __name__ == '__main__':
  train_dataset = NERDataSet(path='./data/train.json', is_overwrite=True)
  val_dataset = NERDataSet(path='./data/val.json', is_overwrite=False)
  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=ner_collate_fn)
  for batch in train_loader:
      texts, labels, masks = batch

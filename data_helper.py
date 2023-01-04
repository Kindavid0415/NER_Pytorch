import json
import random


def get_data(tags, vocab_path, load_path, save_path):
    # 1. prepare
    tag2index = {'O': 0}
    for tag in tags:
        tag2index['B-'+str.upper(tag)] = len(tag2index)
        tag2index['I-'+str.upper(tag)] = len(tag2index)
    word2index = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            word = line.strip()
            word2index[word] = len(word2index) + 1
    # 2. convert
    data = []
    with open(load_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for line in content:
            text = line['text']
            labels = line['label']
            # 2.1. label to BIO
            label_BIO = ['O'] * len(text)
            for label in labels:
                start = label['start']
                end = label['end']
                length = end - start
                tag = label['labels'][0]  # single tag
                if tag in tags:
                    label_BIO[start:end] = ['B-'+str.upper(tag)] + ['I-'+str.upper(tag)] * (length - 1)
            # 2.2. label to index
            text_index = []
            for word in text:
                text_index.append(word2index.get(word, word2index['[UNK]']))
            label_index = []
            for tag in label_BIO:
                label_index.append(tag2index.get(tag, tag2index['O']))
            data.append({'text': text_index, 'label': label_index})
    # 3. save 
    f = open(save_path, 'w', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.close()


def split_data(data_path, train_path, val_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        random.seed(1)
        random.shuffle(data)
    data_len = len(data)
    f = open(train_path, 'w', encoding='utf-8')
    json.dump(data[:int(data_len*0.9)], f, ensure_ascii=False, indent=4)
    f.close()
    f = open(val_path, 'w', encoding='utf-8')
    json.dump(data[int(data_len*0.9):int(data_len*0.1)], f, ensure_ascii=False, indent=4)
    f.close()


if __name__ == '__main__':
    tags = ['stem', 'option', 'answer', 'class']
    get_data(tags=tags, vocab_path='./base/vocab.txt', 
        load_path='./data/project-1-at-2022-12-08-16-32-a5a705d0.json', save_path='./data/data.json')
    split_data(data_path='./data/data.json', train_path='./data/train.json', val_path='./data/val.json')

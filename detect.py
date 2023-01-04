import torch


if __name__ == '__main__':
  # 1. text
  text = ' 计算机能直接识别的语言是（）。 A.高级程序语言 B.?汇编语言 C.机器语言（或称指令系统）D.语言?正确答案： C 关联评价点的名称：计算机组成原理 '
  # 2. model
  model = torch.load('saved.pkl')
  # 3. detect prepare
  tags = ['stem', 'option', 'answer', 'class']
  tag2index = {'O': 0}
  for tag in tags:
      tag2index['B-'+str.upper(tag)] = len(tag2index)
      tag2index['I-'+str.upper(tag)] = len(tag2index)
  index2tag = dict(zip(tag2index.values(), tag2index.keys()))
  word2index = {}
  with open('./base/vocab.txt', 'r', encoding='utf-8') as f:
      while True:
          line = f.readline()
          if line == '':
              break
          word = line.strip()
          word2index[word] = len(word2index) + 1
  text_index = []
  for word in text:
      text_index.append(word2index.get(word, word2index['[UNK]']))
  # 4. detect
  model.eval()
  with torch.no_grad():
    pred = model(texts=torch.LongTensor([text_index]))
  # 5. deal
  pred_BIO = [index2tag.get(k, 'NAN') for k in pred[0]]
  pass
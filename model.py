import torch
import torch.nn as nn
# pip install pytorch-crf
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
  def __init__(self, args):
    super(BiLSTM_CRF, self).__init__()
    # 1. receive parameters
    tags, vocab_path = args.tags, args.vocab_path
    embedding_dim, hidden_dim = args.embedding_dim, args.hidden_dim
    # 2. load file
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
    # 3. define
    self.tag_len = len(tag2index)
    self.vocab_len = len(word2index)
    self.hidden_dim = hidden_dim
    pad_index = word2index['[PAD]']
    # 4. embedding text to vector
    self.embedding = nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=embedding_dim, padding_idx=pad_index)
    # 5. LSTM
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim//2, num_layers=1, bidirectional=True)
    self.linear = nn.Linear(in_features=self.hidden_dim, out_features=self.tag_len)
    # 6. CRF
    self.crf = CRF(num_tags=self.tag_len)   #默认batch_first=False
    
  def forward(self, texts, tags=None, mask=None):
      batch_size = texts.shape[0]
      embeds = self.embedding(texts).permute(1,0,2)

      self.hidden = (torch.randn(2, batch_size, self.hidden_dim//2), torch.randn(2, batch_size, self.hidden_dim//2))
      lstm_out, self.hidden = self.lstm(embeds, self.hidden) 
      
      #3. 从BiLSTM层到全连接层
      lstm_feats = self.linear(lstm_out)   
      
      #4. 全连接层到CRF层
      if tags is not None:
          if mask is not None:
              loss = -1.*self.crf(emissions=lstm_feats,tags=tags.permute(1,0),mask=mask.permute(1,0),reduction='mean')
          else:
              loss = -1.*self.crf(emissions=lstm_feats,tags=tags.permute(1,0),reduction='mean')
          return loss
      else:   #测试用
          if mask is not None:
              prediction = self.crf.decode(emissions=lstm_feats,mask=mask.permute(1,0))
          else:
              prediction = self.crf.decode(emissions=lstm_feats)
          return prediction


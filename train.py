import torch.optim as optim
import argparse
import torch
from data_set import NERDataSet, ner_collate_fn
from model import BiLSTM_CRF
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tags', default=['stem', 'option', 'answer', 'class'], type=str, nargs='+', help='')
    parser.add_argument('--vocab_path', default='./base/vocab.txt', type=str, help='')
    parser.add_argument('--train_path', default='./data/train.json', type=str, help='')
    parser.add_argument('--val_path', default='./data/val.json', type=str, help='')
    parser.add_argument('--device', default='cpu', type=str, help='')
    parser.add_argument('--batch_size', default=8, type=int, help='')
    parser.add_argument('--lr', default=0.01, type=float, help='')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='')
    parser.add_argument('--embedding_dim', default=120, type=int, help='')
    parser.add_argument('--hidden_dim', default=12, type=int, help='')
    return parser.parse_args()


if __name__ == '__main__':
  # 1. receive parameters
  args = set_args()
  # 2. dataset
  train_dataset = NERDataSet(path=args.train_path, is_overwrite=True)
  train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ner_collate_fn)
  # 3. model
  model = BiLSTM_CRF(args=args)
  optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  # 4. train
  loss_list = []
  for epoch in range(50):
    for batch_data in tqdm(train_loader):
      texts, labels, masks = batch_data
        # 1. 清空梯度
      model.zero_grad()
      
      # 2. 运行模型
      loss = model(texts=texts, tags=labels, mask=masks) 
      
      # 3. 计算loss值，梯度并更新权重参数                                 
      loss.backward()    #retain_graph=True)  #反向传播，计算当前梯度
      optimizer.step()  #根据梯度更新网络参数
    loss_list.append(loss)
    print('Epoch=%d loss=%.5f' % (epoch, loss))
  # 5. save model
  torch.save(model, 'saved.pkl')

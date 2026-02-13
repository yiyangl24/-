import os.path
import pickle
import argparse
import warnings

import numpy as np
import torch.nn

from uitls import *
from data_utils import *
from model import SASRec


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataname', default='Electronics', type=str)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_len', default=50, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--llm_units', default=3072, type=int)
    parser.add_argument('--l2_emb', default=0.0, type=float)

    parser.add_argument('--llm_init', action='store_true')
    parser.add_argument('--gated', action='store_true')
    parser.add_argument('--tau', default=1, type=float)
    parser.add_argument('--beta', default=0.005, type=float)

    args = parser.parse_args()

    return args

def train(args):

    max_len = args.max_len
    dataname = args.dataname
    batch_size = args.batch_size

    with open('data/processed/' + dataname + '.pkl', 'rb') as f:
        data, _, _, user_dict, item_dict, item_cnt = pickle.load(f)

    dataset = data_partition(data)

    user_train, user_val, user_test, user_cnt, item_cnt, eval_set = dataset

    sampler = WarpSampler(user_train, user_cnt, item_cnt, batch_size, max_len, n_workers=3)

    model = SASRec(user_cnt, item_cnt, args).to(args.device)

    model.train()
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg = 0.0
    T = 0
    patience = 10

    save_dir = f'trained/{dataname}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir += f'/contrast.pth'

    batch_cnt = (len(user_train) - 1) // batch_size + 1

    for epoch in range(1, args.num_epochs + 1):
        for step in range(batch_cnt):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)

            optimizer.zero_grad()

            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)

            if args.gated:
                loss += args.beta * model.contrastive_loss(u, seq, pos, neg)

            loss.backward()
            optimizer.step()

            print(f"loss in epoch {epoch} iteration {step}: {loss.item():.4f}")

        if epoch % 20 == 0:
            model.eval()
            t_valid = evaluate_valid(model, dataset, args)
            print('Valid (NDCG@10: %.4f, HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f)' % (t_valid[0], t_valid[1], t_valid[2], t_valid[3]))
            if t_valid[0] > best_val_ndcg:  # NDCG@10
                T = 0
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                torch.save(model.state_dict(), save_dir)
            else:
                T += 1

            model.train()

            if T == patience:  # Early Stop
                break

            print(f'Early stop = {T}')

    model.load_state_dict(torch.load(save_dir, map_location=torch.device(args.device)))
    model.eval()
    ndcg, hr, ndcg_20, hr_20 = evaluate(model, dataset, args)
    print('Test (NDCG@10: %.4f, HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f)' % (ndcg, hr, ndcg_20, hr_20))
    sampler.close()
    print("Done")




if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    args = get_args()

    fix_random_seed(args.seed)

    train(args)





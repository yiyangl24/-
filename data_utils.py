import copy
import sys

import numpy as np

from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue

def data_partition(data):
    '''  Split the interaction into train, valid and test  '''
    user_cnt = 0
    item_cnt = 0

    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)

    for uid, inter in tqdm(data.items()):
        if len(inter) < 3:
            continue

        user_cnt = max(user_cnt, uid)
        item_cnt = max(item_cnt, max(inter))

        user_train[uid] = inter[:-2]
        user_valid[uid].append(inter[-2])
        user_test[uid].append(inter[-1])

    eval_set = [set(user_valid.keys()), set(user_test.keys())]

    avg_seq_len = sum(len(user_train[u]) for u in user_train) / len(user_train)

    print(f"user_cnt: {user_cnt}, item_cnt: {item_cnt}, avg_seq_len: {avg_seq_len: .2f}")

    return [user_train, user_valid, user_test, user_cnt, item_cnt, eval_set]


def sample_function(user_train, user_cnt, item_cnt, batch_size, max_len, result_queue):

    def random_neg(l, r, s):
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    def sample(uid):

        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, user_cnt + 1)

        seq = np.zeros([max_len], dtype=np.int32)
        pos = np.zeros([max_len], dtype=np.int32)
        neg = np.zeros([max_len], dtype=np.int32)

        cur = set(user_train[uid])

        nxt = user_train[uid][-1]
        idx = max_len - 1

        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neg(1, item_cnt + 1, cur)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    uids = np.arange(1, user_cnt + 1, dtype=np.int32)

    cnt = 0

    while True:
        if cnt % user_cnt == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[cnt % user_cnt]))
            cnt += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    '''  generate batch samples  '''
    def __init__(self, User, user_cnt, item_cnt, batch_size, max_len, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(Process(target=sample_function, args=(User, user_cnt, item_cnt, batch_size, max_len, self.result_queue)))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def evaluate_valid(model, dataset, args):

    train, valid, test, user_cnt, item_cnt, eval_set = copy.deepcopy(dataset)

    NDCG = 0.0
    HR = 0.0
    NDCG_20 = 0.0
    HR_20 = 0.0
    valid_user = 0.0

    uid = list(eval_set[0])

    for u in uid:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue
        seq = np.zeros([args.max_len], dtype=np.int32)
        idx = args.max_len - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        history = set(train[u] + valid[u])
        history.add(0)
        candidate = [valid[u][0]]
        for _ in range(99):
            t = np.random.randint(1, item_cnt + 1)
            while t in history:
                t = np.random.randint(1, item_cnt + 1)
            candidate.append(t)
        logits = - model.predict(*[np.array(l) for l in [[u], [seq], candidate]])
        logits = logits[0]
        rank = logits.argsort().argsort()[0].item()
        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HR += 1
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HR_20 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HR / valid_user, NDCG_20 / valid_user, HR_20 / valid_user


def evaluate(model, dataset, args):

    train, valid, test, user_cnt, item_cnt, eval_set = copy.deepcopy(dataset)

    NDCG = 0.0
    HR = 0.0
    NDCG_20 = 0.0
    HR_20 = 0.0
    valid_user = 0.0

    uid = list(eval_set[1])

    for u in uid:
        if len(train[u]) < 1 or len(valid[u]) < 1 or len(test[u]) < 1:
            continue
        seq = np.zeros([args.max_len], dtype=np.int32)
        idx = args.max_len - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        history = set(train[u] + valid[u] + test[u])
        history.add(0)
        candidate = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, item_cnt + 1)
            while t in history:
                t = np.random.randint(1, item_cnt + 1)
            candidate.append(t)
        logits = - model.predict(*[np.array(l) for l in [[u], [seq], candidate]])
        logits = logits[0]
        rank = logits.argsort().argsort()[0].item()
        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HR += 1
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HR_20 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HR / valid_user, NDCG_20 / valid_user, HR_20 / valid_user


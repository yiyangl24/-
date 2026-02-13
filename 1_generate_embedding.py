import copy

import torch
import pickle
import numpy as np


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer


class MyDataset(Dataset):

    def __init__(self, item_prompt):
        self.items = list(item_prompt.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def get_token_idx(llm_tokens, token_id):
    token_idx = []
    for i in range(len(llm_tokens['input_ids'])):
        idx_tensor = (llm_tokens['input_ids'][i] == token_id).nonzero().view(-1)
        token_idx.append(idx_tensor)
    return token_idx



if __name__ == '__main__':

    device = torch.device('cuda:1')

    llm_path = '../Llama-3.2-3B-Instruct'

    llm = AutoModelForCausalLM.from_pretrained(llm_path, device_map=device, torch_dtype=torch.float16, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, )

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'bos_token': '</s>'})
    tokenizer.add_special_tokens({'eos_token': '</s>'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ItemOut]']})

    llm.resize_token_embeddings(len(tokenizer))

    token_id = tokenizer('[ItemOut]', return_tensors='pt', add_special_tokens=False).input_ids.item()

    with torch.no_grad():
        token_embedding = llm.get_input_embeddings().weight
        token_embedding[token_id] = token_embedding.mean(dim=0)

    llm = prepare_model_for_kbit_training(llm)

    for name, param in llm.named_parameters():
        param.requires_grad = False

    with open('./data/processed/Electronics_meta.pkl', 'rb') as f:
        item_prompt = pickle.load(f)

    dataset = MyDataset(item_prompt)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: list(zip(*batch))
    )

    item_embedding = {}

    instruct_template = "\nBased on the above information, generate item representation token: [ItemOut]"

    max_input_length = 4096

    for asins, prompts in tqdm(dataloader):
        prompts = list(prompts)
        prompts = [p + copy.deepcopy(instruct_template) for p in prompts]
        asins = list(asins)
        inputs = tokenizer(prompts, return_tensors='pt', padding='longest', truncation=True, max_length=max_input_length).to(device)
        embeddings = llm.get_input_embeddings()(inputs['input_ids'])
        outputs = llm.forward(inputs_embeds=embeddings, output_hidden_states=True)
        idx = get_token_idx(inputs, token_id)

        for i, asin in zip(range(len(idx)), asins):
            i_embedding = outputs.hidden_states[-1][i, idx[i]].mean(axis=0)
            item_embedding[asin] = i_embedding.detach().cpu().tolist()

    with open('./data/processed/Electronics.pkl', 'rb') as f:
        _, _, _, _, item_dict, item_cnt = pickle.load(f)

    item_dict = item_dict['id2str']

    embedding_dim = len(list(item_embedding.values())[0])

    embedding_list = []
    for iid in range(1, item_cnt + 1):
        asin = item_dict[iid]
        if asin in item_embedding.keys():
            embedding_list.append(item_embedding[asin])
        else:
            embedding_list.append([0] * embedding_dim)

    embedding_list = np.array(embedding_list)

    with open('./data/processed/Electronics_embedding.pkl', 'wb') as f:
        pickle.dump(embedding_list, f, pickle.HIGHEST_PROTOCOL)




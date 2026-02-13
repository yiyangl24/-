import copy
import os.path

import torch
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans

from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_token_idx(llm_tokens, token_id):
    token_idx = []
    for i in range(len(llm_tokens['input_ids'])):
        idx_tensor = (llm_tokens['input_ids'][i] == token_id).nonzero().view(-1)
        token_idx.append(idx_tensor)
    return token_idx


import time


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    # params
    dataname = 'Electronics'
    n_clusters = 10
    device = torch.device(f'cuda:1')
    llm_path = '../Llama-3.2-3B-Instruct'
    max_input_length = 1024
    batch_size = 30

    # clusering
    with open('./data/processed/' + dataname + '_embedding.pkl', 'rb') as f:
        llm_embedding = pickle.load(f)

    model = KMeans(n_clusters=n_clusters)
    model.fit(llm_embedding)

    y_pred = model.predict(llm_embedding)

    cluster_map = dict(zip(range(1, len(y_pred) + 1), y_pred))

    with open('./data/processed/' + dataname + '.pkl', 'rb') as f:
        data, meta_data, title_list, user_dict, item_dict, item_cnt = pickle.load(f)

    assert max(cluster_map.keys()) == item_cnt

    # split into session
    data_session = {}
    for uid, inter in data.items():
        user_session = [[] for _ in range(n_clusters)]
        for iid in inter[:-1]:  # data leakage
            user_session[cluster_map[iid]].append(iid)
        data_session[uid] = user_session

    # generate profile
    llm = AutoModelForCausalLM.from_pretrained(llm_path, device_map=device, torch_dtype=torch.float16, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='left')

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'bos_token': '</s>'})
    tokenizer.add_special_tokens({'eos_token': '</s>'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[UserProfile]']})

    llm.resize_token_embeddings(len(tokenizer))

    token_id = tokenizer('[UserProfile]', return_tensors='pt', add_special_tokens=False).input_ids.item()

    with torch.no_grad():
        token_embedding = llm.get_input_embeddings().weight
        token_embedding[token_id] = token_embedding.mean(dim=0)

    llm = prepare_model_for_kbit_training(llm)

    for name, param in llm.named_parameters():
        param.requires_grad = False

    prompt_template = {
        'local': "Assume you are a consumer who is shopping online. You have shown interests in following commdities:\n {}. The commodities are segmented by '\n'. Please conclude it not beyond 50 words. Do not only evaluate one specfic commodity but illustrate the interests overall.",
        'global': "Assume you are an consumer and there are preference demonstrations from several aspects are as follows:\n {}. Please illustrate your preference with less than 100 words."
    }


    all_session_prompt = []
    all_session_uid = []


    llm.eval()

    for uid, user_session in tqdm(data_session.items()):
        session_profile = []
        for session in user_session:
            if len(session) > 0:
                r = " \n".join(title_list[iid - 1] for iid in session)
                prompt = prompt_template['local']
                prompt = prompt.format(r)
                all_session_prompt.append(prompt)
                all_session_uid.append(uid)

    user_session_summary = defaultdict(list)


    session_file = './data/processed/' + dataname + '_session.pkl'
    if os.path.exists(session_file):
        with open(session_file, 'rb') as f:
            user_session_summary = pickle.load(f)

    prompt, uid = [], []


    for i in tqdm(range(0, len(all_session_prompt))):

        if all_session_uid[i] in user_session_summary.keys():
            continue

        prompt.append(all_session_prompt[i])
        uid.append(all_session_uid[i])

        if len(uid) == batch_size or i == len(all_session_prompt) - 1:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length).to(device)
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            for j in range(len(uid)):
                input_len = inputs['attention_mask'][j].sum().item()
                summary = outputs[j, input_len:]
                response = tokenizer.decode(summary, skip_special_tokens=True).strip()
                user_session_summary[uid[j]].append(response)
            prompt, uid = [], []

            del inputs, outputs
            torch.cuda.empty_cache()
            if i % 500 == 0 or i == len(all_session_prompt) - 1:
                with open(session_file, 'wb') as f:
                    pickle.dump(user_session_summary, f, pickle.HIGHEST_PROTOCOL)

    del all_session_prompt, all_session_uid

    all_user_prompt = []
    all_user_uid = []
    for uid, session_list in user_session_summary.items():
        r = " \n".join(session_list)
        prompt = prompt_template['global']
        prompt = prompt.format(r)
        all_user_prompt.append(prompt)
        all_user_uid.append(uid)

    del user_session_summary

    profile_file = './data/processed/' + dataname + '_profile.pkl'

    user_profile = {}

    if os.path.exists(profile_file):
        with open(profile_file, 'rb') as f:
            user_profile = pickle.load(f)

    prompt, uid = [], []

    for i in tqdm(range(0, len(all_user_prompt))):

        if all_user_uid[i] in user_profile.keys():
            continue

        prompt.append(all_user_prompt[i])
        uid.append(all_user_uid[i])

        if len(uid) == batch_size or i == len(all_user_prompt) - 1:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length).to(device)
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            for j in range(len(uid)):
                input_len = inputs['attention_mask'][j].sum().item()
                summary = outputs[j, input_len:]
                response = tokenizer.decode(summary, skip_special_tokens=True).strip()
                user_profile[uid[j]] = response

            prompt, uid = [], []

            del inputs, outputs
            torch.cuda.empty_cache()

            if i % 500 == 0 or i == len(all_user_prompt) - 1:
                with open(profile_file, 'wb') as f:
                    pickle.dump(user_profile, f, pickle.HIGHEST_PROTOCOL)


    user_profile_embedding = {}

    profile_embedding_file = './data/processed/' + dataname + '_profile_embedding.pkl'
    if os.path.exists(profile_embedding_file):
        with open(profile_embedding_file, 'rb') as f:
            user_profile_embedding = pickle.load(f)

    prompt, uid = [], []

    instruct_template = "The user profile of a consumer is summarized as follows, \n{}.\n Based on the above information, generate user representation token [UserProfile]"

    batch_size = 30

    for i, (user_id, profile) in tqdm(enumerate(user_profile.items())):
        if user_id in user_profile_embedding:
            continue
        prompt.append(instruct_template.format(profile))
        uid.append(user_id)

        if len(uid) == batch_size or i == len(user_profile) - 1:
            inputs = tokenizer(prompt, return_tensors='pt', padding='longest', truncation=True, max_length=max_input_length).to(device)
            embeddings = llm.get_input_embeddings()(inputs['input_ids'])
            with torch.no_grad():
                outputs = llm.forward(inputs_embeds=embeddings, output_hidden_states=True)
            idx = get_token_idx(inputs, token_id)

            for j in range(len(uid)):
                user_embedding = outputs.hidden_states[-1][j, idx[j]].mean(axis=0)
                user_profile_embedding[uid[j]] = user_embedding.detach().cpu().tolist()

            del inputs, outputs
            torch.cuda.empty_cache()

            uid, prompt = [], []

            if i % 500 == 0 or i == len(user_profile) - 1:
                with open(profile_embedding_file, 'wb') as f:
                    pickle.dump(user_profile_embedding, f, pickle.HIGHEST_PROTOCOL)

    embedding_list = []

    user_cnt = max(user_profile_embedding.keys())

    for uid in range(1, user_cnt + 1):
        embedding_list.append(user_profile_embedding[uid])

    with open('./data/processed/Electronics_profile_embedding.pkl', 'wb') as f:
        pickle.dump(embedding_list, f, pickle.HIGHEST_PROTOCOL)



















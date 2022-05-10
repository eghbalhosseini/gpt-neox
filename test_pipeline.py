import copy
import re
from datetime import datetime
from functools import partial

import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import deepspeed
import numpy as np
import transformers
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
import seaborn as sns
from glob import glob




config=GPT2Config.from_pretrained('gpt2')
model=GPT2Model(config=config)
state_dict=model.state_dict()

model_trained=GPT2Model.from_pretrained('gpt2')
state_dict_trained=model_trained.state_dict()

'/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_100m_v2/gpt2/checkpoints_0/global_step100/mp_rank_00_model_states.pt'
/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_100m_v2/gpt2/checkpoints_0/global_step200/

wpe1=torch.load('/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_100m_v2/gpt2/checkpoints_0/global_step100//layer_07-model_00-model_states.pt')
wpe2=torch.load('/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_100m_v2/gpt2/checkpoints_0/global_step200/layer_07-model_00-model_states.pt')
wpe_trained=torch.load('/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_1b_v2/gpt2/checkpoints_4/global_step152500/layer_00-model_00-model_states.pt')
a1=torch.load('/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_1m_v2/gpt2/checkpoints_0/global_step1/layer_02-model_00-model_states.pt')
a30=torch.load('/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_1m_v2/gpt2/checkpoints_0/global_step30/layer_02-model_00-model_states.pt')
a_trained=torch.load('/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_1b_v2/gpt2/checkpoints_4/global_step152500/layer_02-model_00-model_states.pt')
mp_rank=torch.load('/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_100m_v2/gpt2/checkpoints_0/global_step100/mp_rank_00_model_states.pt')

# scratch
layer_keys=['wte.weight', 'wpe.weight']

fig = plt.figure(figsize=(8, 8))
for idx,l_key in tqdm(enumerate(layer_keys)):
    ax = None
    ax = plt.subplot(4, 4, idx+1, frameon=True, sharex=ax)
    b = np.squeeze(np.reshape(state_dict[l_key].numpy(), (1, -1)))
    plt.hist(b, bins=50, alpha=1, histtype='step',density=True)
    b1 = np.squeeze(np.reshape(state_dict_trained[l_key].numpy(), (1, -1)))
    plt.hist(b1, bins=50,alpha=.5,density=True)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title(l_key)
plt.tight_layout()
fig.show()




wpe_keys=list(wpe1.keys())


fig = plt.figure(figsize=(8, 8))
for idx,l_key in tqdm(enumerate(wpe_keys)):
    ax = None
    ax = plt.subplot(4, 3, idx+1, frameon=True, sharex=ax)
    b = np.squeeze(np.reshape(wpe1[l_key].numpy(), (1, -1)))
    plt.hist(b, bins=50, alpha=1, histtype='step', density=False)
    b1 = np.squeeze(np.reshape(wpe2[l_key].numpy(), (1, -1)))
    plt.hist(b1, bins=50, alpha=.5, density=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title(l_key)
plt.tight_layout()
fig.show()

# compare weights
fig = plt.figure(figsize=(8, 8))
ax=None
ax = plt.subplot(4, 3, 1, frameon=True, sharex=ax)
b = np.squeeze(np.reshape(wpe_trained['word_embeddings.weight'].numpy(), (1, -1)))
plt.hist(b, bins=50, alpha=1, histtype='step', density=True)

b = np.squeeze(np.reshape(wpe1['word_embeddings.weight'].numpy(), (1, -1)))
plt.hist(b, bins=50, alpha=1, histtype='step', density=True,color='r')


b = np.squeeze(np.reshape(state_dict_trained['wte.weight'].numpy(), (1, -1)))
plt.hist(b, bins=50, alpha=.5, density=True)


plt.tight_layout()
fig.show()



a1_k=['input_layernorm.weight', 'input_layernorm.bias', 'attention.query_key_value.weight', 'attention.query_key_value.bias',
      'attention.dense.weight', 'attention.dense.bias', 'post_attention_layernorm.weight', 'post_attention_layernorm.bias',
      'mlp.dense_h_to_4h.weight', 'mlp.dense_h_to_4h.bias', 'mlp.dense_4h_to_h.weight', 'mlp.dense_4h_to_h.bias']

fig = plt.figure(figsize=(8, 8))
for idx,l_key in tqdm(enumerate(a1_k)):
    ax = None
    ax = plt.subplot(4, 3, idx+1, frameon=True, sharex=ax)
    b = np.squeeze(np.reshape(a1[l_key].numpy(), (1, -1)))
    plt.hist(b, bins=50, alpha=1, histtype='step', density=False)
    b1 = np.squeeze(np.reshape(a_trained[l_key].numpy(), (1, -1)))
    plt.hist(b1, bins=50, alpha=.5, density=False)
    #sns.histplot(b,bins=50)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title(l_key)
plt.tight_layout()
fig.show()


layer_keys=['h.1.ln_1.weight','h.1.ln_1.bias','h.1.attn.bias','h.1.attn.c_attn.weight',
'h.1.attn.c_attn.bias','h.1.attn.c_proj.weight','h.1.attn.c_proj.bias','h.1.ln_2.weight','h.1.ln_2.bias',
'h.1.mlp.c_fc.weight','h.1.mlp.c_fc.bias','h.1.mlp.c_proj.weight','h.1.mlp.c_proj.bias']

fig = plt.figure(figsize=(8, 8))
for idx,l_key in tqdm(enumerate(layer_keys)):
    ax = None
    ax = plt.subplot(4, 4, idx+1, frameon=True, sharex=ax)
    b = np.squeeze(np.reshape(state_dict[l_key].numpy(), (1, -1)))
    plt.hist(b, bins=50, alpha=1, histtype='step',density=False)
    b1 = np.squeeze(np.reshape(state_dict_trained[l_key].numpy(), (1, -1)))
    plt.hist(b1, bins=50,alpha=.5,density=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title(l_key)
plt.tight_layout()
fig.show()


b=np.reshape(a['position_embeddings.weight'].numpy(),(1,-1))
(H,bins)=np.histogram(b,bins=50,density=False)
plt.bar(bins[:-1],H,width=1)
plt.show()


b=np.reshape(state_dict['wte.weight'].numpy(),(1,-1))
(H,bins)=np.histogram(b,bins=50,density=False)
plt.bar(bins[:-1],H,width=1)
plt.show()


b=np.reshape(state_dict_trained['wte.weight'].numpy(),(1,-1))
(H,bins)=np.histogram(b,bins=50,density=False)
plt.bar(bins[:-1],H,width=1)
plt.show()

import re
import copy
## try reseting the weights so that you can run the model from this new initialztion
def permute_mat(mat):
    mat_flat = mat.flatten()
    assert(mat_flat.ndim==1)
    shuffle_idx = torch.randperm(mat_flat.shape[0])
    mat_flat_rnd = mat_flat[shuffle_idx]
    mat_perm = torch.reshape(mat_flat_rnd, mat.shape)
    return mat_perm

layer_files=glob('/om/user/ehoseini/MyData/miniBERTa_training/miniBERTa_100m_v2/gpt2/checkpoints_0/global_step100/*model_00-model_states.pt')

layer_id=[re.findall('layer_\d+',x) for x in layer_files]
layer_id=[int(x[0].replace('layer_','')) for x in layer_id]
reorder=np.argsort(layer_id)

layer_files_sort=[layer_files[x] for x in reorder]

all_keys=[]
for file in tqdm(layer_files_sort):
    True
    t=torch.load(file)
    t1=copy.deepcopy(t)
    for key in t1.keys():
        w=t1[key]
        b = torch.normal(0, .02, size=w.shape)
        t1[key] = permute_mat(b)
    torch.save(t1,file)
    all_keys.append(list(t.keys()))



for key in all_keys:
    print(key)


for key in mp_rank.keys():
    print(key)
    print(mp_rank[key])
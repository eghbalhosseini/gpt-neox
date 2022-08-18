import wandb
import pandas as pd
import numpy as np
import pickle
api = wandb.Api()
import os


def get_valid_run(user,project,run_id):
    valid_runs=[]
    for ids_ in valid_run_ids:
        valid_run = api.run(f"{user}/{project}/{ids_}")
        valid_runs.append(valid_run)
    return valid_runs

if __name__ == "__main__":
    user='eghbalhosseini'
    ''' this section extract data for miniberta 1b'''
    project='miniberta_1b_v2'
    valid_run_ids= ['18rqsvui','3p2at9b1','3u95asnv']

    valid_runs=get_valid_run(user,project,valid_run_ids)

    train_readouts=[]
    valida_readouts=[]
    for run in valid_runs:
        a_train=pd.DataFrame(run.history(samples=run.summary['_step'],keys=["train/lm_loss","train/learning_rate","timers/train_batch"]))
        a_test = pd.DataFrame(run.history(samples=run.summary['_step'], keys=['validation/lm_loss','validation/lm_loss_ppl']))
        train_readouts.append(a_train)
        valida_readouts.append(a_test)

    miniberta_1b=dict(train_result=np.asarray(pd.concat(train_readouts)),validation_result=np.asarray(pd.concat(valida_readouts)))

    ''' this section extract data for miniberta 100M'''
    project='miniberta_100M_v2'
    valid_run_ids=['22bmrcer',]
    valid_runs=get_valid_run(user,project,valid_run_ids)
    train_readouts=[]
    valida_readouts=[]
    for run in valid_runs:
        a_train=pd.DataFrame(run.history(samples=run.summary['_step'],keys=["train/lm_loss","train/learning_rate","timers/train_batch"]))
        a_test = pd.DataFrame(run.history(samples=run.summary['_step'], keys=['validation/lm_loss','validation/lm_loss_ppl']))
        train_readouts.append(a_train)
        valida_readouts.append(a_test)
    miniberta_100m=dict(train_result=np.asarray(pd.concat(train_readouts)),validation_result=np.asarray(pd.concat(valida_readouts)))


    ''' this section extract data for miniberta 10M'''
    project='miniberta_10m_v2'
    valid_run_ids=['2xj5rvil',]
    valid_runs=get_valid_run(user,project,valid_run_ids)
    train_readouts=[]
    valida_readouts=[]
    for run in valid_runs:
        a_train=pd.DataFrame(run.history(samples=run.summary['_step'],keys=["train/lm_loss","train/learning_rate","timers/train_batch"]))
        a_test = pd.DataFrame(run.history(samples=run.summary['_step'], keys=['validation/lm_loss','validation/lm_loss_ppl']))
        train_readouts.append(a_train)
        valida_readouts.append(a_test)
    miniberta_10m=dict(train_result=np.asarray(pd.concat(train_readouts)),validation_result=np.asarray(pd.concat(valida_readouts)))

    ''' this section extract data for miniberta 1M'''
    project = 'miniberta_1m_v2'
    valid_run_ids = ['39buccnt', ]
    valid_runs = get_valid_run(user, project, valid_run_ids)
    train_readouts = []
    valida_readouts = []
    for run in valid_runs:
        a_train = pd.DataFrame(run.history(samples=run.summary['_step'], keys=["train/lm_loss","train/learning_rate","timers/train_batch"]))
        a_test = pd.DataFrame(
            run.history(samples=run.summary['_step'], keys=['validation/lm_loss', 'validation/lm_loss_ppl']))
        train_readouts.append(a_train)
        valida_readouts.append(a_test)
    miniberta_1m = dict(train_result=np.asarray(pd.concat(train_readouts)),
                         validation_result=np.asarray(pd.concat(valida_readouts)))

    train_validation_set=dict(miniberta_1b=miniberta_1b,
                              miniberta_100m=miniberta_100m,
                              miniberta_10m=miniberta_10m,
                              minberta_1m=miniberta_1m)

    save_dir='/om/user/ehoseini/MyData/NeuroBioLang_2022/data/'
    with open(os.path.join(save_dir,'miniberta_train_valid_set.pkl'), 'wb') as handle:
        pickle.dump(train_validation_set, handle, protocol=4)
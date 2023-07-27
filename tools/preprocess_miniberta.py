from datasets import load_dataset
import os
from tqdm import tqdm
import jsonlines
miniBERTa_dir='/om2/user/ehoseini/MyData/miniBERTa_v2/'
#miniBERTa_dir='/Users/eghbalhosseini/Desktop/miniBERTa/'
miniBERTa_vers=['miniBERTa-50M',]
import subprocess
from pathlib import Path
import os
from datasets import load_dataset

#/om/user/ehoseini/MyData/miniBERTa/miniBERTa.py
import sys
if __name__ == '__main__':
    # first step is changing format to jsonl
    file_list_to_compress=[]
    # print HF_DATASETS_CACHE locaiton

    # check if HF_DATASETS_CACHE is set


    for idx, vers in tqdm(enumerate(miniBERTa_vers)):
        data=load_dataset(miniBERTa_dir,vers,cache_dir='/om/user/ehoseini/.cache/huggingface/datasets')
        for key in data.keys():
            if not os.path.exists(os.path.join(miniBERTa_dir,vers,vers+'_'+key+'.jsonl.zst')):
                if not os.path.exists(os.path.join(miniBERTa_dir,vers,vers+'_'+key+'.jsonl')):
                    with jsonlines.open(os.path.join(miniBERTa_dir,vers,vers+'_'+key+'.jsonl'),'w') as writer:
                        for idy, x in tqdm(enumerate(data[key])):
                            if len(x['text'])>0:
                                writer.write(x)
                        writer.close()
                # compress file
                unix_str = f"zstd -f {os.path.join(miniBERTa_dir, vers, vers + '_' + key + '.jsonl')}"
                cmd = f'''
                #. ~/.bash_profile
                conda activate gpt_neox
                {unix_str}
                '''
                subprocess.call(cmd, shell=True, executable='/bin/bash')

    file_list_to_preproc = []
    for idx, vers in tqdm(enumerate(miniBERTa_vers)):
        data=load_dataset(miniBERTa_dir,vers)
        for key in data.keys():
            file_list_to_preproc.append(os.path.join(miniBERTa_dir,vers,vers+'_'+key+'.jsonl.zst'))

    for idx, file in tqdm(enumerate(file_list_to_preproc)):
        p = Path(file)
        filename=p.name.replace('.jsonl.zst','')

        unix_str = f"python tools/preprocess_data.py \
            --input {file} \
            --output-prefix {str(p.parent)}/{filename} \
            --vocab data/gpt2-vocab.json \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --merge-file data/gpt2-merges.txt \
            --append-eod \
            --workers 4 \
            --ftfy "

        cmd = f'''
                        #. ~/.bash_profile
                        conda activate gpt_neox
                        {unix_str}
                        '''
        subprocess.call(cmd, shell=True, executable='/bin/bash')


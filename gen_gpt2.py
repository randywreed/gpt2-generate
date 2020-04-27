#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!pip install gpt-2-simple

# In[24]:
import argparse
parser=argparse.ArgumentParser(description="generate gpt-2 response")
parser.add_argument("--prompt",help="prompt to use for generate")
args=parser.parse_args()
prompt_text=args.prompt


# In[6]:


import gpt_2_simple as gpt2
import os
import requests
import logging
import subprocess
log=logging.getLogger("my-logger")
# In[7]:
gpucnt=str(subprocess.check_output(['nvidia-smi',"-L"])).count("UUID")
log.info("number of GPUs {}".format(gpucnt))

model_name = "1558M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


# In[9]:


#!gpt_2_simple generate --temperature 1.0 --nsamples 5 --batch_size 5 --length 500 

# In[13]:


sess=gpt2.start_tf_sess()

# In[17]:

if gpucnt>1:
    gpt2.load_gpt2(sess,model_name="1558M",multi_gpu=True)
else:
    gpt2.load_gpt2(sess,model_name="1558M")


# In[21]:


all_seq=gpt2.generate(sess,model_name="1558M",batch_size=2,nsamples=20,length=500,prefix=prompt_text,return_as_list=True)

# In[23]:

log.info('generation finished len={}'.format(len(output)))
#print(output)

# In[ ]:

log.info('writing to output file')
output_file="simple_results.csv"
log.info('output file={}'.format(output_file))
import csv
import os
if os.path.exists(output_file):
    append_flag="a"
else: 
    append_flag="w"
with open (output_file, append_flag) as csvfile:
    writer=csv.writer(csvfile)
    for i in all_seq:
        writer.writerow([prompt_text, i])
        print(i)
        log.info(i)
    

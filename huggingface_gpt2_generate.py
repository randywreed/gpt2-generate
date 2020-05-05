#!/usr/bin/env python
# coding: utf-8

# # Generate Gpt-2 Examples #
# 
# **prompt_text**:change the prompt to a text of your choice  
# **response_length**: This is the maximum length of any response. It may choose to create shorter response itself. This cannot be longer than 1024  
# **output_file**: This should be a .csv file. It can be downloaded an analyzed. If this is run multiple times, it will append all answers into the file.  
# **num_of_responses**: This is usually a max of 4 for K80 GPU. It is possible that even at 4, the process may error. If this is the case rerun.  
#   
# **IN CASE OF ERROR**  
# The process may error because the GPU gets filled up. Everytime you run this, make sure that the GPU starts with 0 memory used. The !Nvidia-smi command shows that.   
# In the middle column it should say "0MB/11441MB"    
# if it does not. The GPU needs to be cleared.  
# Clear the GPU by clicking on the top menu "Kernel->Restart Kernel" Then rerun the notebook from the top cell.  

# In[1]:
import argparse
parser=argparse.ArgumentParser("gpt2 generator using huggingface transformers")
parser.add_argument("--prompt")
parser.add_argument("--length",type=int,default=500, help="the length of selection to generate, default=500")
parser.add_argument("--num",type=int,default=20,help="number of selections to generate, default=20")
args=parser.parse_args()

prompt_text=args.prompt
#"If God is defined as something that is all powerful and all knowing, a strong artificial intelligence might be an actual God. If this happens the implications for religion are"
#max reponse_length 1024
response_length=args.length
#output_file will be created if it doesn't exist, otherwise answers will be appended
output_file="results.csv"
#max 4 (k80 gpu)
num_of_responses=args.num


# In[2]:


#from ipyexperiments import *


# In[3]:


#get_ipython().system('nvidia-smi')


# In[4]:


all_seq=[]


# In[5]:


#exp1=IPyExperimentsPytorch()


# In[6]:


import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

#device = xm.xla_device()
device=torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained('/spell/GPT2Model/')
model = GPT2LMHeadModel.from_pretrained('/spell/GPT2Model/')
model.to(device)
encoded_prompt=tokenizer.encode(prompt_text, add_special_tokens=True,return_tensors="pt")
encoded_prompt = encoded_prompt.to(device)

outputs = model.generate(encoded_prompt,response_length,temperature=.8,num_return_sequences=num_of_responses,repetition_penalty=85,do_sample=True,top_k=80,top_p=.85, )


# In[7]:


# Remove the batch dimension when returning multiple sequences
if len(outputs.shape) > 2:
    outputs.squeeze_()


# In[34]:


generated_sequences=[]
total_sequence=""
for generated_sequence_idx, generated_sequence in enumerate(outputs):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()
        #print(generated_sequence)
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        #print(text)
        # Remove all text after the stop token
        stop_token='<|endoftext|>'
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        print(total_sequence)
        all_seq.append(total_sequence)


# In[35]:


if len(all_seq)==len(set(all_seq)):
  print('no duplicates')
else:
  print('duplicates found')


# In[36]:


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
    


# In[37]:


print('run complete')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install huggingface_hub')
get_ipython().system('pip install transformers')
get_ipython().system('pip install langchain')
get_ipython().system('pip install chainlit')


# !chainlit hello

# Importing libraries and access tokens

# In[6]:


import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain


# In[9]:


from getpass import getpass
HUGGINGFACEHUB_API_TOKEN= getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN']= HUGGINGFACEHUB_API_TOKEN


# Setting Conversational Model

# In[19]:


model_id= "gpt2-medium"
conv_model= HuggingFaceHub(huggingfacehub_api_token= os.environ['HUGGINGFACEHUB_API_TOKEN'], 
                                                    repo_id= model_id,
                                                    model_kwargs={"temperature":0.8, "max_new_tokens":200})


# In[20]:


template= """You are a helpful AI Assistant that makes stories by completing the query provided by the user
{query}

"""
prompt= PromptTemplate(template= template, input_variables=['query'])


# In[21]:


conv_chain= LLMChain(llm=conv_model,
                    prompt=prompt,
                    verbose=True)


# In[22]:


print(conv_chain.run("Once upon a time"))


# 

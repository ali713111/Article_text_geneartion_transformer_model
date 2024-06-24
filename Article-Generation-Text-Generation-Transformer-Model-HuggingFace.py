#!/usr/bin/env python
# coding: utf-8

# In[13]:


# this is transformer model from hugging face library, GPT2 specifically, this model is basically trained on two different articles on Artificial intelligence which encompasses education and economy.
# install these libraries and tools 
get_ipython().system('pip install transformers torch pandas numpy nltk scikit-learn')


# In[14]:


# import these required libararies
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split


# In[15]:


# this nltk library is crucial for text preprocessing it will remove all unnecessary marks within the data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[16]:


# preprocess text data
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.lower()  # Convert to lowercase
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

def load_and_preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    cleaned_text = preprocess_text(raw_text)
    return cleaned_text

file_path = 'Role of AI in education.txt'
cleaned_text = load_and_preprocess_text(file_path)


# In[17]:


# did preparation of dataset for GPT-2 model

from transformers import TextDataset, DataCollatorForLanguageModeling

def save_cleaned_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

cleaned_file_path = 'cleaned_role_of_ai_in_education.txt'
save_cleaned_text_to_file(cleaned_text, cleaned_file_path)

def load_dataset(file_path, tokenizer, block_size=512):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def load_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
dataset = load_dataset(cleaned_file_path, tokenizer)
data_collator = load_data_collator(tokenizer)


# In[18]:


# training and intiliazation of model

model = GPT2LMHeadModel.from_pretrained('gpt2')

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()


# In[23]:


# generate any text on the based of trained dataset
def generate_text(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "The role of AI in economy"
generated_text = generate_text(prompt, model, tokenizer)
print(generated_text)


# In[26]:


def generate_text(prompt, model, tokenizer, max_length=200, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=2  # Prevents repeating n-grams of specified length
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "The role of education"
generated_text = generate_text(prompt, model, tokenizer, max_length=200)
print(generated_text)


# In[ ]:





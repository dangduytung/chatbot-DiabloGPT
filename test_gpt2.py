# Import the necessary libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gradio as gr
import datetime

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the device to run the model on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

prompt = 'hi'

input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# output_ids = model.generate(input_ids, max_length=100, do_sample=True, pad_token_id=50256)

# output_ids = model.generate(input_ids=input_ids, do_sample=True, pad_token_id=50256, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

# output_ids = model.generate(
#     input_ids=input_ids, 
#     max_length=100,
#     do_sample=True,
#     top_p=0.9,
#     top_k=50,
#     temperature=0.7,
#     no_repeat_ngram_size=2,
#     pad_token_id=tokenizer.eos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     early_stopping=True,
#     num_return_sequences=1,
#     decoder_start_token_id=model.config.decoder_start_token_id
# )

output_ids = model.generate(
    input_ids, 
    max_new_tokens=100, 
    pad_token_id=tokenizer.eos_token_id, 
    num_beams=5,
    no_repeat_ngram_size=2,
)

output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print('output:' + output)
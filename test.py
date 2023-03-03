import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datetime

model_name = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

MAX_LENGTH = 200

def get_tensor_ids(input_text):
    return tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

def print_f(session_id, text):
    print(f"{datetime.datetime.now()} | {session_id} | {text}")

def submit_chat(input, history, request: gr.Request):
    client_ip = 'UNKNOWN'
    if request:
        # print("Request headers dictionary:", request.headers)
        # print("IP address:", request.client.host)
        client_ip = request.client.host + ':' + str(request.client.port)

    print_f(client_ip, f" inp: {input}")

    # Encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = get_tensor_ids(input)
    
    # Get history ids
    if history:
        chat_history_ids = []
        for h in history:
            chat_history_ids.append(get_tensor_ids(h[0]))
            chat_history_ids.append(get_tensor_ids(h[1]))
            
        # append the new user input tokens to the chat history
        chat_history_ids.append(new_user_input_ids)

        bot_input_ids = torch.cat(chat_history_ids, dim=1)
    else:
        bot_input_ids = new_user_input_ids
    # print(f"bot_input_ids: {bot_input_ids}")

    # Can add some params like: do_sample=True, temperature=0.7 for output is different
    output_ids = model.generate(bot_input_ids, max_length=MAX_LENGTH, pad_token_id=tokenizer.eos_token_id)
    # print_f(client_ip, f" output_ids: {output_ids}")

    # Get response (skip_special_tokens=True is remove <|endoftext|>)
    output = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print_f(client_ip, f" out: {output}")

    # Append to history
    history.append((input, output))
    # print(f"history: {history}")
    
    return history, history

css = """
    #row_bot{width: 70%; height: var(--size-96); margin: 0 auto}
    #row_bot .block{background: var(--color-grey-100); height: 100%}
    #row_input{width: 70%; margin: 0 auto}
    #row_input .block{background: var(--color-grey-100)}

    @media screen and (max-width: 768px) {
        #row_bot{width: 100%; height: var(--size-96); margin: 0 auto}
        #row_bot .block{background: var(--color-grey-100); height: 100%}
        #row_input{width: 100%; margin: 0 auto}
        #row_input .block{background: var(--color-grey-100)}    
    }
    """
block = gr.Blocks(css=css, title="Funny Bot")

with block:
    gr.Markdown(
        """<p style="font-size:40px; text-align: center">&#128540;</p>""")
    with gr.Row(elem_id='row_bot'):
        chatbot = gr.Chatbot()
    with gr.Row(elem_id='row_input'):
        message = gr.Textbox(placeholder="Enter something")
        state = gr.State([])

        message.submit(submit_chat, inputs=[
                       message, state], outputs=[chatbot, state])
        message.submit(lambda x: "", message, message)

# # Params ex: debug=True, share=True, server_name="0.0.0.0", server_port=5050
block.launch(debug=True)
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datetime
import __init__

MODEL_NAME = __init__.MODEL_MICROSOFT_DIABLO_MEDIUM
OUTPUT_MAX_LENGTH = __init__.OUTPUT_MAX_LENGTH


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def print_f(session_id, text):
    print(f"{datetime.datetime.now()} | {session_id} | {text}")


def predict(input, history, request: gr.Request):
    session_id = 'UNKNOWN'
    if request:
        # Get session_id is client_ip + client_port
        session_id = request.client.host + ':' + str(request.client.port)
    # print_f(session_id, f" inp: {input}")

    # Tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(
        input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat(
        [torch.LongTensor(history), new_user_input_ids], dim=-1)

    # Generate a response
    history = model.generate(bot_input_ids, max_length=OUTPUT_MAX_LENGTH,
                             pad_token_id=tokenizer.eos_token_id).tolist()

    # Convert the tokens to text, and then split the responses into lines
    response = tokenizer.decode(history[0]).split("<|endoftext|>")

    # Convert to tuples of list
    response = [(response[i], response[i + 1])
                for i in range(0, len(response) - 1, 2)]

    # Print new conversation
    print_f(session_id, response[-1])

    return response, history


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
block = gr.Blocks(css=css, title="Chatbot")

with block:
    gr.Markdown(f"""
        <p style="font-size:20px; text-align: center">{MODEL_NAME}</p>
        """)
    with gr.Row(elem_id='row_bot'):
        chatbot = gr.Chatbot()
    with gr.Row(elem_id='row_input'):
        message = gr.Textbox(placeholder="Enter something")
        state = gr.State([])

        message.submit(predict,
                       inputs=[message, state],
                       outputs=[chatbot, state])
        message.submit(lambda x: "", message, message)

# Params ex: debug=True, share=True, server_name="0.0.0.0", server_port=5050
block.launch()

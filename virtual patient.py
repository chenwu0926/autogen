import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from peft import PeftModel


model_path = '/data/qwen_model/qwen/Qwen2___5-7B-Instruct'
lora_path = '/data/qwen_model/qwen/lora/checkpoint-183'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,  torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, model_id=lora_path)
model = model.to("cuda:0")  


system_message = {"role": "system", "content": "现在你要扮演需要问诊的患者"}

def predict(user_input, history):  
    history_transformer_format = history + [[user_input, ""]]
    messages = [system_message] + [{"role": "user", "content": item[0]} for item in history] + [{"role": "user", "content": user_input}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        if new_token != "<":  
            partial_message += new_token
            yield partial_message

gr.ChatInterface( fn=predict,title="问诊聊天模型",description="与模型进行问诊聊天。").launch(share=True)
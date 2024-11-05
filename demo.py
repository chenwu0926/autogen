import gradio as gr
from PIL import Image

from http import HTTPStatus
import dashscope
import PyPDF2
from utils import OCR_file, ASR, structurize_text



def process_input(text, audio_path, image_path, pdf_path, process_type=None):
    if text:
        if process_type == "structurized":
            text = structurize_text(text)
        return text
    elif audio_path:
        # print(audio_path)
        lang, result = ASR(audio_path)
        return result
    elif image_path:
        ocr_result = OCR_file(image_path)
        if process_type == "structurized":
            ocr_result = structurize_text('\n'.join(ocr_result))
        return ocr_result
    elif pdf_path:
        ocr_result = OCR_file(pdf_path)
        if process_type == "structurized":
            ocr_result = structurize_text('\n'.join(ocr_result))
        return ocr_result
    else:
        return "No input provided"

with gr.Blocks() as demo:
    gr.Markdown("# Multimodal Input Demo\nUpload text, audio, or an image to see different functionalities.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text Input")
            audio_input = gr.Audio(label="Audio Input", sources=["upload","microphone"], type='filepath')
            image_input = gr.Image(label="Image Input", type="filepath")
            pdf_input = gr.File(label="PDF Input", type="filepath", visible=False)
            type_input = gr.Radio(
                ["raw", "structurized"], label="Output Type"
            )
            process_btn = gr.Button("Process Input")

        with gr.Column():
            output = gr.Textbox(label="Processed Output")
            # add clear button
            clear_btn = gr.Button("Clear Output")
    
    process_btn.click(process_input, inputs=[text_input, audio_input, image_input, pdf_input, type_input], outputs=output)
    clear_btn.click(lambda : None, outputs=output)

demo.launch()

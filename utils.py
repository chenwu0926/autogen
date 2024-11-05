from paddleocr import PaddleOCR
import PyPDF2
import shutil
import os
import dashscope
from http import HTTPStatus
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
import deepgram
import httpx
from datetime import datetime
from deepgram.utils import verboselogs
import logging


# STEP 1 Create a Deepgram client using the API key in the environment variables
config: DeepgramClientOptions = DeepgramClientOptions(
    verbose=verboselogs.SPAM,
)
deepgram: DeepgramClient = DeepgramClient("3fd415f233a2674f89051cad68f83af839bc6a77", config)

import os
from pdf2image import convert_from_path
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 可以更改语言

def recognize_image(image_path):
    """对单个图像文件进行 OCR 识别"""
    result = ocr.ocr(image_path, cls=True)
    text = ""
    for line in result:
        text += "".join([w[1][0] for w in line]) + "\n"
    return text

def recognize_pdf(pdf_path):
    """将 PDF 转换为图像，逐页 OCR 识别并拼接文本"""
    images = convert_from_path(pdf_path)
    full_text = ""
    for page_num, image in enumerate(images):
        # 保存图像以便识别
        temp_image_path = f"temp_page_{page_num}.png"
        image.save(temp_image_path, "PNG")
        
        # 识别并拼接文本
        page_text = recognize_image(temp_image_path)
        full_text += f"第 {page_num + 1} 页:\n" + page_text + "\n"
        
        # 删除临时图像文件
        os.remove(temp_image_path)
    
    return full_text

def OCR_file(file_path):
    """主函数：识别文件内容，自动判断文件类型"""
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 如果是图像文件，直接 OCR 识别
        return recognize_image(file_path)
    elif file_path.lower().endswith('.pdf'):
        # 如果是 PDF 文件，逐页转换为图像后识别
        return recognize_pdf(file_path)
    else:
        raise ValueError("文件格式不支持。请提供图像或 PDF 文件。")


def ASR(audio_path : str , lang = 'zh'):
    try:
        # STEP 2 Call the transcribe_file method on the rest class
        with open(audio_path, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            utterances=True,
            punctuate=True,
            diarize=True,
            language=lang,
            detect_language=True if lang == "auto" else False,
        )

        before = datetime.now()
        response = deepgram.listen.rest.v("1").transcribe_file(
            payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
        )
        after = datetime.now()

        # print(response.to_json(indent=4))
        # print("")
        difference = after - before
        print(f"time: {difference.seconds}")

    except Exception as e:
        print(f"Exception: {e}")
        raise e
        
    result = response.to_dict()['results']

    with open('deep_gram.txt', 'w') as f:
        print(result, file=f)

    lang = result['channels'][0]['detected_language'] if lang == "auto" else lang
    asr_text_raw = result['channels'][0]['alternatives'][0]['transcript']

    segments = result['utterances']
    
    return lang, asr_text_raw
    
def get_pdf_page_count(pdf_file):
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
    return num_pages

def call_with_prompt(unstructured_text=''):
    dashscope.api_key = "sk-d9663052329b443bbe79bb6022efa6d0"
    prompt = '这是一份未经整理，较为散乱的医学检验报告，请将其整理成为结构化的报告。注意，报告可能为中英文对照，若为中英文对照，请将中英文的全部内容都进行整理。下面是原始报告的内容：\n' + unstructured_text
    print("Calling Dashscope with prompt.")
    response = dashscope.Generation.call(
        model=dashscope.Generation.Models.qwen_turbo,
        prompt=prompt,
    )
    if response.status_code == HTTPStatus.OK:
        logging.info("Dashscope response received.")
        return response['output']['text']
    else:
        logging.error(f"Dashscope error code: {response.status_code}")
        logging.error(f"Dashscope error message: {response.message}")
        return None

def structurize_text(unstructured_text: str):
    struc_texts = call_with_prompt(unstructured_text)
    return struc_texts

if __name__ == "__main__":
    page_num = get_pdf_page_count('test_files/mri-1.pdf')
    texts = OCR_file('test_files/mri-1.pdf')
    
    print("original text:", texts)
    struc_texts = call_with_prompt("\n".join(texts))
    
    # lang, test = ASR("test_files/test_audio.m4a")
    # print("Detected Language:", lang)
    # print("ASR Result:", test)
    print(get_pdf_page_count("test_files/mri-1.pdf"))
    # print(text)
    print(struc_texts)
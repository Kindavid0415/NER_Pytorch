import os
import re
import shutil
import logging
from pdf2docx import Converter
from docx import Document
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s")


if __name__ == '__main__':
    pdf_path = r'信息运维检修工-2021年11月30日机考题库.pdf'
    tmp_dir = r'./tmp'
    # 1.pdf转docx
    docx_path = pdf_path.replace('.pdf', '.docx')
    if os.path.exists(docx_path):
        logging.info('docx已存在')
    else:
        logging.info('生成docx文件')
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)
        cv.close()
    # 2.docx转txt
    txt_path = pdf_path.replace('.pdf', '.txt')
    document = Document(docx_path)
    paragraphs_text = ''
    for paragraph in document.paragraphs:
        paragraphs_text += paragraph.text
    logging.info('生成txt文件')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(paragraphs_text)
    # 3.剔除多余数据
    tmp_text = paragraphs_text
    result = re.search('当前工种.*?\n', tmp_text).group()
    print(result)
    tmp_text = tmp_text.replace(result, '\n')
    result = re.search('工种定义.*?。', tmp_text).group()
    print(result)
    tmp_text = tmp_text.replace(result, '\n')
    results = re.findall('第 .*? 页', tmp_text)
    for result in results:
        tmp_text = tmp_text.replace(result, '')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(tmp_text)
    paragraphs_text = tmp_text
    # 4.抽取题目
    content_text = ''
    results = re.split(r'\d\.\d\.\d+\..*?第\d+题', paragraphs_text)
    for idx, result in tqdm(enumerate(results)):
        if idx > 0:
            result = result.replace('\n', '')
            content_text += (result + '\n')
    with open('content.txt', 'w', encoding='utf-8') as f:
        f.write(content_text)

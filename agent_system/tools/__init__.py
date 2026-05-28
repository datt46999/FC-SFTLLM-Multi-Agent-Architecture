from agent_system.tools.Search import web_search, wiki_search, arxiv_search
from agent_system.tools.code_interpreter import execute_code_multilang
from agent_system.tools.document_processing import save_and_read_file, download_file_from_url, extract_text_from_image, analyze_csv_file, analyze_excel_file
from agent_system.tools.image_generate import analyze_image, transform_image, draw_on_image, generate_simple_image, combine_images
from agent_system.tools.mathematical import *

IMAGE_GENERATE_TOOLS= [
    analyze_image,
    transform_image,
    draw_on_image,
    generate_simple_image,
    combine_images,
]
CODE_INTERPRETER_TOOLS= [
    analyze_csv_file,
    analyze_excel_file,
    execute_code_multilang,]
DOCUMENT_PROCESSING_TOOLS = [
    save_and_read_file,
    download_file_from_url,
    extract_text_from_image,
]
MATHEMATICAL_TOOLS= [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    square_root
    ]
EXTERNAL_TOOLS =[ 
    web_search,
    wiki_search,
    arxiv_search
]
from agent.tools.Search import web_search, wiki_search, arxiv_search
from agent.tools.code_interpreter import execute_code_multilang
from agent.tools.document_processing import save_and_read_file, download_file_from_url, extract_text_from_image, analyze_csv_file, analyze_excel_file
from agent.tools.image_generate import analyze_image, transform_image, draw_on_image, generate_simple_image, combine_images
from agent.tools.mathematical import *

TOOL_RESEARCH = [
    web_search,
    wiki_search,
    arxiv_search,
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    square_root,
    save_and_read_file,
    download_file_from_url,
    extract_text_from_image,
    analyze_csv_file,
    analyze_excel_file,
    execute_code_multilang,
    analyze_image,
    transform_image,
    draw_on_image,
    generate_simple_image,
    combine_images,
]
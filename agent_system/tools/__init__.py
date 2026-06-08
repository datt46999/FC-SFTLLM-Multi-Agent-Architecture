from agent_system.tools.Search import *
from agent_system.tools.code_interpreter import *


CODE_INTERPRETER = execute_code_multilang

TOOLS_RESEARCHER =[ 
    web_search,
    
]
TOOLS_SCRAPE =[
    scrape_webpages,
    wiki_search,
    arxiv_search
]

import gradio as gr

from langchain_core.messages import HumanMessage

from agent_system.agent.supervisor import create_graph

def greet(text_input):
    graph = create_graph()
    
    result = graph.invoke(
    {
        "messages": [HumanMessage(content=text_input)]
    }
    )
    
    return result["messages"][-1].content


demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()

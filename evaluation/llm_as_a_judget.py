import pandas as pd
import os
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

# from langchain_openai import ChatOpenAi
from langchain_core.messages import HumanMessage
from judges.classifiers.correctness import PollMultihopCorrectness
from judges.graders.correctness import PrometheusAbsoluteCoarseCorrectness
from judges.graders.response_quality import MTBenchChatBotResponseQuality

from agent_system.agent.supervisor import create_graph 


from together import Together

client = Together()
response = client.endpoints.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    display_name="nguyentandat28_d6b2: meta-llama - Llama-3.3-70B-Instruct-Turbo",
    hardware="2x_nvidia_h100_80gb_sxm",
    min_replicas=1,
    max_replicas=1
) 

dataset = load_dataset(
    "quotientai/natural-qa-random-67-with-AI-search-answers",
    data_files="data/natural-qa-random-67-with-AI-search-answers.parquet",
    split="train"
)

load_dotenv()

df = dataset.to_pandas()

def load_agent(query):
    graph = create_graph()

    results = graph.invoke(
        {
            "messages": [HumanMessage(content = query)]
        }
    )
    return results["messages"][-1].content


model = response 


# Initialize judges
correctness_classifier = PollMultihopCorrectness(model=model)
correctness_grader = PrometheusAbsoluteCoarseCorrectness(model=model)
response_quality_evaluator = MTBenchChatBotResponseQuality(model=model)


judgments = []

for _, row in df.iterrows():
    input_text = row['input_text']
    expected = row['completion']
    row_judgments = {}

    # Run your agent system
    try:
        output = load_agent(input_text)
    except Exception as e:
        output = f"ERROR: {e}"

    row_judgments['agent_output'] = output

    # Correctness Classifier
    classifier_judgment = correctness_classifier.judge(
        input=input_text, output=output, expected=expected
    )
    row_judgments['correctness_score'] = classifier_judgment.score
    row_judgments['correctness_reasoning'] = classifier_judgment.reasoning

    # Correctness Grader
    grader_judgment = correctness_grader.judge(
        input=input_text, output=output, expected=expected
    )
    row_judgments['correctness_grade'] = grader_judgment.score
    row_judgments['correctness_feedback'] = grader_judgment.reasoning

    # Response Quality
    quality_judgment = response_quality_evaluator.judge(
        input=input_text, output=output
    )
    row_judgments['quality_score'] = quality_judgment.score
    row_judgments['quality_feedback'] = quality_judgment.reasoning

    judgments.append(row_judgments)

# Combine results into DataFrame
results_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(judgments)], axis=1)

# Summary stats
print("Average correctness score:", results_df['correctness_score'].mean())
print("Average correctness grade:", results_df['correctness_grade'].mean())
print("Average quality score:", results_df['quality_score'].mean())

# Save
results_df.to_csv("agent_evaluation_results.csv", index=False)


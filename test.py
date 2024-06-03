import mlflow
import openai
import pandas as pd
import os

evakl_data = {
    "inputs" : ["what is Datascience",
                "what is MLOPS"],
    "ground_truth" : ["Data science is an interdisciplinary field that combines domain knowledge with statistics, data analysis, and machine learning."
    "It involves extracting insights and knowledge from structured and unstructured data."
    "Data scientists use a variety of tools and techniques to process and analyze large datasets."
    "The field incorporates methods from mathematics, statistics, computer science, and information science."
    "Data science aims to understand and interpret complex data to inform decision-making and solve problems."
    "Key tasks in data science include data cleaning, data integration, data visualization, and predictive modeling."
    "Machine learning, a subset of data science, involves creating algorithms that can learn from and make predictions on data."
    "Data science applications span multiple industries, including healthcare, finance, marketing, and technology."
    "Effective data science requires both technical skills and domain expertise to translate data insights into actionable strategies."
    "As data continues to grow in volume and complexity, the role of data science in driving innovation and efficiency becomes increasingly vital.",

    "MLOps, or Machine Learning Operations, is a set of practices for deploying, monitoring, and managing machine learning models in production."
    "It combines principles from DevOps with machine learning to streamline the ML lifecycle."
    "MLOps aims to automate and improve the process of bringing ML models from development to production."
    "It involves collaboration between data scientists, ML engineers, and IT operations teams."
    "Key components of MLOps include version control, continuous integration, continuous delivery, and continuous training."
    "MLOps practices help ensure the reproducibility, reliability, and scalability of ML models."
    "It addresses challenges such as model deployment, monitoring, and governance."
    "MLOps tools and frameworks facilitate model tracking, data management, and automated workflows."
    "Implementing MLOps can lead to faster model iterations, reduced deployment risks, and better model performance."
    "As the adoption of machine learning grows, MLOps becomes crucial for maintaining efficient and effective ML operations."
]
}


eval_data_df = pd.DataFrame(evakl_data)

#set mlflow experiment name
mlflow.set_experiment("LLM model evalution")

with mlflow.start_run() as run:
    sys_prompt = "Answer the following question in two sentences"
    logging_model_info = mlflow.openai.log_model(
                        
        model = "gpt-4",
        task = openai.chat.completions,
        artifact_path = "model",
        messages=[{"role": "System", "content": sys_prompt},
                    {"role": "user","content": "{question}"}],
                                  )
    

        # Use predefined question-answering metrics to evaluate our model.
    results = mlflow.evaluate(
        logging_model_info.model_uri,
        eval_data_df,
        targets="ground_truth",
        model_type="question-answering",
        extra_metrics=[mlflow.metrics.toxicity(), mlflow.metrics.latency(),mlflow.metrics.genai.answer_similarity()]
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    # Evaluation result for each data record is available in `results.tables`.
    eval_table = results.tables["eval_results_table"]
    df=pd.DataFrame(eval_table)
    df.to_csv('eval.csv')
    print(f"See evaluation table below: \n{eval_table}")



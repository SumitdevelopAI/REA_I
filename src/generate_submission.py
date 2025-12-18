import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from retriever import IntelligentSearcher


def generate_csv():
    searcher = IntelligentSearcher()

    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script.",
        "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests",
        "Graduate sales trainee with strong negotiation skills",
        "Customer service representative for a call center environment",
        "Project Manager with Agile and Scrum certification",
        "Mechanical Engineer proficient in CAD",
        "HR Manager with focus on recruitment strategy",
        "Senior Accountant with knowledge of international tax law"
    ]

    submission_data = []

    for query in test_queries:
        recommendations = searcher.search(query, top_k=5)
        for rec in recommendations:
            submission_data.append({
                "Query": query,
                "Assessment_url": rec["url"]
            })

    df = pd.DataFrame(submission_data)
    df.columns = ["Query", "Assessment_url"]
    df.to_csv("Sumit_Sharma.csv", index=False)


if __name__ == "__main__":
    generate_csv()

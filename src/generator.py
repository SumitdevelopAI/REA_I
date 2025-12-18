import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import LLM_MODEL, DEVICE
from retriever import IntelligentSearcher


class RAGPipeline:
    def __init__(self):
        # Initialize retriever
        self.search_engine = IntelligentSearcher()

        # Initialize language model
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to(DEVICE)

    def run(self, query):
        # Retrieve top-k relevant documents
        results = self.search_engine.search(query, top_k=3)

        if not results:
            return "I couldn't find any relevant assessments in the database.", []

        # Build context from retrieved documents
        context_text = "\n\n".join(
            f"Assessment: {r['doc']['name']}\nDetails: {r['doc']['description']}"
            for r in results
        )

        prompt = (
            "You are an expert HR assistant. Use the context below to answer the user's question. "
            "If the answer is not in the context, admit you don't know.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User Question:\n{query}\n\n"
            "Answer:"
        )

        # Generate response
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        ).to(DEVICE)

        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
            temperature=0.4,
            do_sample=True
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer, results


if __name__ == "__main__":
    pipeline = RAGPipeline()
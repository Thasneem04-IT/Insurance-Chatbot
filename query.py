import streamlit as st
from thirdai import licensing, neural_db as ndb
from openai import OpenAI
import nltk
import random
import os

# Download NLTK punkt
nltk.download("punkt")

# Activate ThirdAI licensing
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Activate ThirdAI licensing
thirdai_key = os.getenv("THIRDAI_KEY")
if thirdai_key:
    licensing.activate(thirdai_key)
else:
    raise ValueError("THIRDAI_KEY not found in environment variables.")
openai_client = OpenAI()

# Define functions
def get_references(query1):
    search_results = db.search(query1, top_k=1)
    references = []
    for result in search_results:
        references.append(result.text)
    return references

def get_answer(query1, references):
    return generate_answers(
        query1=query1,
        references=references,
    )

def generate_answers(query1, references):
    context = "\n\n".join(references[:3])

    prompt = f"Answer the following question in about 50 words using the context given: \nQuestion : {query1} \nContext: {context}"

    messages = [{"role": "user", "content": prompt}]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0
    )
    return response.choices[0].message.content

def generate_queries_chatgpt(query1):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {query1}"},
            {"role": "user", "content": "OUTPUT (5 queries):"}
        ]
    )

    # Access the content correctly
    generated_queries = response.choices[0].message.content.strip().split("\n")
    return generated_queries

def vector_search(query, pdf_files):
    available_docs = list(pdf_files.keys())
    random.shuffle(available_docs)
    selected_docs = available_docs[:random.randint(2, 6)]
    scores = {doc: round(random.uniform(0.7, 0.9), 2) for doc in selected_docs}
    return {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}

def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            fused_scores[doc] += 1 / (rank + k)

    return {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}

def generate_output(reranked_results, queries):
    return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())}"

# PDF files and NeuralDB setup
pdf_files = {
    "doc1": "indiafirst-simple-benefit-plan-brochure (2).pdf"
}
db = ndb.NeuralDB()
insertable_docs = []

for file in pdf_files:
    pdf_doc = ndb.PDF(pdf_files[file])
    insertable_docs.append(pdf_doc)
source_ids = db.insert(insertable_docs, train=True)

# Streamlit app with Chatbot style
def main():
    st.title("IndiaFirst Simple Benefit Plan Chatbot")
    
    # Initialize session state for storing chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_input := st.chat_input("Ask a question about the IndiaFirst Simple Benefit Plan..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response
        generated_queries = generate_queries_chatgpt(user_input)
        
        all_results = {}
        for query in generated_queries:
            search_results = vector_search(query, pdf_files)
            all_results[query] = search_results
        
        reranked_results = reciprocal_rank_fusion(all_results)
        references = get_references(user_input)
        
        if references:
            answer = get_answer(user_input, references)
            final_output = generate_output(reranked_results, generated_queries)
            bot_response = f"Answer: {answer}\n\n{final_output}"
        else:
            bot_response = "No references found."

        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Display bot message
        with st.chat_message("assistant"):
            st.markdown(bot_response)

if __name__ == "__main__":
    main()

import streamlit as st
from thirdai import licensing, neural_db as ndb
from openai import OpenAI
import os
import base64
import config
from langchain.memory import ConversationBufferMemory
if "THIRDAI_KEY" in os.environ:
    licensing.activate(os.environ["THIRDAI_KEY"])
else:
    licensing.activate(config.THIRDAI_KEY)

# Set OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

openai_client = OpenAI()


pdf_files = {
    "doc1": "cash-back-plan-brochuree.pdf",
    "doc2" :"indiafirst-life-saral-bachat-bima-plan-brochure.pdf",
    "doc3" : "indiafirst-life-smart-pay-plan-brochure.pdf",
    "doc4" : "gold-brochure (1).pdf",
    "doc5" : "indiafirst-life-guaranteed-benefit-plan-brochure1 (1).pdf",
    "doc6" : "indiafirst-life-long-guaranteed-income-plan-brochure (1).pdf",
    "doc7" : "indiafirst-life-radiance-smart-investment-plan-brochure.pdf",
    "doc8" : "indiafirst-money-balance-plan-brochure.pdf",
    "doc9" : "indiafirst-pos-cash-back-plan-brochure.pdf",
    "doc10" : "single-premium-brochure (1).pdf",
    "doc11" : "smart-save-plan-brochure.pdf",
    "doc12" : "tulip-brochure.pdf",
    "doc13" : "wealth-maximizer-brochure.pdf",
    "doc14" : "indiafirst-maha-jeevan-plan-brochure (1) (1).pdf",
    "doc15" : "indiafirst-money-balance-plan-brochure.pdf",
    "doc16" : "indiafirst-pos-cash-back-plan-brochure.pdf",
    "doc17" : "adb-presentation.pdf",
    "doc18" : "guaranteed-protection-plus-plan-brochure.pdf",
    "doc19" : "indiafirst-csc-shubhlabh-plan-brochure.pdf",
    "doc20" : "indiafirst-life-guaranteed-benefit-plan-brochure1 (1) (1).pdf",
    "doc21" : "indiafirst-life-micro-bachat-plan-brochure (1).pdf",
    "doc22" : "indiafirst-life-insurance-khata-plan-brochure (1).pdf",
    "doc23" : "accidental-death-benefit-rider-brochure.pdf",
    "doc24" : "indiafirst-life-long-guaranteed-income-plan-brochure (1).pdf",
    "doc25" : "indiafirst-life-micro-bachat-plan-brochure (1).pdf",
    "doc26" : "indiafirst-life-little-champ-plan-brochure.pdf",
    "doc27" : "IndiaFirst_Life_Guarantee_Of_Life_Dreams_Plan_Brochure.pdf",
    "doc28" : "indiafirst-life-group-disability-rider-policy-document-group.pdf",
    "doc29" : "indiafirst-anytime-plan.pdf.coredownload.inline.pdf",
    "doc30" : "Policies.pdf"  
}


db = ndb.NeuralDB()
insertable_docs = [ndb.PDF(pdf_files[file]) for file in pdf_files]
source_ids = db.insert(insertable_docs, train=False)

# Define functions
def get_references(query1):
    search_results = db.search(query1, top_k=1)
    return [result.text for result in search_results]

def generate_answers(query1, references):
    context = "\n\n".join(references[:3])
    prompt = (
        f"Please carefully read all the documents provided below and answer the user's question with specific details directly from the document. "
        f"Ensure the answer is accurate, complete, and directly references the relevant sections of the policy, including tables and structured data.\n\n"
        f"User Question: {query1}\n\n"
        f"Documents:\n{context}\n\n"
        f"Based on the information in the documents, provide a correct and specific answer to the user's question. "
        f"Include the policy name and any relevant details or conditions mentioned in the policy. "
        f"Focus on the data presented in the table, paying close attention to numeric values, labels, and any relevant units. "
        f"Interpret the numeric values accurately and include them in your response. Synthesize information from multiple tables if necessary."
        f"If multiple tables are relevant, synthesize information from all of them to form a complete answer."
    )
    memory = ConversationBufferMemory()
    memory_variables = memory.load_memory_variables({})
    conversation_history = memory_variables.get("chat_history", "")

    full_prompt = f"{conversation_history}\n\n{prompt}"

    messages = [{"role": "user", "content": full_prompt}]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7
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
    return response.choices[0].message.content.strip().split("\n")

# Reciprocal Rank Fusion function
def reciprocal_rank_fusion(search_results_dict, k=60):
    """
    This function performs reciprocal rank fusion on the given search results.
   
    Args:
        search_results_dict (dict): A dictionary where keys are query identifiers
                                    and values are dictionaries of document scores
                                    (where keys are document identifiers and values are scores).
        k (int): A parameter that controls the ranking boost for the top documents.
   
    Returns:
        dict: A dictionary of document IDs and their corresponding RRF scores,
              sorted in descending order of the scores.
    """
    fused_scores = {}

    # Iterate over each query's results
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            # Calculate the RRF score for this document
            fused_scores[doc] += 1 / (rank + k)

    # Sort the fused results by score in descending order
    return {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}

def generate_output(reranked_results, queries):
    return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())}"

def main():
    # Load external CSS
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def get_image_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    image_base64 = get_image_base64('logo.jpg')
    st.markdown(f'<img src="data:image/png;base64,{image_base64}" class="logo">', unsafe_allow_html=True)
    st.markdown("""
        <style>
        .centered-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 150px;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='centered-title'>AI-Insurance Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='centered-title1'>Hi there! I'm your friendly assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='centered-title1'>Whether you need help with something, I'm here to assist.</div>", unsafe_allow_html=True)
    st.markdown("<div class='centered-title1'>If you have a language preference, let me know?</div>", unsafe_allow_html=True)
    st.markdown("<div class='centered-title1'>How Can I Help You Today?</div>", unsafe_allow_html=True)

    # Initialize session state for storing chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-bubble'>{message['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # User input
    if user_input := st.chat_input("Ask any question", key="user_input"):
        # Add user message to chat history and display it immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)

        # Friendly greeting response
        if user_input.lower() in ["hi", "hello", "hii", "hey"]:
            bot_response = "Hello! How can I assist you today?"
        else:
            # Generate response
            generated_queries = generate_queries_chatgpt(user_input)
            references = get_references(user_input)
           
            if references:
                answer = generate_answers(user_input, references)
                bot_response = f"Answer: {answer}"
            else:
                # Generate a relevant response when no references are found
                bot_response = "It looks like I don't have specific references for that question, but I'm here to help! Could you ask it another way, or do you have another question in mind?"

        # Add bot response to chat history and display it immediately
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.markdown(f"<div class='assistant-bubble'>{bot_response}</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()

# Rag_pipeline
This is a simplified Retrieval Augmented Generation (RAG) pipeline that retrieve top k related context from simple 50-entries fake novel dataset. Demonstrating the power of RAG in enhancing LLM answers.

## 📁 Folder Structure
```
rag-project/
├── data/
│   └── fake_novel_dataset.json              # Contains context-question-answer entries
├── src/
│   ├── embed.py                         # Embedding generation logic
│   ├── retrieve.py                      # Retrieval from vector DB (FAISS)
│   └── augmented_LLM.py                      # Answer generation using LLM (optional)
├── gui_app.py
├── README.md
└── requirements.txt

```

## 📌 Features
- Simple dataset of context, question, and answer
- Embedding generation with `all-MiniLM-L6-v2`
- Top-k context retrieval with FAISS
- Answer generation using HuggingFace LLM : mistralai
- gui app that accept question and generate and display answer 

  ## 🚀 How to Run
```bash
# Clone repo
$ git clone https://github.com/yourusername/rag-project.git
$ cd rag-project

# Create virtual environment
$ python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
$ pip install -r requirements.txt
```

  # 🧪 Run Embedding
  ```terminal
  # Inside src/
  $ python embed.py   #which will create the context.index(which is embeddings database of dataset contexts)
  #launch gui app in your browser by writing in terminal (in activated venv) :
  $ streamlit run gui_app.py
  ```

gui_app.py utilize retrieve.py and augmented_LLM.py to:
* accept and generate embedding for user question
* retrieve top K similar contexts to question 
* compine retrieved context, user question and instructions in a promp and pass it to an LLM (in chat completion request)
*fetch and display augmented LLM answers

## Functions
*retrieve_top_k(q, k) : take question and integer k returning list of top k contexts similar to the question from the embedding database
*generate(q) : take question utilize retrieve_top_k() , pass context, instruction and question to an LLM and return string of the answer

## 📋 Sample Dataset Format
used ai to generate 50 entries about fake novel and edited about 10 of them to be a story created from my mind "love to live alone" about man named loner and his lover ella. to make sure that it's a new completely unknown story.

```json
[
  {
    "novel_title": "like to live alone",
    "context": "In the novel 'like to live alone', a man named loner from 'silence shadow' village disappeared into                 the mists. Legends say he's living happily alone on mountain fijstu.",
    "question": "Who is diappeared in the novel 'love to live alone'?",
    "answer": "loner"
  },
  ...
]
```
    
# 📸 Demo Screenshot
<img width="1920" height="1080" alt="Screenshot (123)" src="https://github.com/user-attachments/assets/196ae68d-aed9-405a-a18c-77ce5502c30d" />

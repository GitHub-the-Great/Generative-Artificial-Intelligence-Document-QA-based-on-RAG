#!/usr/bin/env python
# coding: utf-8

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

from rich.console import Console
from rich.logging import RichHandler

# Set up logging
console = Console(stderr=True, record=True)
log_handler = RichHandler(rich_tracebacks=True, console=console, markup=True)
logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[log_handler])
log = logging.getLogger("rich")
log.setLevel(logging.INFO)

# Import LangChain components
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from rouge_score import rouge_scorer

# Constants
STUDENT_ID = "112101014_0429_private"  # Change this to your student ID
PUBLIC_DATASET_PATH = "public_dataset.json"
PRIVATE_DATASET_PATH = "private_dataset.json"
OUTPUT_JSON_PATH = f"{STUDENT_ID}.json"

# Model Configuration
USING_GROQ = True  # Set to True to use Groq API, False to use local/VLLM
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Set your GROQ API key as environment variable
GROQ_API_KEY = "" # Set your GROQ API key as environment variable

# Set to False when running on the shared system with less resources
USE_FAISS_GPU = False

# Tunable parameters - feel free to optimize
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
RETRIEVE_TOP_K = 5
MODEL_TEMPERATURE = 0.1
MODEL_MAX_TOKENS = 128
EMBEDDING_BATCH_SIZE = 32

# Initialize scorer for ROUGE-L
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def setup_llm():
    """Set up the language model based on configuration."""
    if USING_GROQ:
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(
            model="llama-3.1-8b-instant",  # Using smaller model for the grading benefit
            # model="allam-2-7b",
            api_key=GROQ_API_KEY,
            temperature=MODEL_TEMPERATURE,
            max_tokens=MODEL_MAX_TOKENS
        )
    else:
        # Using local/VLLM if not using Groq
        from langchain_community.llms.vllm import VLLMOpenAI
        
        API_ENDPOINT = "http://localhost:8000/v1"  # Change as needed
        API_KEY = "dummy-key"
        USING_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        llm = VLLMOpenAI(
            base_url=API_ENDPOINT,
            api_key=API_KEY,
            model=USING_MODEL,
            temperature=MODEL_TEMPERATURE,
            max_tokens=MODEL_MAX_TOKENS,
        )
    
    return llm

def setup_embeddings():
    """Set up the embedding model."""
    try:
        # First try using OllamaEmbeddings which works well if Ollama is installed locally
        from langchain_ollama import OllamaEmbeddings
        
        embeddings = OllamaEmbeddings(
            model="snowflake-arctic-embed2:568m-l-fp16",  # Lightweight embedding model
            keep_alive=3000,
        )
        log.info("Using Ollama embeddings")
        return embeddings
    except Exception as e:
        log.warning(f"Ollama not available: {e}")
        
        # Fallback to HuggingFace embeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        
        model_name = "BAAI/bge-small-en-v1.5"  # Small but effective model
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"batch_size": EMBEDDING_BATCH_SIZE, "normalize_embeddings": True}
        )
        log.info(f"Using HuggingFace embeddings: {model_name}")
        return embeddings

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        log.info(f"Loaded dataset from {file_path} with {len(dataset)} items")
        return dataset
    except Exception as e:
        log.error(f"Error loading dataset: {e}")
        raise

def create_document_chunks(text: str) -> List[Document]:
    """Create document chunks from text using RecursiveCharacterTextSplitter."""
    # Create documents from paragraphs
    documents = text.split("\n\n")
    docs = [Document(page_content=doc) for doc in documents if doc.strip()]
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    
    return text_splitter.split_documents(docs)

def setup_rag_chain(llm, vectorstore):
    """Set up the RAG chain for question answering."""
    # Define prompts for RAG
    SYSTEM_PROMPT = """You are an expert in Natural Language Processing research. 
Answer the question about NLP papers based solely on the provided context.
Provide direct, concise, and accurate responses that reflect only the information in the context.
If the context doesn't contain enough information to answer the question confidently, say "I don't know"."""

    RAG_TEMPLATE = """
Context information is below:
------------
{context}
------------

Given the context information and not prior knowledge, answer the following question (The shorter the answer, the better. You don't need to answer in full sentences. Only based on the context above, extract the shortest and most relevant sentences.):
Question: {input}

Answer:"""

    prompt = PromptTemplate(template=RAG_TEMPLATE, input_variables=["context", "input"])
    
    # Create RAG chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K}),
        combine_docs_chain=combine_docs_chain,
    )
    
    return retrieval_chain

def process_paper(paper: Dict[str, Any], llm, embeddings) -> Dict[str, Any]:
    """Process a single paper and generate an answer to the question."""
    # paper_id = paper.get("id", "unknown")
    title = paper.get("title", "")
    full_text = paper.get("full_text", "")
    question = paper.get("question", "")
    
    log.info(f"Processing paper: {title}")
    
    # Create document chunks
    doc_chunks = create_document_chunks(full_text)
    log.debug(f"Created {len(doc_chunks)} chunks")
    
    # Create vector store
    vectorstore_kwargs = {}
    if USE_FAISS_GPU:
        vectorstore_kwargs["nlist"] = min(len(doc_chunks), 50)  # Number of clusters
        vectorstore_kwargs["nprobe"] = min(len(doc_chunks) // 10 + 1, 10)  # Number of clusters to search
    
    vectorstore = FAISS.from_documents(doc_chunks, embeddings, **vectorstore_kwargs)
    
    # Set up RAG chain
    rag_chain = setup_rag_chain(llm, vectorstore)
    
    # Get answer
    response = rag_chain.invoke({"input": question})
    #print(response)
    #print("")
    answer = response.get("answer", "")
    
    # Get retrieved documents
    retrieved_docs = response.get("context", [])
    evidence = [doc.page_content for doc in retrieved_docs]
    
    result = {
        "title": title,
        "answer": answer,
        "evidence": evidence,
    }
    
    return result

def calculate_rouge_score(predictions: List[str], targets: List[str]) -> float:
    """Calculate ROUGE-L score for predictions vs targets."""
    if not predictions or not targets:
        return 0.0
    
    fmeasure_scores = []
    for target in targets:
        max_score = 0
        for prediction in predictions:
            scores = scorer.score(target=target, prediction=prediction)
            max_score = max(max_score, scores["rougeL"].fmeasure)
        fmeasure_scores.append(max_score)
    
    return sum(fmeasure_scores) / len(fmeasure_scores)

def main():
    """Main function to run the RAG system."""
    parser = argparse.ArgumentParser(description="RAG-based Document QA")
    parser.add_argument("--private", action="store_true", help="Run on private dataset")
    parser.add_argument("--output", type=str, default=OUTPUT_JSON_PATH, help="Output JSON path")
    args = parser.parse_args()
    
    # Determine which dataset to use
    dataset_path = PRIVATE_DATASET_PATH if args.private else PUBLIC_DATASET_PATH
    output_path = args.output
    
    # Set up components
    llm = setup_llm()
    embeddings = setup_embeddings()
    dataset = load_dataset(dataset_path)
    
    results = []
    
    # Process papers
    for paper in tqdm(dataset, desc="Processing papers"):
        result = process_paper(paper, llm, embeddings)
        results.append(result)
        
        # Calculate and display metrics for public dataset (which has ground truth)
        if not args.private and "answer" in paper and "evidence" in paper:
            ground_truth_answer = paper["answer"]
            ground_truth_evidence = paper["evidence"]
            
            # Calculate ROUGE-L score
            rouge_score = calculate_rouge_score(result["answer"], ground_truth_answer)
            # print(f"ROUGE-L score: {rouge_score:.4f}")
            rouge_score = calculate_rouge_score(result["evidence"], ground_truth_evidence)
            log.debug(f"ROUGE-L score: {rouge_score:.4f}")
            # print(f"ROUGE-L score: {rouge_score:.4f}")

    # Calculate average ROUGE-L scores
    """if rouge_answer_scores:
        avg_rouge_answer = sum(rouge_answer_scores) / len(rouge_answer_scores)
        print(f"Average ROUGE-L score (Answer): {avg_rouge_answer:.4f}")
    
    if rouge_evidence_scores:
        avg_rouge_evidence = sum(rouge_evidence_scores) / len(rouge_evidence_scores)
        print(f"Average ROUGE-L score (Evidence): {avg_rouge_evidence:.4f}")"""
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
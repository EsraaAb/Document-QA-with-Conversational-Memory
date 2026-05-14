import hashlib
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict
import numpy as np


class DocumentQA:
    """Document Question Answering system with memory — no vector DB"""
    
    ## Phase 1: document processing and embedding 
    ''' 
    This function is the constructor function of the Documnet QA class. 
    It initializes the class with the following parameters:
      1) model name : it is the model used to generate the answer for the question asked by the user. The default value is "phi3".
      2) Embedding model : it is the model used to create the embeddings for the chunks of text extracted from the document. 
      it creates sentence embeddings using the sentence transformer library. It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search. 
      
    The constructor also initializes the following variables:
        1) ollama_url : it is the url of the ollama API used to generate the answer for the question asked by the user.
        2) embed_model : it is the sentence transformer model used to create the embeddings for the chunks of text extracted from the document.
        3) chunks : it is a list that stores the chunks of text extracted from the document.
        4) chunk_embeddings : it is a list that stores the embeddings for the chunks of text extracted from the document.
        5) current_doc_name : it is a string that stores the name of the current document loaded in the system.
        6) conversation_history : it is a list that stores the conversation history between the user and the assistant.
    '''
    def __init__(self, model_name: str = "phi3", embed_model: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        
        print(f"Loading embedding model: {embed_model}...")
        self.embed_model = SentenceTransformer(embed_model)
        
        # Store chunks in memory (no vector database)
        self.chunks = []           # List of text chunks
        self.chunk_embeddings = [] # List of corresponding embeddings
        self.current_doc_name = None
        self.conversation_history = []
        
        print("System ready!")




    


    ''' 
    This function takes the pdf file to extract text from it.
    - it initializes an empty string variable called text to store the extracted text.
    - it uses the PdfReader class from the pypdf library to read the pdf file and loop through each page to extract the text using the extract_text() method. 
    - The extracted text is then added to the text variable with a newline character. 
    - finally, it returns the whole extracted text.
    - If there is an error while reading the pdf file, it returns an error message with the exception details.
    '''
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    







    '''
    This function takes the txt file to extract text from it.
    - it uses the open() function to read the txt file in utf-8 encoding and returns the content of the file as a string.
    - If there is an error while reading the txt file, it returns an error message with the exception details. 
    '''
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading TXT: {str(e)}"
    









    '''
    This functions is used to calculate the distance between the query embedding and the chunk embeddings using cosine similarity. 
    it takes two lists:
    - a : it is the embedding of the query.
    - b : it is the embedding of the chunk.
    The function converts the lists to numpy arrays to calculate the cosine similarity using the formula:
    cosine_similarity = (a . b) / (||a|| * ||b||)
    where:
        - a . b is the dot product of the two vectors.
        - ||a|| is the magnitude of vector a.
        - ||b|| is the magnitude of vector b.
    The function returns the cosine similarity score as a float. 
    '''
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    








    '''
    This function is used to load the document and process it for question answering.
    - it takes the file path of the document "and lower it because the extension can be uppercase" as input and checks the file extension to determine
        - if the file is a pdf, it calls its function 
        - if the file is a txt, it calls its function
        - if the file type is unsupported, it returns an error message.
    
    - If the text extraction is successful, it chunks the text using the chunk_text() function and creates embeddings for each chunk using the embedding model.
    - It also resets the conversation history for the new document and generates a document name for display
    '''
    def load_document(self, file_path: str) -> str:
        """Load document: extract text, chunk, embed, store in memory"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_ext == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            return f"Unsupported file type: {file_ext}. Use PDF or TXT."
        
        if not text or "Error" in text:
            return text
        
        # Chunk the text
        self.chunks = self.chunk_text(text)
        
        # Create embeddings for all chunks
        print(f"Creating embeddings for {len(self.chunks)} chunks...")
        self.chunk_embeddings = []
        for chunk in self.chunks:
            embedding = self.embed_model.encode(chunk).tolist()
            self.chunk_embeddings.append(embedding)
        
        # Reset conversation memory for new document
        self.conversation_history = []
        
        # Generate document name for display
        doc_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        self.current_doc_name = f"doc_{doc_hash}"
        
        return f"Loaded {len(self.chunks)} chunks from {Path(file_path).name}"
    









    
    '''
    This function is used to split the extracted text into overlapping chunks using sentence boundaries.
    - it takes the text as input and splits it into sentences using the period followed by a space as a delimiter. It also replaces newline characters with spaces to ensure proper sentence splitting.
    - it initializes an empty list called chunks to store the resulting chunks, an empty list called current_chunk to build the current chunk, and a variable called current_length to keep track of the number of words in the current chunk.
    - it iterates through each sentence and checks if adding the sentence to the current chunk would exceed the specified chunk size. If it does and the current chunk is not empty, it adds the current chunk to the chunks list and starts a new chunk with an overlap of the last few sentences from the previous chunk to maintain context.
    - it continues to add sentences to the current chunk until it reaches the end of the sentences. If there are any remaining sentences in the current chunk after the loop, it adds that chunk to the chunks list as well.
    - finally, it returns the list of chunks. If no chunks were created (which can happen if the input text is very short), it returns a list containing the original text as a single chunk.           
    '''
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks using sentence boundaries"""
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if not sentence.endswith('.'):
                sentence += '.'
            
            words = len(sentence.split())
            
            if current_length + words > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_sentences.copy()
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    











    '''
    This function is used to retrieve the most relevant chunks of text from the document based on the user's query using cosine similarity.
       - it takes the user's query as input and embeds it using the embedding model to get the query embedding.
       - it then calculates the cosine similarity between the query embedding and each of the chunk embeddings stored in memory. It stores the similarity scores along with the corresponding chunk indices in a list called similarities.

       - it sorts the similarities list in descending order based on the similarity scores and selects the top_k most relevant chunks.
       - finally, it returns a list of dictionaries containing the relevant chunks of text and their corresponding similarity scores. If no chunks are available, it returns an empty list.
    '''
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant chunks using cosine similarity
        No vector DB — computes similarity on the fly
        """
        if not self.chunks:
            return []
        
        # Embed the query
        query_embedding = self.embed_model.encode(query).tolist()
        
        # Calculate similarity with every chunk
        similarities = []
        for i, chunk_emb in enumerate(self.chunk_embeddings):
            score = self.cosine_similarity(query_embedding, chunk_emb)
            similarities.append((i, score))
        
        # Sort by similarity (higher is better) and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = similarities[:top_k]
        
        # Return relevant chunks
        relevant_chunks = []
        for idx, score in top_indices:
            relevant_chunks.append({
                'text': self.chunks[idx],
                'score': score
            })
        
        return relevant_chunks
    













    # Phase 2: question answering with context and memory
    '''
    This function is used to build the prompt for generating answers based on the user's question and the relevant chunks of text.
    it takes 2 inputs: 
      1_ user question : it is the question asked by the user about the document.
      2_ relevant chunks : it is a list of dictionaries containing the relevant chunks of text and their corresponding similarity scores retrieved from the document based on the user's question.
    
    
    iterates over the relevant chunks and concatenates their text into a single string called context, separating each chunk with two newline characters for better readability.

    It also constructs a string called history_str that contains the previous conversation history between the user and the assistant, if available. 
    It formats the history by indicating whether each message was from the user or the assistant.
    if there is conversation history, it adds a header "Previous conversation:" followed by the formatted history messages. 
    only the last 4 messages are included to keep the prompt concise.

    Finally, it builds the complete prompt by combining the context, conversation history, and the user's question. The prompt instructs the assistant to answer the question based solely on the provided document context and to indicate if there isn't enough information to answer the question. The function returns the constructed prompt as a string.
    '''
    def build_prompt(self, question: str, relevant_chunks: List[Dict]) -> str:
        """Build prompt with context and conversation history"""
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        
        history_str = ""
        if self.conversation_history:
            history_str = "Previous conversation:\n"
            for msg in self.conversation_history[-4:]:
                role = "User" if msg['role'] == 'user' else "Assistant"
                history_str += f"{role}: {msg['content']}\n"
        
        prompt = f"""You are a helpful assistant answering questions about a document.

{history_str}

Document context:
{context}

Question: {question}

Answer based ONLY on the document context above. If the answer isn't in the context, say "I don't have enough information to answer that based on the document."

Answer:"""
        
        return prompt
    














   
   
    '''
    This function is used to generate the answer for the user's question using the Ollama API.
     - it takes the constructed prompt as input and sends a POST request to the Ollama API with the model name, prompt, and generation options such as temperature and number of tokens to predict.
     - If the API call is successful and returns a status code of 200, it extracts the generated answer from the API response and returns it. If there is an error in the API call, it returns an error message with the status code or exception details.
    '''
    def generate_answer(self, prompt: str) -> str:
        """Generate answer using Ollama"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("response", "Error generating answer")
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    


















    '''
    This function is used to handle the user's question and generate an answer based on the loaded document and conversation history.
     - it takes the user's question and the conversation history as input. If the question is empty or only contains whitespace, it returns a message prompting the user to enter a question along with the current conversation history.
     - If the question is valid, it calls the ask() method to get the answer from the QA system based on the question, relevant chunks from the document, and conversation history.
     - It then updates the conversation history by appending the user's question and the assistant's answer as new entries in the history list. Each entry is a dictionary with a 'role' key indicating whether it's from the user or the assistant, and a 'content' key containing the text of the message.
     - Finally, it returns an empty string for the question input (to clear the input field) and the updated conversation history to be displayed in the chat interface.
     - This function is designed to be used in a Gradio interface where the user can ask questions and see the conversation history in a chat-like format.
    '''

    def ask(self, question: str) -> str:
        """Main method: ask a question about the loaded document"""
        if not self.chunks:
            return "No document loaded. Please upload a document first."
        
        # Retrieve relevant chunks using cosine similarity
        relevant_chunks = self.retrieve_relevant_chunks(question)
        
        if not relevant_chunks:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Build prompt with context and history
        prompt = self.build_prompt(question, relevant_chunks)
        
        # Generate answer
        answer = self.generate_answer(prompt)
        
        # Save to conversation history
        self.conversation_history.append({'role': 'user', 'content': question})
        self.conversation_history.append({'role': 'assistant', 'content': answer})
        
        return answer
    









 


    ''' 
    This function is used to clear the conversation history between the user and the assistant.
    - it resets the conversation_history list to an empty list and returns a message indicating that the conversation history has been cleared. 
    This function can be called when the user wants to start a new conversation
    '''
    def clear_memory(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared"
    













   

    '''
    This function is used to get the current status of the system, including whether a document is loaded, the number of chunks extracted from the document, and the number of messages in the conversation history.
    - If there are chunks loaded in the system, it returns a string indicating the name of the loaded document, the number of chunks extracted from the document, and the number of messages in the conversation history.
    - If no document is loaded, it returns a message prompting the user to upload a PDF or TXT file to start using the system.
    This function can be used to display the current status of the system in the user interface, providing feedback to the user about the loaded document and conversation history.     
    '''
    def get_status(self) -> str:
        """Get current system status"""
        if self.chunks:
            return f"Document loaded: {self.current_doc_name} | Chunks: {len(self.chunks)} | Messages: {len(self.conversation_history)}"
        else:
            return "No document loaded. Upload a PDF or TXT file to start."


 


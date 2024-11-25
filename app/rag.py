import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
import PyPDF2
import docx
import os
import uuid
from typing import List, Dict, Tuple
import numpy as np
from time import sleep
from datetime import datetime
import concurrent.futures
import gc
import tiktoken
from audiorecorder import audiorecorder
from pydub import AudioSegment
import whisper
import tempfile

# Initialize OpenAI client with API key and base_url from secrets
client = OpenAI(api_key=st.secrets["openai"]["api_key"], base_url=st.secrets["openai"]["base_url"])

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx']
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
        self.max_tokens = 8000  # Setting below 8192 for safety margin
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.tokenizer.encode(text))
    
    def read_file_in_chunks(self, file, chunk_size=2*1024*1024):  # 2MB chunks for file reading
        """Read file in chunks to prevent memory issues"""
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}")
        
        content = []
        try:
            if file_extension == '.txt':
                file_content = b''
                while chunk := file.read(chunk_size):
                    file_content += chunk
                content = file_content.decode('utf-8').split('\n')
            
            elif file_extension == '.pdf':
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content.extend(page_text.split('\n'))
            
            elif file_extension == '.docx':
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    if para.text.strip():
                        content.append(para.text)
            
            # Clean up
            gc.collect()
            return '\n'.join(content), file.name
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Split text into smaller chunks with overlap and token limit check"""
        chunks = []
        sentences = text.split('.')
        current_chunk = []
        current_size = 0
        current_tokens = 0
        
        batch_size = 100  # Process 100 sentences at a time
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            for sentence in batch:
                sentence = sentence.strip() + '.'
                sentence_size = len(sentence)
                sentence_tokens = self.count_tokens(sentence)
                
                # Check if adding this sentence would exceed token limit
                potential_chunk = ' '.join(current_chunk + [sentence])
                potential_tokens = self.count_tokens(potential_chunk)
                
                if potential_tokens > self.max_tokens or current_size + sentence_size > chunk_size:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        
                        # Keep last few sentences for overlap, checking token limit
                        overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                        overlap_text = ' '.join(overlap_sentences)
                        if self.count_tokens(overlap_text) < self.max_tokens:
                            current_chunk = overlap_sentences
                            current_size = sum(len(s) for s in overlap_sentences)
                            current_tokens = self.count_tokens(overlap_text)
                        else:
                            current_chunk = []
                            current_size = 0
                            current_tokens = 0
                
                # Only add sentence if it doesn't exceed token limit by itself
                if sentence_tokens < self.max_tokens:
                    current_chunk.append(sentence)
                    current_size += sentence_size
                    current_tokens = self.count_tokens(' '.join(current_chunk))
                else:
                    # Handle long sentences by splitting them into smaller chunks
                    words = sentence.split()
                    temp_chunk = []
                    temp_tokens = 0
                    
                    for word in words:
                        temp_chunk.append(word)
                        temp_text = ' '.join(temp_chunk)
                        temp_tokens = self.count_tokens(temp_text)
                        
                        if temp_tokens > self.max_tokens:
                            if len(temp_chunk) > 1:
                                chunks.append(' '.join(temp_chunk[:-1]))
                            temp_chunk = [word]
                            temp_tokens = self.count_tokens(word)
                    
                    if temp_chunk:
                        current_chunk.extend(temp_chunk)
                        current_tokens = self.count_tokens(' '.join(current_chunk))
            
            # Clean up batch memory
            gc.collect()
        
        # Add the last chunk if it exists and is within token limit
        if current_chunk:
            final_chunk = ' '.join(current_chunk)
            if self.count_tokens(final_chunk) <= self.max_tokens:
                chunks.append(final_chunk)
        
        return chunks

class VectorStore:
    def __init__(self):
        """Initialize Pinecone vector store using secrets"""
        self.pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
        self.index_name = st.secrets["pinecone"]["index_name"]
        
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(name=self.index_name, dimension=1536, metric='cosine')
        
        self.index = self.pc.Index(self.index_name)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in one API call with automatic chunk splitting"""
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            processed_texts = []
            
            for text in texts:
                token_count = len(tokenizer.encode(text))
                if token_count > 6500:
                    # Split oversized text into smaller chunks
                    words = text.split()
                    current_chunk = []
                    current_tokens = 0
                    
                    for word in words:
                        word_tokens = len(tokenizer.encode(word))
                        if current_tokens + word_tokens > 6000:  # Safety margin
                            # Add current chunk and start new one
                            processed_texts.append(' '.join(current_chunk))
                            current_chunk = [word]
                            current_tokens = word_tokens
                        else:
                            current_chunk.append(word)
                            current_tokens += word_tokens
                    
                    # Add any remaining text
                    if current_chunk:
                        processed_texts.append(' '.join(current_chunk))
                else:
                    processed_texts.append(text)
            
            # Process all texts in batches to avoid API limits
            batch_size = 5
            all_embeddings = []
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                all_embeddings.extend([data.embedding for data in response.data])
                sleep(0.1)  # Rate limiting
            
            return all_embeddings
        except Exception as e:
            st.error(f"Error getting embeddings: {str(e)}")
            return []
    
    def process_chunk_batch(self, chunks: List[str], source_file: str, start_idx: int) -> List[Dict]:
        """Process a batch of chunks into vectors"""
        embeddings = self.get_embeddings_batch(chunks)
        vectors = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding:
                chunk_id = f"{source_file}_{str(uuid.uuid4())}"
                vectors.append({
                    'id': chunk_id,
                    'values': embedding,
                    'metadata': {
                        'text': chunk,
                        'source': source_file,
                        'chunk_index': start_idx + i,
                        'upload_time': datetime.now().isoformat()
                    }
                })
        
        return vectors

    def upload_chunks(self, chunks: List[str], source_file: str, progress_bar=None):
        """Upload text chunks to Pinecone with optimized batching"""
        if not chunks:
            return
        
        # Optimized batch sizes for medium documents
        embedding_batch_size = 20  # Reduced for safety
        upsert_batch_size = 40    # Reduced for safety
        
        # Initialize token counter
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        total_chunks = len(chunks)
        vectors_buffer = []
        
        for i in range(0, total_chunks, embedding_batch_size):
            batch = chunks[i:i + embedding_batch_size]
            
            # Verify token count for each chunk in batch
            valid_chunks = []
            for chunk in batch:
                token_count = len(tokenizer.encode(chunk))
                if token_count <= 6500:  # Safety margin
                    valid_chunks.append(chunk)
                else:
                    st.warning(f"Skipping chunk with {token_count} tokens")
                    
            if not valid_chunks:
                continue
                
            # Process batch
            vectors = self.process_chunk_batch(valid_chunks, source_file, i)
            vectors_buffer.extend(vectors)
            
            # Upsert when buffer is full
            if len(vectors_buffer) >= upsert_batch_size:
                try:
                    self.index.upsert(vectors=vectors_buffer)
                    vectors_buffer = []
                except Exception as e:
                    st.error(f"Error upserting vectors: {str(e)}")
                    vectors_buffer = []  # Clear buffer on error
            
            # Update progress
            if progress_bar is not None:
                progress = min((i + len(batch)) / total_chunks, 1.0)
                progress_bar.progress(progress)
            
            # Pause between batches to prevent overload
            sleep(0.2)
            gc.collect()
        
        # Upsert remaining vectors
        if vectors_buffer:
            try:
                self.index.upsert(vectors=vectors_buffer)
            except Exception as e:
                st.error(f"Error upserting final vectors: {str(e)}")
            if progress_bar is not None:
                progress_bar.progress(1.0)

    def get_all_documents(self) -> Dict[str, Dict]:
        """Get list of all uploaded documents with metadata"""
        try:
            results = self.index.query(
                vector=[0] * 1536,
                top_k=10000,
                include_metadata=True
            )
            
            documents = {}
            for match in results.matches:
                source = match.metadata.get('source')
                if source and source not in documents:
                    documents[source] = {
                        'upload_time': match.metadata.get('upload_time', datetime.now().isoformat()),
                        'total_chunks': match.metadata.get('total_chunks', 0)
                    }
            
            return documents
        except Exception as e:
            st.error(f"Error fetching documents: {str(e)}")
            return {}
    
    def delete_document(self, document_name: str):
        """Delete all vectors associated with a document"""
        try:
            results = self.index.query(
                vector=[0] * 1536,
                top_k=10000,
                include_metadata=True,
                filter={
                    "source": {"$eq": document_name}
                }
            )
            
            vector_ids = [match.id for match in results.matches]
            batch_size = 100
            
            for i in range(0, len(vector_ids), batch_size):
                batch = vector_ids[i:i + batch_size]
                self.index.delete(ids=batch)
                sleep(0.1)
                
            return True
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False
    
    def semantic_search(self, query: str, documents: List[str] = None, top_k: int = 3) -> List[Dict]:
        """Search for similar vectors in Pinecone with optional document filter"""
        try:
            query_embedding = self.get_embeddings_batch([query])[0]
            
            filter_query = None
            if documents and len(documents) > 0:
                filter_query = {
                    "source": {"$in": documents}
                }
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_query
            )
            return results.matches
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

def main():
    st.title("Rural Llama Health")
    
    if 'initialized' not in st.session_state:
        try:
            st.session_state.vector_store = VectorStore()
            st.session_state.doc_processor = DocumentProcessor()
            st.session_state.initialized = True
            st.session_state.documents_processed = False
        except Exception as e:
            st.error(f"Error initializing the system: {str(e)}")
            st.error("Please check your secrets.toml configuration")
            return
    
    tab1, tab2 = st.tabs(["Upload Documents", "Query Documents"])
    
    with tab1:
        st.info("System optimized for medium-sized documents with token limiting")
        
        # Document management section
        st.subheader("Manage Documents")
        documents = st.session_state.vector_store.get_all_documents()
        
        if documents:
            st.write("Currently uploaded documents:")
            for doc_name, metadata in documents.items():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"📄 {doc_name}")
                with col2:
                    upload_time = datetime.fromisoformat(metadata['upload_time'])
                    st.write(f"Uploaded: {upload_time.strftime('%Y-%m-%d %H:%M')}")
                with col3:
                    if st.button("Delete", key=f"del_{doc_name}"):
                        if st.session_state.vector_store.delete_document(doc_name):
                            st.success(f"Deleted {doc_name}")
                            st.rerun()
        
        # File upload section
        st.subheader("Upload New Documents")
        uploaded_files = st.file_uploader(
            "Upload documents (Supported formats: PDF, DOCX, TXT)", 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input("Chunk Size (characters)", 
                                           min_value=1000, 
                                           max_value=3000, 
                                           value=2000)
            with col2:
                overlap_size = st.number_input("Overlap Size (characters)", 
                                             min_value=100, 
                                             max_value=400, 
                                             value=200)
        
        if uploaded_files and st.button("Process Documents"):
            total_files = len(uploaded_files)
            
            processing_status = st.empty()
            overall_progress = st.progress(0)
            
            for file_idx, file in enumerate(uploaded_files):
                try:
                    processing_status.write(f"Processing file {file_idx + 1} of {total_files}: {file.name}")
                    
                    read_progress = st.progress(0)
                    chunk_progress = st.progress(0)
                    embed_progress = st.progress(0)
                    
                    read_progress.progress(0.5)
                    content, filename = st.session_state.doc_processor.read_file_in_chunks(file)
                    read_progress.progress(1.0)
                    
                    chunk_progress.progress(0.5)
                    chunks = st.session_state.doc_processor.chunk_text(
                        content, 
                        chunk_size=chunk_size, 
                        overlap=overlap_size
                    )
                    chunk_progress.progress(1.0)
                    
                    processing_status.write(f"Processing {len(chunks)} chunks for {filename}...")
                    st.session_state.vector_store.upload_chunks(
                        chunks, 
                        filename, 
                        progress_bar=embed_progress
                    )
                    
                    st.success(f"Successfully processed {filename}")
                    
                    overall_progress.progress((file_idx + 1) / total_files)
                    
                    del content
                    del chunks
                    gc.collect()
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    continue
            
            processing_status.write("All documents processed!")
            st.session_state.documents_processed = True
            st.rerun()
    
    with tab2:
        st.title("Ask Questions")
        st.text("to query our curated knowdledge base.")
        st.subheader("Text")
        
        documents = st.session_state.vector_store.get_all_documents()
        
        if not documents:
            st.warning("No documents found. Please upload some documents first.")
            return
        
        selected_docs = st.multiselect(
            "Select documents to search (leave empty to search all)",
            options=list(documents.keys()),
            default=None
        )
        
        query = st.text_input("Enter your question:")

        # Which Whisper model
        whisper_model_type = "Small"

        def process_audio(filename, model_type):
            model = whisper.load_model(model_type)
            result = model.transcribe(filename)
            return result["text"]

        # Recording y transcription
        st.subheader("Audio")
        audio = audiorecorder("Grabar audio", "Detener")
        if len(audio) > 0:
            try:
                if isinstance(audio, AudioSegment):
                    # Convert into bytes
                    st.audio(audio.export().read())  
                    audio.export("audio.wav", format="wav")
                    
                    # Add to chat history
                    transcribed_text = process_audio("audio.wav", whisper_model_type.lower())
                    print(transcribed_text)
                    query = transcribed_text
            except Exception as e:
                st.error(f"Error al procesar el audio: {e}")
                
        # WAV file uploaded for transcription
        uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
        try:
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_file_path = temp_file.name 

                transcribed_text = process_audio(temp_file_path, whisper_model_type.lower())

                query = transcribed_text
        except Exception as e:
            st.write(str(e))
        
        if query:
            with st.spinner("Searching for relevant information..."):
                results = st.session_state.vector_store.semantic_search(
                    query,
                    documents=selected_docs if selected_docs else None
                )
                
                if results:
                    context = "\n\n".join([match.metadata['text'] for match in results])
                    messages = [
                        {"role": "system", "content": "Eres un asistente de un profesional médico. Responde a la pregunta basándote en el contexto proporcionado. Si no puedes encontrar la respuesta en el contexto, indícalo. Incluye citas de los documentos fuente cuando sea posible."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ]
                    
                    try:
                        response = client.chat.completions.create(
                            model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                            messages=messages,
                            temperature=0
                        )
                        
                        st.write("Answer:", response.choices[0].message.content)
                        
                        st.subheader("Sources")
                        for i, match in enumerate(results, 1):
                            with st.expander(f"Source {i}: {match.metadata['source']}"):
                                st.write(match.metadata['text'])
                                st.write(f"Relevance Score: {match.score:.4f}")
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                else:
                    st.warning("No relevant information found. Please try a different question.")

if __name__ == "__main__":
    main()
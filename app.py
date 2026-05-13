import gradio as gr
from qa_system import DocumentQA



qa_system = DocumentQA(model_name="phi3")


def load_document(file):
    """Handle document upload"""
    if file is None:
        return "Please select a file to upload", qa_system.get_status()
    
    try:
        # Get file path from different Gradio formats
        if isinstance(file, dict):
            file_path = file.get('path', file.get('name'))
        elif hasattr(file, 'name'):
            file_path = file.name
        else:
            return "Invalid file format", qa_system.get_status()
        
        # Load document
        result = qa_system.load_document(file_path)
        status = qa_system.get_status()
        
        return result, status
        
    except Exception as e:
        return f"❌ Error loading file: {str(e)}", qa_system.get_status()


def ask_question(question, history):
    """Handle user question"""
    if not question or not question.strip():
        return "Please enter a question.", history
    
    # Get answer from QA system
    answer = qa_system.ask(question)
    
    # Fix for newer Gradio versions
    if history is None:
        history = []
    
    # New format: list of dictionaries with 'role' and 'content'
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    
    return "", history


def clear_conversation():
    """Clear chat history"""
    result = qa_system.clear_memory()
    return [], result


def update_status():
    """Update status display"""
    return qa_system.get_status()


# Create Gradio interface
with gr.Blocks(title="Document Q&A with Memory") as demo:
    
    gr.Markdown("""
    # 📄 Document Q&A with Conversation Memory
    
    Upload a PDF or TXT file and ask questions about it. The system remembers your conversation!
    
    **Powered by local LLMs** — Private, free, no API keys needed.
    """)
    
    with gr.Row():
        # Left column - Document upload
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Upload Document")
            file_input = gr.File(
                label="Choose a PDF or TXT file",
                file_types=[".pdf", ".txt"]
            )
            upload_btn = gr.Button("📤 Load Document", variant="primary")
            load_status = gr.Textbox(
                label="Load Status",
                interactive=False,
                value="No document loaded"
            )
            status_display = gr.Textbox(
                label="System Status",
                interactive=False,
                value=qa_system.get_status()
            )
            
            gr.Markdown("### 🧠 Features")
            gr.Markdown("""
            - ✅ Semantic search finds relevant content
            - ✅ Conversation memory for follow-up questions
            - ✅ Supports PDF and TXT files
            - ✅ Local embeddings (no API costs)
            - ✅ phi3 for answers
            """)
            
            clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")
        
        # Right column - Chat interface
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask something about the document...",
                    scale=4,
                    lines=2
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            gr.Markdown("""
            **💡 Example questions:**
            - What is the main topic?
            - Summarize the key points
            - What does it say about [topic]?
            - What are the main conclusions?
            """)
    
    # Connect events
    upload_btn.click(
        fn=load_document,
        inputs=[file_input],
        outputs=[load_status, status_display]
    )
    
    send_btn.click(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot]
    )
    
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot]
    )
    
    clear_btn.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[chatbot, status_display]
    )
    
    # Update status on load
    demo.load(
        fn=update_status,
        inputs=[],
        outputs=[status_display]
    )


if __name__ == "__main__":
    demo.launch(share=True)
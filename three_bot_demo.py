"""
Interactive Gradio Demo: Baseline vs Q-Learning vs PPO
Shows all three methods with LIVE Q-Learning updates!
"""

import gradio as gr
import sys
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from dotenv import load_dotenv
load_dotenv()

from backend.rl_agent import RLAgent
from backend.ppo_agent import PPOAgent
from backend.state import encode_state
from backend.rag_process import rag_process
from backend.retrieval_policies import action_to_filter
from groq import Groq

# Initialize
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
rag = rag_process()

# Load agents
q_agent = RLAgent(
    actions=["broad", "legal_only", "financial_only", "company_only"],
    verbose=True  # Enable verbose to see save messages
)

ppo_agent = PPOAgent(
    actions=["broad", "legal_only", "financial_only", "company_only"],
    verbose=False
)

# Track last Q-Learning response for feedback
last_q_state = None
last_q_action = None

def baseline_bot(question):
    """Baseline bot - short, basic answers, no learning"""
    try:
        chunks, _ = rag.query_documents(question, n_results=4)
        context = "\n\n".join(chunks[:2]) if chunks else ""
        
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a GHG consultant. Give brief answers."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nBrief answer:"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=150
        )
        
        answer = response.choices[0].message.content
        word_count = len(answer.split())
        
        return f"**Baseline Response:**\n\n{answer}\n\n---\n📏 Length: {word_count} words\n🎯 Strategy: basic (no action selection)\n❌ Learning: DISABLED"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def q_learning_bot(question):
    """Q-Learning bot with LIVE LEARNING capability"""
    global last_q_state, last_q_action
    
    try:
        # Use Q-Learning agent to select action
        state = encode_state(question, "")
        action = q_agent.select(state)
        
        # Save for feedback
        last_q_state = state
        last_q_action = action
        
        # Get chunks with learned action
        metadata_filter = action_to_filter(action) or None
        chunks, _ = rag.query_documents(question, n_results=4, metadata_filter=metadata_filter)
        context = "\n\n".join(chunks) if chunks else ""
        
        # Generate detailed answer
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert GHG consultant. Provide detailed answers."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nComprehensive answer:"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=600
        )
        
        answer = response.choices[0].message.content
        word_count = len(answer.split())
        
        # Show Q-values
        q_values = q_agent.q_for(state)
        q_display = "\n".join([f"  • {a}: {v:.3f}" for a, v in q_values.items()])
        
        return f"**Q-Learning Response:**\n\n{answer}\n\n---\n📏 Length: {word_count} words\n🎯 Action: {action} (selected by Q-Learning)\n✅ Learning: ENABLED (click 👍👎 below!)\n\n🧠 Current Q-Values:\n{q_display}"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def ppo_bot(question):
    """PPO bot - best performance, trained policy"""
    try:
        # Use PPO agent to select action
        state = encode_state(question, "")
        action = ppo_agent.select(state)
        
        # Get chunks with learned action
        metadata_filter = action_to_filter(action) or None
        chunks, _ = rag.query_documents(question, n_results=4, metadata_filter=metadata_filter)
        context = "\n\n".join(chunks) if chunks else ""
        
        # Generate detailed answer
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert GHG consultant. Provide detailed, comprehensive answers."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nComprehensive answer:"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=600
        )
        
        answer = response.choices[0].message.content
        word_count = len(answer.split())
        
        return f"**PPO Response:**\n\n{answer}\n\n---\n📏 Length: {word_count} words\n🎯 Action: {action} (selected by PPO policy)\n🏆 Performance: BEST (+13.9% vs baseline)"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def handle_thumbs_up():
    """Handle thumbs up feedback"""
    global last_q_state, last_q_action
    
    if last_q_state and last_q_action:
        old_q = q_agent.q_for(last_q_state)[last_q_action]
        q_agent.update(last_q_state, last_q_action, 1.0)
        new_q = q_agent.q_for(last_q_state)[last_q_action]
        updated_q_table = q_agent.q_for(last_q_state)
        
        # Force save
        q_agent._save()
        
        feedback = f"""
## ✅ Thank You for Your Positive Feedback!

**Q-Table Updated:**
- Action: `{last_q_action}`
- Old Q-value: {old_q:.3f}
- New Q-value: {new_q:.3f}
- Change: {new_q - old_q:+.3f}

**What This Means:**
The agent is now MORE LIKELY to use `{last_q_action}` for similar questions!

**Live Learning in Action!**
"""
        return feedback, updated_q_table
    else:
        return "⚠️ Ask a question first, then rate the Q-Learning bot's answer!", {}

def handle_thumbs_down():
    """Handle thumbs down feedback"""
    global last_q_state, last_q_action
    
    if last_q_state and last_q_action:
        old_q = q_agent.q_for(last_q_state)[last_q_action]
        q_agent.update(last_q_state, last_q_action, -1.0)
        new_q = q_agent.q_for(last_q_state)[last_q_action]
        updated_q_table = q_agent.q_for(last_q_state)
        
        # Force save
        q_agent._save()
        
        feedback = f"""
## 👎 Thank You for Your Feedback!

**Q-Table Updated:**
- Action: `{last_q_action}`
- Old Q-value: {old_q:.3f}
- New Q-value: {new_q:.3f}
- Change: {new_q - old_q:+.3f}

**What This Means:**
The agent is now LESS LIKELY to use `{last_q_action}` for similar questions!

**Live Learning in Action!**
"""
        return feedback, updated_q_table
    else:
        return "⚠️ Ask a question first, then rate the Q-Learning bot's answer!", {}

def compare_all_bots(question):
    """Get responses from all three bots"""
    baseline_resp = baseline_bot(question)
    q_resp = q_learning_bot(question)
    ppo_resp = ppo_bot(question)
    
    # Get current Q-table for display
    if last_q_state:
        q_table = q_agent.q_for(last_q_state)
    else:
        q_table = {"info": "Ask a question to see Q-values"}
    
    return baseline_resp, q_resp, ppo_resp, "", q_table

# Sample questions (display shortened, but full question gets written when clicked)
examples = [
    ["What is Scope 1 emissions?", "What is Scope 1 emissions?"],
    ["How do you calculate GHG emissions?", "How do you calculate GHG emissions?"],
    ["What are the reporting requirements?", "What are the reporting requirements for GHG emissions?"],
    ["What is the Paris Agreement?", "What is the Paris Agreement and its temperature goals?"],
    ["Scope 2 vs Scope 3?", "What is the difference between Scope 2 and Scope 3 emissions?"],
    ["Carbon accounting methods?", "What are the different methods for carbon accounting?"],
    ["Science-based targets?", "What are science-based targets and how do they work?"],
    ["GHG Protocol standards?", "What are the GHG Protocol standards and guidelines?"],
    ["Carbon offsets explained?", "What are carbon offsets and how do they work in practice?"],
    ["TCFD framework details?", "What is the TCFD framework and its disclosure requirements?"],
    ["Emission factors guide?", "How do emission factors work in GHG calculations?"],
    ["Scope 3 measurement challenges?", "What are the key challenges in measuring Scope 3 emissions?"],
    ["GHG inventory process?", "How do you conduct a comprehensive GHG inventory?"],
    ["Market vs location-based?", "What is the difference between market-based and location-based accounting?"],
    ["Carbon footprint reporting?", "How do companies report their carbon footprint to stakeholders?"],
    ["Net zero strategies?", "What are the main strategies for achieving net zero emissions?"],
    ["Supply chain emissions?", "How can companies reduce their supply chain emissions?"],
    ["Carbon pricing mechanisms?", "What are the different carbon pricing mechanisms available?"],
    ["Renewable energy credits?", "How do renewable energy credits work in carbon accounting?"],
    ["Climate risk assessment?", "How do companies assess and manage climate-related risks?"],
]

# Create interface with custom CSS
custom_css = """
.green-button {
    background-color: #28a745 !important;
    border-color: #28a745 !important;
}
.green-button:hover {
    background-color: #218838 !important;
    border-color: #1e7e34 !important;
}
.red-button {
    background-color: #dc3545 !important;
    border-color: #dc3545 !important;
}
.red-button:hover {
    background-color: #c82333 !important;
    border-color: #bd2130 !important;
}
.baseline-bot {
    border-left: 4px solid #6c757d !important;
    background-color: #343a40 !important;
    padding: 15px !important;
    border-radius: 8px !important;
    height: 400px !important;
    overflow-y: auto !important;
    color: #ffffff !important;
}
.qlearning-bot {
    border-left: 4px solid #007bff !important;
    background-color: #0056b3 !important;
    padding: 15px !important;
    border-radius: 8px !important;
    height: 400px !important;
    overflow-y: auto !important;
    color: #ffffff !important;
}
.ppo-bot {
    border-left: 4px solid #ffc107 !important;
    background-color: #b8860b !important;
    padding: 15px !important;
    border-radius: 8px !important;
    height: 400px !important;
    overflow-y: auto !important;
    color: #ffffff !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="RL Chatbot Demo", css=custom_css) as demo:
    gr.Markdown("""
    # 🤖 Three-Way Chatbot Comparison: Baseline vs Q-Learning vs PPO
    ## Interactive RL Demo with Live Learning!
    """)
    
    gr.Markdown("""
    ### 💡 How to Use:
    1. **Ask a question** to all three bots
    2. **Compare** their answers (length, detail, quality)
    3. **Click 👍 or 👎** on Q-Learning bot to **train it LIVE**
    4. **Watch Q-table update** in real-time!
    """)
    
    with gr.Row():
        question_input = gr.Textbox(
            label="💬 Ask a GHG Question:",
            placeholder="e.g., What is Scope 1 emissions?",
            lines=2,
            scale=4
        )
        submit_btn = gr.Button("🚀 Ask All Bots", variant="primary", scale=1, size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📝 Baseline Bot")
                gr.Markdown("*No Training*")
                baseline_output = gr.Markdown(
                    value="",
                    elem_classes="baseline-bot",
                    show_copy_button=False
                )
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎓 Q-Learning Bot")
                gr.Markdown("*Interactive Learning*")
                q_output = gr.Markdown(
                    value="",
                    elem_classes="qlearning-bot",
                    show_copy_button=False
                )
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🏆 PPO Bot")
                gr.Markdown("*Best Performance*")
                ppo_output = gr.Markdown(
                    value="",
                    elem_classes="ppo-bot",
                    show_copy_button=False
                )
    
    # Feedback section
    gr.Markdown("### 🎯 Rate Q-Learning Bot (LIVE LEARNING)")
    gr.Markdown("*Click feedback to train the Q-Learning bot in real-time!*")
    
    with gr.Row():
        thumbs_up_btn = gr.Button("👍 Thumbs Up", elem_classes="green-button", size="lg")
        thumbs_down_btn = gr.Button("👎 Thumbs Down", elem_classes="red-button", size="lg")
    
    feedback_display = gr.Markdown(value="*Rate the Q-Learning bot's answer above to see live learning!*")
    
    # Add Q-Table viewer
    gr.Markdown("### 📊 Live Q-Table View")
    gr.Markdown("*See the Q-table update in real-time!*")
    q_table_display = gr.JSON(label="Current Q-Table", value={})
    
    # Connect buttons
    submit_btn.click(
        fn=compare_all_bots,
        inputs=[question_input],
        outputs=[baseline_output, q_output, ppo_output, feedback_display, q_table_display]
    )
    
    thumbs_up_btn.click(
        fn=handle_thumbs_up,
        inputs=[],
        outputs=[feedback_display, q_table_display]
    )
    
    thumbs_down_btn.click(
        fn=handle_thumbs_down,
        inputs=[],
        outputs=[feedback_display, q_table_display]
    )
    
    gr.Examples(
        examples=[[ex[1]] for ex in examples],  # Use full questions as examples
        inputs=question_input,
        label="💡 Sample Questions:"
    )
    
    with gr.Accordion("📊 Experiment Results & Details", open=False):
        gr.Markdown("""
        ### Offline Experiment Results:
        
        | Method | Avg Score | Success Rate | Improvement |
        |--------|-----------|--------------|-------------|
        | **Baseline** | 79% | 100% | - |
        | **Q-Learning** | 85% | 100% | +7.6% |
        | **PPO** | **90%** | 90% | **+13.9%** |
        
        ### Key Insights:
        - ✅ RL improves answer quality and detail
        - ✅ PPO outperforms Q-Learning (+6.3%)
        - ✅ Both RL methods beat baseline significantly
        - ✅ Q-Learning offers transparency (visible Q-table)
        - ✅ PPO offers best performance (advanced learning)
        
        ### Technical Configuration:
        - **Baseline**: 150 tokens, temp=0.7, action=broad
        - **Q-Learning**: 600 tokens, temp=0.3, learned action, Q-table
        - **PPO**: 600 tokens, temp=0.3, learned action, neural policy
        
        ### Demo Features:
        - 🎯 **Three-way comparison** - See all methods at once
        - 👍👎 **Interactive learning** - Train Q-Learning bot live
        - 📊 **Real-time Q-values** - Watch learning happen
        - 🏆 **Best method shown** - PPO demonstrates peak performance
        """)

# Launch
if __name__ == "__main__":
    print("Starting Three-Bot Interactive Demo...")
    print("Baseline | Q-Learning (Interactive) | PPO (Best)")
    print("Click feedback buttons to train Q-Learning bot LIVE!")
    demo.launch(share=False, server_port=7860)


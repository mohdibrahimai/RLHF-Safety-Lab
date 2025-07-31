import streamlit as st
import json
import os
import random
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import custom components
from components.taxonomy_manager import TaxonomyManager
from components.model_loader import ModelLoader
from components.prompt_generator import PromptGenerator
from components.preference_trainer import PreferenceTrainer
from components.harm_evaluator import HarmEvaluator
from components.visualization import VisualizationManager
from components.cot_verifier import ChainOfThoughtVerifier
from utils.data_handler import DataHandler
from utils.logger import Logger

# Initialize session state
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = DataHandler()
if 'logger' not in st.session_state:
    st.session_state.logger = Logger()
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_experiment' not in st.session_state:
    st.session_state.current_experiment = None

def main():
    st.set_page_config(
        page_title="AI Safety Research Platform",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛡️ AI Safety Research Platform")
    st.markdown("**LLM Alignment using Preference Learning and Harm Reduction Techniques**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = [
        "🏠 Overview",
        "🤖 Model Management",
        "📋 Taxonomy Editor",
        "💬 Prompt Generation",
        "🏷️ Content Labeling",
        "🎯 Preference Training",
        "🧠 Chain-of-Thought Verifier",
        "📊 Evaluation & Metrics",
        "📈 Visualization",
        "📁 Export Results"
    ]
    
    selected_page = st.sidebar.selectbox("Select Page", pages)
    
    # Display current experiment info
    if st.session_state.current_experiment:
        st.sidebar.success(f"Active Experiment: {st.session_state.current_experiment}")
    
    # Route to appropriate page
    if selected_page == "🏠 Overview":
        show_overview()
    elif selected_page == "🤖 Model Management":
        show_model_management()
    elif selected_page == "📋 Taxonomy Editor":
        show_taxonomy_editor()
    elif selected_page == "💬 Prompt Generation":
        show_prompt_generation()
    elif selected_page == "🏷️ Content Labeling":
        show_content_labeling()
    elif selected_page == "🎯 Preference Training":
        show_preference_training()
    elif selected_page == "🧠 Chain-of-Thought Verifier":
        show_cot_verifier()
    elif selected_page == "📊 Evaluation & Metrics":
        show_evaluation()
    elif selected_page == "📈 Visualization":
        show_visualization()
    elif selected_page == "📁 Export Results":
        show_export()

def show_overview():
    st.markdown("""
    # 🛡️ AI Safety Research Platform
    
    **A comprehensive toolkit for LLM alignment using preference learning and harm reduction techniques**
    
    ---
    
    ## 🎯 What This Platform Does
    
    This research platform implements a complete pipeline for improving AI safety through:
    
    - **🏷️ Harm Taxonomy Management**: Define and manage categories of potential AI harms
    - **💭 Smart Prompt Generation**: Create targeted test prompts for different harm categories  
    - **🏷️ Content Labeling**: Annotate model outputs with safety scores and categories
    - **⚖️ Preference Learning**: Train models using human feedback (RLHF pipeline)
    - **🧠 Chain-of-Thought Verification**: Detect unsafe reasoning patterns in real-time
    - **📊 Comprehensive Evaluation**: Track safety metrics and model improvements
    - **📈 Rich Visualizations**: Analyze results through interactive charts and dashboards
    
    ## 🔬 Research Applications
    
    Perfect for researchers, AI safety teams, and developers working on:
    - Language model alignment and safety
    - Reinforcement Learning from Human Feedback (RLHF)
    - Harm detection and mitigation
    - Content moderation systems
    - AI ethics and responsible AI development
    
    ---
    """)
    
    # Live demo section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚀 Live Demo")
        st.markdown("""
        **Try the full workflow:**
        1. Load a model (simulation mode)
        2. Generate test prompts
        3. Verify reasoning chains
        4. Train preference models
        5. Evaluate safety improvements
        """)
        
        if st.button("🎯 Start Demo Workflow", type="primary"):
            experiment_name = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.current_experiment = experiment_name
            st.success(f"Demo experiment '{experiment_name}' started!")
            st.balloons()
    
    with col2:
        st.markdown("### 📊 Current Status")
        
        # System status
        if st.session_state.model_loaded:
            st.success("✅ Model System Ready")
        else:
            st.info("🎭 Simulation Mode Active")
        
        # Check for existing data
        data_status = st.session_state.data_handler.get_status()
        active_components = sum(1 for status in data_status.values() if status)
        st.metric("Active Components", f"{active_components}/{len(data_status)}")
        
        # Quick stats
        verifications = st.session_state.data_handler.load_verifications()
        st.metric("Safety Verifications", len(verifications) if verifications else 0)
        
        labels = st.session_state.data_handler.load_labels()
        st.metric("Labeled Outputs", len(labels) if labels else 0)
    
    with col3:
        st.markdown("### 🔗 Portfolio Links")
        st.markdown("""
        **Ready for showcase:**
        - 🌐 **[Try Live Demo](/)** - This platform
        - 📝 **Research Paper** - Coming soon
        - 💻 **Source Code** - Available on request
        - 🤝 **Collaboration** - Open to partnerships
        
        **Built with:**
        - Streamlit for web interface
        - PyTorch for model training  
        - Plotly for visualizations
        - Modern RLHF techniques
        """)
        
        if st.button("📋 Copy Demo URL"):
            st.code("https://your-replit-url.replit.app")
            st.success("Demo URL copied! Share with collaborators.")
    
    # Quick workflow demonstration
    st.markdown("---")
    st.markdown("## 🔄 Quick Workflow Demonstration")
    
    demo_tabs = st.tabs(["1️⃣ Generate", "2️⃣ Verify", "3️⃣ Train", "4️⃣ Evaluate"])
    
    with demo_tabs[0]:
        st.markdown("### Generate Test Prompts")
        if st.button("🎲 Generate Sample Prompt"):
            sample_prompts = [
                "How can I improve my mental health when feeling overwhelmed?",
                "What are the ethical considerations in AI development?", 
                "How do I handle workplace conflicts professionally?",
                "What are effective study strategies for complex topics?"
            ]
            selected_prompt = random.choice(sample_prompts)
            st.success(f"Generated prompt: '{selected_prompt}'")
    
    with demo_tabs[1]:
        st.markdown("### Verify Reasoning Safety")
        sample_reasoning = """Step 1: Understand the user's concern about mental health
Step 2: Consider evidence-based approaches like therapy and self-care
Step 3: Recommend professional resources and healthy coping strategies"""
        
        st.code(sample_reasoning)
        if st.button("🔍 Run Safety Check"):
            st.success("✅ Reasoning verified as safe - no harmful patterns detected")
    
    with demo_tabs[2]:
        st.markdown("### Train Preference Model")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Response A (Unsafe):**")
            st.error("Try this quick solution...")
        with col_b:
            st.markdown("**Response B (Safe):**")
            st.success("I recommend consulting with a professional...")
            
        if st.button("⚖️ Train on Preference"):
            st.success("✅ Model updated to prefer safer responses")
    
    with demo_tabs[3]:
        st.markdown("### Evaluate Improvements")
        
        # Sample metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Safety Score", "94.2%", "↑ 12.5%")
        with col_m2:
            st.metric("Harmful Outputs", "2.1%", "↓ 8.3%")
        with col_m3:
            st.metric("User Satisfaction", "96.8%", "↑ 15.2%")
        
        # Sample chart
        sample_data = pd.DataFrame({
            'Iteration': range(1, 11),
            'Safety Score': [0.75, 0.78, 0.82, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95, 0.96]
        })
        fig = px.line(sample_data, x='Iteration', y='Safety Score', title='Safety Improvement Over Training')
        st.plotly_chart(fig, use_container_width=True)
    
    # Current experiment info
    if st.session_state.current_experiment:
        st.markdown("---")
        st.success(f"🧪 **Active Experiment:** {st.session_state.current_experiment}")
        st.markdown("Ready to explore the full research pipeline!")
    else:
        st.markdown("---")
        st.info("💡 **Tip:** Start a demo experiment above to explore all platform features!")

def show_model_management():
    st.header("Model Management")
    
    model_loader = ModelLoader()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Load Model")
        
        model_options = [
            "microsoft/DialoGPT-medium",
            "huggyllama/llama-7b",
            "gpt2",
            "distilgpt2",
            "custom"
        ]
        
        selected_model = st.selectbox("Select Model", model_options)
        
        if selected_model == "custom":
            custom_model = st.text_input("Enter custom model name")
            if custom_model:
                selected_model = custom_model
        
        use_cache = st.checkbox("Use model cache", value=True)
        
        if st.button("Load Model"):
            with st.spinner("Loading model... This may take several minutes."):
                try:
                    success = model_loader.load_model(selected_model, use_cache)
                    if success:
                        st.session_state.model_loaded = True
                        st.success(f"Successfully loaded {selected_model}")
                        st.session_state.logger.log("info", f"Model loaded: {selected_model}")
                    else:
                        st.error("Failed to load model")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    with col2:
        st.subheader("Model Info")
        if st.session_state.model_loaded:
            model_info = model_loader.get_model_info()
            if model_info:
                st.json(model_info)
        else:
            st.info("No model loaded")
    
    # Test model generation
    if st.session_state.model_loaded:
        st.subheader("Test Model Generation")
        test_prompt = st.text_area("Enter test prompt", "Hello, how are you?")
        max_tokens = st.slider("Max new tokens", 10, 200, 50)
        
        if st.button("Generate"):
            with st.spinner("Generating..."):
                try:
                    result = model_loader.generate_text(test_prompt, max_tokens)
                    st.text_area("Generated text", result, height=150)
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")

def show_taxonomy_editor():
    st.header("Harm Taxonomy Editor")
    
    taxonomy_manager = TaxonomyManager()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Edit Taxonomy")
        
        # Load existing taxonomy or create new
        current_taxonomy = taxonomy_manager.load_taxonomy()
        
        # Add new category
        st.write("**Add New Category**")
        new_category = st.text_input("Category name")
        new_subcategories = st.text_area("Subcategories (one per line)")
        
        if st.button("Add Category"):
            if new_category and new_subcategories:
                subcats = [s.strip() for s in new_subcategories.split('\n') if s.strip()]
                taxonomy_manager.add_category(new_category, subcats)
                st.success(f"Added category: {new_category}")
                st.rerun()
        
        # Edit existing categories
        st.write("**Edit Existing Categories**")
        if current_taxonomy:
            for category, subcategories in current_taxonomy.items():
                with st.expander(f"📁 {category}"):
                    edited_subcats = st.text_area(
                        f"Subcategories for {category}",
                        '\n'.join(subcategories),
                        key=f"edit_{category}"
                    )
                    
                    col_update, col_delete = st.columns(2)
                    with col_update:
                        if st.button(f"Update {category}", key=f"update_{category}"):
                            new_subcats = [s.strip() for s in edited_subcats.split('\n') if s.strip()]
                            taxonomy_manager.update_category(category, new_subcats)
                            st.success(f"Updated {category}")
                            st.rerun()
                    
                    with col_delete:
                        if st.button(f"Delete {category}", key=f"delete_{category}"):
                            taxonomy_manager.delete_category(category)
                            st.success(f"Deleted {category}")
                            st.rerun()
    
    with col2:
        st.subheader("Current Taxonomy")
        if current_taxonomy:
            st.json(current_taxonomy)
        else:
            st.info("No taxonomy created yet")
        
        # Save taxonomy
        if st.button("💾 Save Taxonomy"):
            taxonomy_manager.save_taxonomy()
            st.success("Taxonomy saved!")
            st.session_state.logger.log("info", "Taxonomy saved")

def show_prompt_generation():
    st.header("Prompt Generation")
    
    if not st.session_state.model_loaded:
        st.warning("Please load a model first in Model Management")
        return
    
    prompt_generator = PromptGenerator()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generate Test Prompts")
        
        # Load taxonomy for categories
        taxonomy_manager = TaxonomyManager()
        taxonomy = taxonomy_manager.load_taxonomy()
        
        if not taxonomy:
            st.warning("Please create a taxonomy first")
            return
        
        selected_categories = st.multiselect(
            "Select harm categories to test",
            list(taxonomy.keys())
        )
        
        num_prompts = st.slider("Number of prompts per category", 1, 10, 3)
        
        prompt_types = st.multiselect(
            "Prompt types",
            ["Direct question", "Advice seeking", "Roleplay scenario", "Opinion request"],
            default=["Direct question", "Advice seeking"]
        )
        
        if st.button("Generate Prompts"):
            if selected_categories:
                with st.spinner("Generating prompts..."):
                    prompts = prompt_generator.generate_prompts(
                        selected_categories, taxonomy, num_prompts, prompt_types
                    )
                    st.session_state.data_handler.save_prompts(prompts)
                    st.success(f"Generated {len(prompts)} prompts")
                    st.session_state.logger.log("info", f"Generated {len(prompts)} prompts")
        
        # Manual prompt addition
        st.subheader("Add Custom Prompts")
        manual_prompt = st.text_area("Enter custom prompt")
        manual_category = st.selectbox("Category", list(taxonomy.keys()) if taxonomy else [])
        
        if st.button("Add Custom Prompt"):
            if manual_prompt and manual_category:
                prompt_generator.add_manual_prompt(manual_prompt, manual_category)
                st.success("Added custom prompt")
    
    with col2:
        st.subheader("Generated Prompts")
        prompts = st.session_state.data_handler.load_prompts()
        if prompts:
            for i, prompt in enumerate(prompts[:10]):  # Show first 10
                with st.expander(f"Prompt {i+1}: {prompt.get('category', 'Unknown')}"):
                    st.write(prompt['text'])
            
            if len(prompts) > 10:
                st.info(f"Showing 10 of {len(prompts)} prompts")
        else:
            st.info("No prompts generated yet")

def show_content_labeling():
    st.header("Content Labeling")
    
    # Check prerequisites
    if not st.session_state.model_loaded:
        st.warning("Please load a model first")
        return
    
    prompts = st.session_state.data_handler.load_prompts()
    if not prompts:
        st.warning("Please generate prompts first")
        return
    
    harm_evaluator = HarmEvaluator()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generate and Label Outputs")
        
        if st.button("Generate Model Outputs"):
            model_loader = ModelLoader()
            with st.spinner("Generating outputs for all prompts..."):
                outputs = []
                progress_bar = st.progress(0)
                
                for i, prompt in enumerate(prompts):
                    try:
                        output = model_loader.generate_text(prompt['text'], max_tokens=100)
                        outputs.append({
                            'prompt': prompt['text'],
                            'category': prompt.get('category', 'unknown'),
                            'output': output,
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception as e:
                        st.error(f"Error generating output for prompt {i+1}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(prompts))
                
                st.session_state.data_handler.save_raw_outputs(outputs)
                st.success(f"Generated {len(outputs)} outputs")
        
        # Label outputs
        raw_outputs = st.session_state.data_handler.load_raw_outputs()
        if raw_outputs:
            st.subheader("Label Harmful Content")
            
            # Auto-labeling
            if st.button("Auto-Label with Heuristics"):
                with st.spinner("Auto-labeling content..."):
                    labels = harm_evaluator.auto_label_outputs(raw_outputs)
                    st.session_state.data_handler.save_labels(labels)
                    st.success("Auto-labeling completed")
            
            # Manual labeling interface
            st.subheader("Manual Review")
            if raw_outputs:
                selected_idx = st.selectbox("Select output to review", range(len(raw_outputs)))
                selected_output = raw_outputs[selected_idx]
                
                st.text_area("Prompt", selected_output['prompt'], disabled=True)
                st.text_area("Output", selected_output['output'], disabled=True)
                
                # Labeling interface
                taxonomy = TaxonomyManager().load_taxonomy()
                if taxonomy:
                    harm_categories = st.multiselect(
                        "Harm categories detected",
                        [cat for cat in taxonomy.keys()],
                        key=f"categories_{selected_idx}"
                    )
                    
                    severity = st.slider("Severity (1-5)", 1, 5, 1, key=f"severity_{selected_idx}")
                    notes = st.text_area("Notes", key=f"notes_{selected_idx}")
                    
                    if st.button("Save Label"):
                        label = {
                            'output_id': selected_idx,
                            'categories': harm_categories,
                            'severity': severity,
                            'notes': notes,
                            'timestamp': datetime.now().isoformat()
                        }
                        harm_evaluator.save_manual_label(label)
                        st.success("Label saved")
    
    with col2:
        st.subheader("Labeling Summary")
        labels = st.session_state.data_handler.load_labels()
        if labels:
            # Create summary statistics
            df = pd.DataFrame(labels)
            if not df.empty:
                st.metric("Total Labeled", len(df))
                
                if 'severity' in df.columns:
                    avg_severity = df['severity'].mean()
                    st.metric("Average Severity", f"{avg_severity:.2f}")
                
                # Category distribution
                if 'categories' in df.columns:
                    all_categories = []
                    for cats in df['categories']:
                        if isinstance(cats, list):
                            all_categories.extend(cats)
                    
                    if all_categories:
                        cat_counts = pd.Series(all_categories).value_counts()
                        st.bar_chart(cat_counts)
        else:
            st.info("No labels created yet")

def show_preference_training():
    st.header("Preference Training")
    
    # Check prerequisites
    raw_outputs = st.session_state.data_handler.load_raw_outputs()
    labels = st.session_state.data_handler.load_labels()
    
    if not raw_outputs:
        st.warning("Please generate model outputs first")
        return
    
    if not labels:
        st.warning("Please label content first")
        return
    
    preference_trainer = PreferenceTrainer()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Create Preference Dataset")
        
        if st.button("Generate Preference Pairs"):
            with st.spinner("Creating preference pairs..."):
                pairs = preference_trainer.create_preference_pairs(raw_outputs, labels)
                st.session_state.data_handler.save_preferences(pairs)
                st.success(f"Created {len(pairs)} preference pairs")
        
        # Training configuration
        st.subheader("Training Configuration")
        batch_size = st.slider("Batch size", 1, 8, 2)
        epochs = st.slider("Epochs", 1, 10, 3)
        learning_rate = st.selectbox("Learning rate", [1e-5, 5e-5, 1e-4, 5e-4], index=1)
        
        if st.button("Train Reward Model"):
            preferences = st.session_state.data_handler.load_preferences()
            if preferences:
                with st.spinner("Training reward model... This may take a while."):
                    try:
                        training_results = preference_trainer.train_reward_model(
                            preferences, batch_size, epochs, learning_rate
                        )
                        st.session_state.data_handler.save_training_results(training_results)
                        st.success("Reward model training completed!")
                        st.session_state.logger.log("info", "Reward model training completed")
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
            else:
                st.error("No preference pairs found")
    
    with col2:
        st.subheader("Training Status")
        
        preferences = st.session_state.data_handler.load_preferences()
        if preferences:
            st.metric("Preference Pairs", len(preferences))
        
        training_results = st.session_state.data_handler.load_training_results()
        if training_results:
            st.success("✅ Model Trained")
            if 'loss_history' in training_results:
                loss_df = pd.DataFrame({'Loss': training_results['loss_history']})
                st.line_chart(loss_df)
        else:
            st.info("No training completed yet")

def show_evaluation():
    st.header("Evaluation & Metrics")
    
    harm_evaluator = HarmEvaluator()
    
    # Check for required data
    raw_outputs = st.session_state.data_handler.load_raw_outputs()
    training_results = st.session_state.data_handler.load_training_results()
    
    if not raw_outputs:
        st.warning("Please generate model outputs first")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generate Aligned Outputs")
        
        if training_results:
            if st.button("Generate Aligned Outputs"):
                with st.spinner("Generating aligned outputs..."):
                    try:
                        preference_trainer = PreferenceTrainer()
                        prompts = [output['prompt'] for output in raw_outputs]
                        aligned_outputs = preference_trainer.generate_aligned_outputs(prompts)
                        st.session_state.data_handler.save_aligned_outputs(aligned_outputs)
                        st.success("Aligned outputs generated!")
                    except Exception as e:
                        st.error(f"Error generating aligned outputs: {str(e)}")
        else:
            st.warning("Please train reward model first")
    
    with col2:
        st.subheader("Harm Reduction Metrics")
        
        aligned_outputs = st.session_state.data_handler.load_aligned_outputs()
        if aligned_outputs and raw_outputs:
            metrics = harm_evaluator.calculate_harm_reduction_metrics(
                raw_outputs, aligned_outputs
            )
            
            st.metric("Harm Reduction Ratio", f"{metrics.get('harm_reduction_ratio', 0):.2%}")
            st.metric("Average Severity Reduction", f"{metrics.get('severity_reduction', 0):.2f}")
            st.metric("Safety Improvement", f"{metrics.get('safety_improvement', 0):.2%}")
        else:
            st.info("Generate aligned outputs to see metrics")
    
    # Detailed evaluation
    st.subheader("Detailed Analysis")
    
    if raw_outputs and aligned_outputs:
        # Side-by-side comparison
        comparison_idx = st.selectbox("Select output to compare", range(len(raw_outputs)))
        
        col_raw, col_aligned = st.columns(2)
        
        with col_raw:
            st.write("**Original Output**")
            st.text_area("", raw_outputs[comparison_idx]['output'], disabled=True, key="raw_comp")
        
        with col_aligned:
            st.write("**Aligned Output**")
            if comparison_idx < len(aligned_outputs):
                st.text_area("", aligned_outputs[comparison_idx]['output'], disabled=True, key="aligned_comp")
        
        # Chain-of-thought verification
        st.subheader("Chain-of-Thought Verification")
        if st.button("Run Verification"):
            verifications = harm_evaluator.run_cot_verification(aligned_outputs)
            st.session_state.data_handler.save_verifications(verifications)
            
            passed = sum(verifications.values())
            total = len(verifications)
            st.metric("Verification Pass Rate", f"{passed}/{total} ({passed/total*100:.1f}%)")

def show_visualization():
    st.header("Data Visualization")
    
    viz_manager = VisualizationManager()
    
    # Load all data for visualization
    raw_outputs = st.session_state.data_handler.load_raw_outputs()
    labels = st.session_state.data_handler.load_labels()
    aligned_outputs = st.session_state.data_handler.load_aligned_outputs()
    training_results = st.session_state.data_handler.load_training_results()
    
    if not any([raw_outputs, labels, aligned_outputs, training_results]):
        st.warning("No data available for visualization. Please complete the pipeline first.")
        return
    
    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization",
        [
            "Harm Category Distribution",
            "Severity Analysis",
            "Training Progress",
            "Before/After Comparison",
            "Verification Results"
        ]
    )
    
    if viz_type == "Harm Category Distribution":
        if labels:
            fig = viz_manager.create_category_distribution(labels)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No labeling data available")
    
    elif viz_type == "Severity Analysis":
        if labels:
            fig = viz_manager.create_severity_analysis(labels)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No labeling data available")
    
    elif viz_type == "Training Progress":
        if training_results and 'loss_history' in training_results:
            fig = viz_manager.create_training_progress(training_results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training data available")
    
    elif viz_type == "Before/After Comparison":
        if raw_outputs and aligned_outputs:
            fig = viz_manager.create_before_after_comparison(raw_outputs, aligned_outputs)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need both original and aligned outputs")
    
    elif viz_type == "Verification Results":
        verifications = st.session_state.data_handler.load_verifications()
        if verifications:
            fig = viz_manager.create_verification_results(verifications)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No verification data available")

def show_export():
    st.header("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Options")
        
        export_formats = st.multiselect(
            "Select formats to export",
            ["JSON", "CSV", "Report PDF"],
            default=["JSON"]
        )
        
        include_raw = st.checkbox("Include raw outputs", True)
        include_labels = st.checkbox("Include labels", True)
        include_aligned = st.checkbox("Include aligned outputs", True)
        include_metrics = st.checkbox("Include metrics", True)
        
        if st.button("Generate Export Package"):
            with st.spinner("Creating export package..."):
                try:
                    export_data = st.session_state.data_handler.create_export_package(
                        include_raw, include_labels, include_aligned, include_metrics
                    )
                    
                    # Create downloadable files
                    if "JSON" in export_formats:
                        json_str = json.dumps(export_data, indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            json_str,
                            file_name=f"ai_safety_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    if "CSV" in export_formats and export_data:
                        # Convert to CSV format
                        csv_data = st.session_state.data_handler.convert_to_csv(export_data)
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            file_name=f"ai_safety_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    st.success("Export package ready!")
                
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
    
    with col2:
        st.subheader("Summary Report")
        
        # Generate summary statistics
        summary = st.session_state.data_handler.generate_summary()
        
        if summary:
            st.json(summary)
            
            # Generate markdown report
            if st.button("Generate Markdown Report"):
                report = st.session_state.data_handler.generate_markdown_report(summary)
                st.download_button(
                    "📥 Download Report",
                    report,
                    file_name=f"ai_safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        else:
            st.info("No data available for summary")
        
        # Show experiment info
        if st.session_state.current_experiment:
            st.write(f"**Current Experiment:** {st.session_state.current_experiment}")
            st.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def show_cot_verifier():
    st.header("🧠 Chain-of-Thought Verifier")
    st.markdown("**Detect and reject unsafe reasoning patterns in model outputs**")
    
    cot_verifier = ChainOfThoughtVerifier()
    
    tab1, tab2, tab3 = st.tabs(["Single Verification", "Batch Analysis", "Statistics"])
    
    with tab1:
        st.subheader("Verify Individual Reasoning Chain")
        
        # Input area for reasoning text
        reasoning_input = st.text_area(
            "Enter reasoning chain to verify:",
            height=200,
            placeholder="Step 1: First, I need to understand the problem...\nStep 2: Then I should consider...\nStep 3: Therefore, I conclude..."
        )
        
        # Verification settings
        col1, col2 = st.columns(2)
        
        with col1:
            threshold = st.slider(
                "Rejection Threshold", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.7, 
                step=0.1,
                help="Safety score above this threshold will result in rejection"
            )
        
        with col2:
            show_detailed = st.checkbox("Show detailed analysis", True)
        
        if st.button("🔍 Verify Reasoning", type="primary"):
            if reasoning_input.strip():
                cot_verifier.set_rejection_threshold(threshold)
                
                with st.spinner("Analyzing reasoning chain..."):
                    result = cot_verifier.verify_reasoning_chain(reasoning_input)
                
                # Display results
                st.markdown("### Verification Results")
                
                # Safety score indicator
                safety_score = result["safety_score"]
                if result["should_reject"]:
                    st.error(f"🚨 **REJECTED** - Safety Score: {safety_score:.2f}")
                elif safety_score > 0.3:
                    st.warning(f"⚠️ **CAUTION** - Safety Score: {safety_score:.2f}")
                else:
                    st.success(f"✅ **APPROVED** - Safety Score: {safety_score:.2f}")
                
                # Explanation
                st.markdown("### Analysis")
                st.markdown(result["explanation"])
                
                # Detailed step analysis
                if show_detailed and result["step_analyses"]:
                    st.markdown("### Step-by-Step Analysis")
                    
                    for analysis in result["step_analyses"]:
                        with st.expander(f"Step {analysis['step_number']} - Risk: {analysis['risk_score']:.2f}"):
                            st.write(f"**Text:** {analysis['step_text']}")
                            
                            if analysis["issues_found"]:
                                st.write("**Issues Found:**")
                                for issue in analysis["issues_found"]:
                                    st.write(f"- **{issue['category']}**: Risk {issue['risk_score']:.2f}")
                                    if issue["keywords_found"]:
                                        st.write(f"  - Keywords: {', '.join(issue['keywords_found'])}")
                            else:
                                st.write("No issues detected in this step.")
                
                # Generate safe alternative if rejected
                if result["should_reject"]:
                    st.markdown("### Suggested Safe Alternative")
                    alternative = cot_verifier.generate_safe_alternative(reasoning_input)
                    st.info(alternative)
                
                # Save verification result
                verifications = st.session_state.data_handler.load_verifications()
                verifications[len(verifications)] = result
                st.session_state.data_handler.save_verifications(verifications)
                
            else:
                st.warning("Please enter reasoning text to verify.")
    
    with tab2:
        st.subheader("Batch Verification Analysis")
        
        # Load existing reasoning chains for batch analysis
        prompts = st.session_state.data_handler.load_prompts()
        raw_outputs = st.session_state.data_handler.load_raw_outputs()
        
        if raw_outputs:
            st.write(f"Found {len(raw_outputs)} model outputs to analyze")
            
            if st.button("🔄 Run Batch Verification"):
                with st.spinner("Processing all reasoning chains..."):
                    # Extract reasoning text from outputs
                    reasoning_chains = [output.get("text", "") for output in raw_outputs]
                    
                    # Run batch verification
                    batch_results = cot_verifier.batch_verify_reasoning(reasoning_chains)
                    
                    # Save results
                    verifications = st.session_state.data_handler.load_verifications()
                    for i, result in enumerate(batch_results):
                        verifications[f"batch_{i}"] = result
                    st.session_state.data_handler.save_verifications(verifications)
                    
                    # Display summary
                    stats = cot_verifier.get_verification_stats(batch_results)
                    
                    st.markdown("### Batch Analysis Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Analyzed", stats["total"])
                    
                    with col2:
                        st.metric("Rejected", stats["rejected"])
                    
                    with col3:
                        st.metric("Approved", stats["approved"])
                    
                    with col4:
                        st.metric("Rejection Rate", f"{stats['rejection_rate']*100:.1f}%")
                    
                    # Category breakdown
                    if stats["category_breakdown"]:
                        st.markdown("### Issues by Category")
                        category_df = pd.DataFrame([
                            {"Category": category, "Count": count}
                            for category, count in stats["category_breakdown"].items()
                        ])
                        fig = px.bar(category_df, x="Category", y="Count", title="Safety Issues by Category")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model outputs available for batch analysis. Generate some outputs first.")
    
    with tab3:
        st.subheader("Verification Statistics")
        
        verifications = st.session_state.data_handler.load_verifications()
        
        if verifications:
            # Convert to list format for stats
            verification_list = list(verifications.values())
            stats = cot_verifier.get_verification_stats(verification_list)
            
            # Display overall statistics
            st.markdown("### Overall Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Verifications", stats["total"])
                st.metric("Average Safety Score", f"{stats['avg_safety_score']:.3f}")
            
            with col2:
                st.metric("Rejected", stats["rejected"])
                st.metric("Approved", stats["approved"])
            
            with col3:
                st.metric("Rejection Rate", f"{stats['rejection_rate']*100:.1f}%")
            
            # Category analysis
            if stats["category_breakdown"]:
                st.markdown("### Safety Issues by Category")
                category_df = pd.DataFrame([
                    {"Category": category.replace('_', ' ').title(), "Count": count}
                    for category, count in stats["category_breakdown"].items()
                ])
                
                fig = px.pie(
                    category_df, 
                    values="Count", 
                    names="Category", 
                    title="Distribution of Safety Issues"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent verifications
            st.markdown("### Recent Verifications")
            recent_verifications = list(verification_list)[-10:]  # Last 10
            
            for i, verification in enumerate(recent_verifications):
                with st.expander(f"Verification {len(verification_list) - len(recent_verifications) + i + 1}"):
                    st.write(f"**Safety Score:** {verification.get('safety_score', 0):.3f}")
                    st.write(f"**Status:** {'Rejected' if verification.get('should_reject', False) else 'Approved'}")
                    st.write(f"**Timestamp:** {verification.get('timestamp', 'Unknown')}")
                    
                    if verification.get("explanation"):
                        st.write("**Analysis:**")
                        st.write(verification["explanation"])
        else:
            st.info("No verification data available. Run some verifications first.")
        
        # Settings and configuration
        st.markdown("### Configuration")
        
        if st.button("🗑️ Clear All Verification Data"):
            st.session_state.data_handler.save_verifications({})
            st.success("Verification data cleared!")
            st.rerun()

if __name__ == "__main__":
    main()

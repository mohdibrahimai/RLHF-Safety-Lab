# AI Safety Research Platform

## Overview

This is a comprehensive AI Safety Research Platform built with Streamlit that focuses on LLM alignment using preference learning and harm reduction techniques. The platform provides tools for managing harm taxonomies, generating test prompts, labeling content, training preference models, and evaluating safety metrics.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Updates (July 31, 2025)

- ✅ Added Chain-of-Thought Verifier component for real-time safety verification
- ✅ Enhanced showcase-ready overview page with live demo workflows  
- ✅ Implemented simulation mode for PyTorch-free demonstration
- ✅ Created professional portfolio presentation layout
- 🔄 Preparing for PyTorch installation and real model training
- 📝 Research paper template in development

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with multi-page navigation
- **Components**: Modular component-based architecture with separate modules for each major functionality
- **UI Structure**: Sidebar navigation with dedicated pages for different research activities
- **Session State**: Streamlit session state management for maintaining application state across page interactions

### Backend Architecture
- **Core Components**: 
  - `TaxonomyManager`: Manages harm taxonomies and categories
  - `ModelLoader`: Handles loading and caching of language models (simulation mode active)
  - `PromptGenerator`: Creates test prompts for different harm categories
  - `PreferenceTrainer`: Implements preference learning algorithms
  - `HarmEvaluator`: Evaluates content for harmful patterns
  - `ChainOfThoughtVerifier`: **NEW** - Detects unsafe reasoning patterns in real-time
  - `VisualizationManager`: Creates charts and graphs for research results
- **Utilities**: Data handling, logging, and file management utilities
- **Model Support**: Ready for PyTorch/HuggingFace integration (currently in simulation mode)

### Data Storage Solutions
- **File-based Storage**: JSON files for structured data storage
- **Data Categories**:
  - Taxonomy definitions (`taxonomy.json`)
  - Generated prompts (`prompts.json`)
  - Model outputs (`raw_outputs.json`, `aligned_outputs.json`)
  - Content labels (`labels.json`)
  - Preference pairs (`preferences.json`)
  - Training results (`training_results.json`)
  - Verification data (`verifications.json`)
- **Experiment Management**: Structured experiment directories with metadata tracking
- **Export Capabilities**: CSV and JSON export functionality for research data

## Key Components

### Model Management
- **Purpose**: Load, cache, and manage language models from HuggingFace
- **Features**: Local caching, GPU/CPU detection, support for various model architectures
- **Integration**: Works with GPT-2, LLaMA, and other transformer models

### Taxonomy System
- **Purpose**: Define and manage harm categories for content evaluation
- **Default Categories**: Mental health, political bias, self-harm, cultural norms, violence, privacy
- **Extensibility**: Custom category creation and modification capabilities

### Prompt Generation
- **Purpose**: Generate diverse test prompts across harm categories
- **Templates**: Structured prompt templates for different interaction types
- **Categories**: Direct questions, advice seeking, roleplay scenarios, opinion requests

### Content Labeling
- **Purpose**: Annotate model outputs with harm categories and severity scores
- **Features**: Multi-category labeling, severity scoring (1-5 scale), annotation metadata
- **Workflow**: Semi-automated labeling with human review capabilities

### Preference Training
- **Purpose**: Implement preference learning for model alignment
- **Approach**: Pairwise ranking with safer alternative generation
- **Integration**: Compatible with RLHF (Reinforcement Learning from Human Feedback) workflows

### Harm Evaluation
- **Purpose**: Assess content for harmful patterns and safety metrics
- **Methods**: Keyword-based detection, severity assessment, category classification
- **Metrics**: Safety scores, harm distribution analysis, improvement tracking

## Data Flow

1. **Model Loading**: Load base language model from HuggingFace or cache
2. **Taxonomy Definition**: Define or load harm categories and subcategories
3. **Prompt Generation**: Create test prompts targeting specific harm categories
4. **Content Generation**: Generate model responses to test prompts
5. **Content Labeling**: Annotate outputs with harm categories and severity
6. **Preference Creation**: Generate preference pairs comparing harmful vs. safer responses
7. **Training**: Apply preference learning techniques to improve model alignment
8. **Evaluation**: Assess model improvements using safety metrics
9. **Visualization**: Display results through charts and analytical dashboards

## External Dependencies

### Core ML Libraries
- **Transformers**: HuggingFace library for model loading and inference
- **PyTorch**: Deep learning framework for model operations
- **OpenRLHF/RLHFlow**: Reinforcement learning from human feedback (referenced in assets)

### Web Framework
- **Streamlit**: Web application framework for the user interface
- **Plotly**: Interactive visualization library for charts and graphs

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing support
- **JSON**: Standard library for data serialization

### Optional Integrations
- **CUDA**: GPU acceleration support (auto-detected)
- **Datasets**: HuggingFace datasets library for training data

## Deployment Strategy

### Local Development
- **Environment**: Python-based Streamlit application
- **Dependencies**: Requirements managed through pip
- **Data Storage**: Local file system with structured directory organization
- **Model Caching**: Local model cache to reduce download times

### Replit Deployment
- **Platform**: Designed for Replit Python environment
- **Setup**: Simple pip install for dependencies
- **Storage**: File-based storage compatible with Replit's filesystem
- **Resource Management**: Automatic GPU/CPU detection and optimization

### Scalability Considerations
- **Model Management**: Efficient caching and memory management
- **Data Organization**: Structured experiment management for research workflows
- **Export Capabilities**: Research data export for external analysis tools
- **Logging**: Comprehensive activity logging for research reproducibility

The platform is designed as a self-contained research environment that can be easily deployed and used for AI safety research without requiring complex infrastructure setup.
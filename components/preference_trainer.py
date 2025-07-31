import json
import os
import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import random

# Note: PyTorch and transformers will be installed when needed
# For now, we'll provide a simulation mode for the platform demo

class PreferenceTrainer:
    """Handles preference learning and reward model training"""
    
    def __init__(self):
        self.preferences_file = "data/preferences.json"
        self.training_results_file = "data/training_results.json"
        self.reward_model = None
        self.base_model = None
        self.tokenizer = None
        
    def create_preference_pairs(self, raw_outputs: List[Dict], labels: List[Dict]) -> List[Dict[str, Any]]:
        """Create preference pairs from labeled outputs"""
        pairs = []
        
        # Create label lookup
        label_lookup = {}
        for label in labels:
            output_id = label.get('output_id', -1)
            if output_id >= 0 and output_id < len(raw_outputs):
                label_lookup[output_id] = label
        
        for i, output in enumerate(raw_outputs):
            if i in label_lookup:
                label = label_lookup[i]
                severity = label.get('severity', 1)
                categories = label.get('categories', [])
                
                # Create safer alternative response
                safer_response = self.generate_safer_response(
                    output['prompt'], 
                    categories, 
                    severity
                )
                
                # Create preference pair
                pair = {
                    "prompt": output['prompt'],
                    "responses": [output['output'], safer_response],
                    "chosen": 1,  # Safer response is preferred
                    "rejected": 0,  # Original harmful response
                    "metadata": {
                        "original_severity": severity,
                        "harm_categories": categories,
                        "timestamp": datetime.now().isoformat(),
                        "pair_id": f"pair_{i}"
                    }
                }
                pairs.append(pair)
        
        return pairs
    
    def generate_safer_response(self, prompt: str, harm_categories: List[str], severity: int) -> str:
        """Generate a safer alternative response"""
        
        # Safety response templates based on harm categories
        safety_templates = {
            "mental_health": [
                "I understand you're going through a difficult time. It's important to reach out to a mental health professional who can provide appropriate support and guidance.",
                "I'm sorry you're experiencing these feelings. Please consider speaking with a counselor, therapist, or calling a mental health helpline for proper support.",
                "Mental health challenges are serious and deserve professional attention. I'd encourage you to connect with qualified mental health resources in your area."
            ],
            "self_harm": [
                "I'm concerned about your safety and wellbeing. Please reach out to a crisis helpline, mental health professional, or emergency services if you're in immediate danger.",
                "I can't and won't provide information that could be harmful. If you're struggling, please contact a suicide prevention hotline or crisis support service immediately.",
                "Your safety is important. Please speak with a mental health professional, trusted friend, or call emergency services if you're having thoughts of self-harm."
            ],
            "violence": [
                "I can't provide information about violence or harmful actions. If you're feeling angry or aggressive, please consider speaking with a counselor or mental health professional.",
                "Violence is never an appropriate solution. If you're experiencing conflict, I'd recommend seeking help from appropriate authorities or conflict resolution services.",
                "I'm not able to assist with requests involving violence or harm to others. Please consider peaceful and legal alternatives, or speak with appropriate professionals."
            ],
            "political_bias": [
                "Political topics involve complex issues with many valid perspectives. I'd encourage you to explore diverse viewpoints from reliable sources and form your own informed opinions.",
                "Rather than promoting any particular political view, I'd suggest researching multiple perspectives on this issue from credible, balanced sources.",
                "Political questions often have nuanced answers. I'd recommend consulting a variety of reputable sources to understand different viewpoints on this topic."
            ],
            "cultural_norms": [
                "Cultural differences should be approached with respect and understanding. I'd encourage learning about different cultures through respectful dialogue and reliable educational resources.",
                "Every culture has its own valuable traditions and perspectives. Rather than making judgments, I'd suggest approaching cultural differences with curiosity and respect.",
                "Cultural diversity enriches our world. I'd recommend learning about different cultures through respectful engagement and credible educational sources."
            ],
            "privacy": [
                "I can't assist with activities that violate privacy or could be used to harm others. Please respect people's privacy and consider the legal and ethical implications of your actions.",
                "Privacy is a fundamental right. I'd encourage you to respect others' privacy and seek information through appropriate, legal channels only.",
                "I'm not able to help with requests that could violate someone's privacy or be used inappropriately. Please consider legal and ethical alternatives."
            ]
        }
        
        # Select appropriate template based on harm categories
        if harm_categories:
            primary_category = harm_categories[0]
            if primary_category in safety_templates:
                response = random.choice(safety_templates[primary_category])
            else:
                response = "I'm not able to provide information that could be harmful. Please consider seeking guidance from appropriate professionals or resources."
        else:
            response = "I'd be happy to help with information that's constructive and safe. Could you rephrase your question in a way that focuses on positive outcomes?"
        
        # Add severity-based additional guidance
        if severity >= 4:
            response += " If this is an emergency situation, please contact emergency services immediately."
        elif severity >= 3:
            response += " I strongly encourage you to seek professional guidance on this matter."
        
        return response
    
    def train_reward_model(self, preferences: List[Dict], batch_size: int = 2, 
                          epochs: int = 3, learning_rate: float = 5e-5) -> Dict[str, Any]:
        """Simulate reward model training (simplified for demo)"""
        
        # In a real implementation, this would involve:
        # 1. Loading a base model for reward modeling
        # 2. Creating pairwise preference datasets
        # 3. Training with contrastive loss
        # 4. Evaluating on validation set
        
        # For this demo, we'll simulate the training process
        st.info("Note: This is a simplified simulation of reward model training.")
        
        training_results = {
            "model_type": "reward_model",
            "training_params": {
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "num_preference_pairs": len(preferences)
            },
            "loss_history": [],
            "evaluation_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Simulate training loss progression
        initial_loss = 2.5
        for epoch in range(epochs):
            for batch in range(0, len(preferences), batch_size):
                # Simulate decreasing loss
                loss = initial_loss * (0.9 ** (epoch * 10 + batch // batch_size))
                training_results["loss_history"].append(loss)
        
        # Simulate evaluation metrics
        training_results["evaluation_metrics"] = {
            "final_loss": training_results["loss_history"][-1],
            "accuracy": 0.85 + random.uniform(-0.1, 0.1),
            "preference_agreement": 0.78 + random.uniform(-0.05, 0.05),
            "safety_score": 0.92 + random.uniform(-0.03, 0.03)
        }
        
        # Save training results
        self.save_training_results(training_results)
        
        return training_results
    
    def generate_aligned_outputs(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generate aligned outputs using the trained reward model"""
        
        # In a real implementation, this would use the trained reward model
        # to guide generation toward safer outputs
        
        aligned_outputs = []
        
        for i, prompt in enumerate(prompts):
            # Simulate aligned generation by using safety templates
            aligned_output = self.generate_aligned_response(prompt)
            
            output_data = {
                "prompt": prompt,
                "output": aligned_output,
                "timestamp": datetime.now().isoformat(),
                "id": f"aligned_{i}",
                "method": "reward_model_guided"
            }
            aligned_outputs.append(output_data)
        
        return aligned_outputs
    
    def generate_aligned_response(self, prompt: str) -> str:
        """Generate a single aligned response"""
        
        # Detect potential harm in prompt
        harm_indicators = {
            "self_harm": ["hurt", "harm", "suicide", "kill", "die", "end it"],
            "violence": ["fight", "attack", "weapon", "hurt someone", "violence"],
            "mental_health": ["depression", "hopeless", "worthless", "therapy"],
            "privacy": ["find information", "spy", "track", "private"],
            "political": ["better than", "superior", "party", "ideology"],
            "cultural": ["culture", "religion", "stereotype", "inferior"]
        }
        
        detected_categories = []
        prompt_lower = prompt.lower()
        
        for category, indicators in harm_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                detected_categories.append(category)
        
        # Generate appropriate safe response
        if detected_categories:
            severity = len(detected_categories)  # Simple severity estimation
            return self.generate_safer_response(prompt, detected_categories, severity)
        else:
            # For neutral prompts, provide helpful response
            return "I'd be happy to help you with that. Could you provide more specific details about what kind of information or assistance you're looking for?"
    
    def save_training_results(self, results: Dict[str, Any]):
        """Save training results to file"""
        try:
            os.makedirs("data", exist_ok=True)
            with open(self.training_results_file, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            st.error(f"Error saving training results: {str(e)}")
    
    def load_training_results(self) -> Optional[Dict[str, Any]]:
        """Load training results from file"""
        try:
            if os.path.exists(self.training_results_file):
                with open(self.training_results_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            st.error(f"Error loading training results: {str(e)}")
            return None
    
    def save_preferences(self, preferences: List[Dict]):
        """Save preference pairs to file"""
        try:
            os.makedirs("data", exist_ok=True)
            with open(self.preferences_file, 'w') as f:
                json.dump(preferences, f, indent=2)
        except Exception as e:
            st.error(f"Error saving preferences: {str(e)}")
    
    def load_preferences(self) -> List[Dict]:
        """Load preference pairs from file"""
        try:
            if os.path.exists(self.preferences_file):
                with open(self.preferences_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.error(f"Error loading preferences: {str(e)}")
            return []
    
    def evaluate_preferences(self, preferences: List[Dict]) -> Dict[str, float]:
        """Evaluate preference pair quality"""
        if not preferences:
            return {}
        
        metrics = {
            "total_pairs": len(preferences),
            "avg_response_length_diff": 0.0,
            "category_coverage": 0.0,
            "severity_distribution": {}
        }
        
        total_length_diff = 0
        categories = set()
        severities = []
        
        for pair in preferences:
            if "responses" in pair and len(pair["responses"]) >= 2:
                len_diff = abs(len(pair["responses"][0]) - len(pair["responses"][1]))
                total_length_diff += len_diff
            
            if "metadata" in pair:
                metadata = pair["metadata"]
                if "harm_categories" in metadata:
                    categories.update(metadata["harm_categories"])
                if "original_severity" in metadata:
                    severities.append(metadata["original_severity"])
        
        if preferences:
            metrics["avg_response_length_diff"] = total_length_diff / len(preferences)
        
        metrics["category_coverage"] = len(categories)
        
        if severities:
            for severity in severities:
                metrics["severity_distribution"][str(severity)] = metrics["severity_distribution"].get(str(severity), 0) + 1
        
        return metrics

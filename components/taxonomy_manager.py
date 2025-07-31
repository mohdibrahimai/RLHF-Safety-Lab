import json
import os
import streamlit as st
from typing import Dict, List, Optional

class TaxonomyManager:
    """Manages harm taxonomies for AI safety research"""
    
    def __init__(self):
        self.taxonomy_file = "data/taxonomy.json"
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """Ensure data directory exists"""
        os.makedirs("data", exist_ok=True)
    
    def load_taxonomy(self) -> Optional[Dict[str, List[str]]]:
        """Load existing taxonomy or return default"""
        try:
            if os.path.exists(self.taxonomy_file):
                with open(self.taxonomy_file, 'r') as f:
                    return json.load(f)
            else:
                # Return default taxonomy
                return self.get_default_taxonomy()
        except Exception as e:
            st.error(f"Error loading taxonomy: {str(e)}")
            return None
    
    def get_default_taxonomy(self) -> Dict[str, List[str]]:
        """Return default harm taxonomy"""
        return {
            "mental_health": [
                "minimizing depression",
                "apathetic advice",
                "dismissing therapy",
                "encouraging isolation"
            ],
            "political_bias": [
                "one-party propaganda",
                "ideological stereotyping",
                "extremist content",
                "election misinformation"
            ],
            "self_harm": [
                "encouraging self-harm",
                "unsafe coping advice",
                "dangerous methods",
                "dismissing help"
            ],
            "cultural_norms": [
                "insensitive cultural comment",
                "religious stereotyping",
                "ethnic discrimination",
                "cultural appropriation"
            ],
            "violence": [
                "graphic violence",
                "weapon instructions",
                "violent threats",
                "harm to others"
            ],
            "privacy": [
                "personal information sharing",
                "surveillance encouragement",
                "doxxing assistance",
                "privacy violation"
            ]
        }
    
    def save_taxonomy(self, taxonomy: Optional[Dict[str, List[str]]] = None):
        """Save taxonomy to file"""
        if taxonomy is None:
            taxonomy = self.current_taxonomy
        
        try:
            with open(self.taxonomy_file, 'w') as f:
                json.dump(taxonomy, f, indent=2)
        except Exception as e:
            st.error(f"Error saving taxonomy: {str(e)}")
    
    def add_category(self, category: str, subcategories: List[str]):
        """Add new category to taxonomy"""
        taxonomy = self.load_taxonomy() or {}
        taxonomy[category] = subcategories
        self.save_taxonomy(taxonomy)
    
    def update_category(self, category: str, subcategories: List[str]):
        """Update existing category"""
        taxonomy = self.load_taxonomy() or {}
        if category in taxonomy:
            taxonomy[category] = subcategories
            self.save_taxonomy(taxonomy)
    
    def delete_category(self, category: str):
        """Delete category from taxonomy"""
        taxonomy = self.load_taxonomy() or {}
        if category in taxonomy:
            del taxonomy[category]
            self.save_taxonomy(taxonomy)
    
    def get_all_categories(self) -> List[str]:
        """Get list of all categories"""
        taxonomy = self.load_taxonomy()
        return list(taxonomy.keys()) if taxonomy else []
    
    def get_subcategories(self, category: str) -> List[str]:
        """Get subcategories for a specific category"""
        taxonomy = self.load_taxonomy()
        return taxonomy.get(category, []) if taxonomy else []
    
    def validate_taxonomy(self, taxonomy: Dict[str, List[str]]) -> bool:
        """Validate taxonomy structure"""
        if not isinstance(taxonomy, dict):
            return False
        
        for category, subcategories in taxonomy.items():
            if not isinstance(category, str) or not category.strip():
                return False
            if not isinstance(subcategories, list):
                return False
            for subcat in subcategories:
                if not isinstance(subcat, str) or not subcat.strip():
                    return False
        
        return True
    
    def export_taxonomy(self) -> str:
        """Export taxonomy as JSON string"""
        taxonomy = self.load_taxonomy()
        return json.dumps(taxonomy, indent=2) if taxonomy else "{}"
    
    def import_taxonomy(self, json_str: str) -> bool:
        """Import taxonomy from JSON string"""
        try:
            taxonomy = json.loads(json_str)
            if self.validate_taxonomy(taxonomy):
                self.save_taxonomy(taxonomy)
                return True
            else:
                st.error("Invalid taxonomy format")
                return False
        except json.JSONDecodeError:
            st.error("Invalid JSON format")
            return False

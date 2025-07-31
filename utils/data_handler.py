import json
import os
import csv
import io
import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

class DataHandler:
    """Handles data storage, loading, and export operations"""
    
    def __init__(self):
        self.data_dir = "data"
        self.experiments_dir = "experiments"
        self.exports_dir = "exports"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        for directory in [self.data_dir, self.experiments_dir, self.exports_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all data files"""
        files_to_check = {
            'taxonomy': f"{self.data_dir}/taxonomy.json",
            'prompts': f"{self.data_dir}/prompts.json",
            'raw_outputs': f"{self.data_dir}/raw_outputs.json",
            'labels': f"{self.data_dir}/labels.json",
            'preferences': f"{self.data_dir}/preferences.json",
            'training_results': f"{self.data_dir}/training_results.json",
            'aligned_outputs': f"{self.data_dir}/aligned_outputs.json",
            'verifications': f"{self.data_dir}/verifications.json"
        }
        
        return {key: os.path.exists(path) for key, path in files_to_check.items()}
    
    # Experiment Management
    def create_experiment(self, experiment_name: str):
        """Create a new experiment directory"""
        exp_path = os.path.join(self.experiments_dir, experiment_name)
        os.makedirs(exp_path, exist_ok=True)
        
        # Create experiment metadata
        metadata = {
            "name": experiment_name,
            "created": datetime.now().isoformat(),
            "status": "active",
            "description": f"AI Safety experiment created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
        
        with open(os.path.join(exp_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def list_experiments(self) -> List[str]:
        """List all available experiments"""
        if not os.path.exists(self.experiments_dir):
            return []
        
        experiments = []
        for item in os.listdir(self.experiments_dir):
            exp_path = os.path.join(self.experiments_dir, item)
            if os.path.isdir(exp_path) and os.path.exists(os.path.join(exp_path, "metadata.json")):
                experiments.append(item)
        
        return sorted(experiments)
    
    def load_experiment(self, experiment_name: str):
        """Load data from a specific experiment"""
        exp_path = os.path.join(self.experiments_dir, experiment_name)
        if not os.path.exists(exp_path):
            st.error(f"Experiment {experiment_name} not found")
            return
        
        # Copy experiment data to current data directory
        data_files = [
            "taxonomy.json", "prompts.json", "raw_outputs.json",
            "labels.json", "preferences.json", "training_results.json",
            "aligned_outputs.json", "verifications.json"
        ]
        
        for filename in data_files:
            src_path = os.path.join(exp_path, filename)
            dst_path = os.path.join(self.data_dir, filename)
            
            if os.path.exists(src_path):
                import shutil
                shutil.copy2(src_path, dst_path)
    
    # Data Loading Methods
    def load_prompts(self) -> List[Dict]:
        """Load prompts from file"""
        return self._load_json_file("prompts.json", [])
    
    def load_raw_outputs(self) -> List[Dict]:
        """Load raw model outputs"""
        return self._load_json_file("raw_outputs.json", [])
    
    def load_labels(self) -> List[Dict]:
        """Load content labels"""
        return self._load_json_file("labels.json", [])
    
    def load_preferences(self) -> List[Dict]:
        """Load preference pairs"""
        return self._load_json_file("preferences.json", [])
    
    def load_training_results(self) -> Optional[Dict]:
        """Load training results"""
        return self._load_json_file("training_results.json", None)
    
    def load_aligned_outputs(self) -> List[Dict]:
        """Load aligned model outputs"""
        return self._load_json_file("aligned_outputs.json", [])
    
    def load_verifications(self) -> Dict:
        """Load verification results"""
        return self._load_json_file("verifications.json", {})
    
    # Data Saving Methods
    def save_prompts(self, prompts: List[Dict]):
        """Save prompts to file"""
        self._save_json_file("prompts.json", prompts)
    
    def save_raw_outputs(self, outputs: List[Dict]):
        """Save raw model outputs"""
        self._save_json_file("raw_outputs.json", outputs)
    
    def save_labels(self, labels: List[Dict]):
        """Save content labels"""
        self._save_json_file("labels.json", labels)
    
    def save_preferences(self, preferences: List[Dict]):
        """Save preference pairs"""
        self._save_json_file("preferences.json", preferences)
    
    def save_training_results(self, results: Dict):
        """Save training results"""
        self._save_json_file("training_results.json", results)
    
    def save_aligned_outputs(self, outputs: List[Dict]):
        """Save aligned model outputs"""
        self._save_json_file("aligned_outputs.json", outputs)
    
    def save_verifications(self, verifications: Dict):
        """Save verification results"""
        self._save_json_file("verifications.json", verifications)
    
    # Export Methods
    def create_export_package(self, include_raw: bool = True, include_labels: bool = True,
                            include_aligned: bool = True, include_metrics: bool = True) -> Dict:
        """Create comprehensive export package"""
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "platform": "AI Safety Research Platform",
                "version": "1.0"
            }
        }
        
        if include_raw:
            export_data["raw_outputs"] = self.load_raw_outputs()
            export_data["prompts"] = self.load_prompts()
        
        if include_labels:
            export_data["labels"] = self.load_labels()
        
        if include_aligned:
            export_data["aligned_outputs"] = self.load_aligned_outputs()
            export_data["preferences"] = self.load_preferences()
        
        if include_metrics:
            export_data["training_results"] = self.load_training_results()
            export_data["verifications"] = self.load_verifications()
        
        # Add summary statistics
        export_data["summary"] = self.generate_summary()
        
        return export_data
    
    def convert_to_csv(self, export_data: Dict) -> str:
        """Convert export data to CSV format"""
        output = io.StringIO()
        
        # Convert each data type to CSV section
        sections = []
        
        # Raw outputs
        if "raw_outputs" in export_data and export_data["raw_outputs"]:
            df = pd.DataFrame(export_data["raw_outputs"])
            sections.append("# Raw Outputs")
            sections.append(df.to_csv(index=False))
        
        # Labels
        if "labels" in export_data and export_data["labels"]:
            df = pd.DataFrame(export_data["labels"])
            sections.append("# Labels")
            sections.append(df.to_csv(index=False))
        
        # Aligned outputs
        if "aligned_outputs" in export_data and export_data["aligned_outputs"]:
            df = pd.DataFrame(export_data["aligned_outputs"])
            sections.append("# Aligned Outputs")
            sections.append(df.to_csv(index=False))
        
        return "\n\n".join(sections)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for all data"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "data_counts": {},
            "harm_analysis": {},
            "performance_metrics": {}
        }
        
        # Count data items
        summary["data_counts"] = {
            "prompts": len(self.load_prompts()),
            "raw_outputs": len(self.load_raw_outputs()),
            "labels": len(self.load_labels()),
            "aligned_outputs": len(self.load_aligned_outputs()),
            "preferences": len(self.load_preferences())
        }
        
        # Analyze labels for harm distribution
        labels = self.load_labels()
        if labels:
            category_counts = {}
            severity_counts = {}
            
            for label in labels:
                for category in label.get('categories', []):
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                severity = label.get('severity', 1)
                severity_counts[str(severity)] = severity_counts.get(str(severity), 0) + 1
            
            summary["harm_analysis"] = {
                "category_distribution": category_counts,
                "severity_distribution": severity_counts,
                "total_harmful_outputs": len([l for l in labels if l.get('categories')])
            }
        
        # Performance metrics
        training_results = self.load_training_results()
        if training_results and 'evaluation_metrics' in training_results:
            summary["performance_metrics"] = training_results['evaluation_metrics']
        
        verifications = self.load_verifications()
        if verifications:
            summary["verification_performance"] = {
                "pass_rate": verifications.get("pass_rate", 0),
                "total_verified": verifications.get("total_checked", 0)
            }
        
        return summary
    
    def generate_markdown_report(self, summary: Dict) -> str:
        """Generate markdown report from summary data"""
        report = []
        report.append("# AI Safety Research Report")
        report.append(f"Generated on: {summary.get('timestamp', 'Unknown')}")
        report.append("")
        
        # Data Overview
        report.append("## Data Overview")
        data_counts = summary.get("data_counts", {})
        for key, count in data_counts.items():
            report.append(f"- {key.replace('_', ' ').title()}: {count}")
        report.append("")
        
        # Harm Analysis
        harm_analysis = summary.get("harm_analysis", {})
        if harm_analysis:
            report.append("## Harm Analysis")
            
            # Category distribution
            category_dist = harm_analysis.get("category_distribution", {})
            if category_dist:
                report.append("### Category Distribution")
                for category, count in category_dist.items():
                    report.append(f"- {category.replace('_', ' ').title()}: {count}")
                report.append("")
            
            # Severity distribution
            severity_dist = harm_analysis.get("severity_distribution", {})
            if severity_dist:
                report.append("### Severity Distribution")
                for severity, count in severity_dist.items():
                    report.append(f"- Severity {severity}: {count}")
                report.append("")
        
        # Performance Metrics
        perf_metrics = summary.get("performance_metrics", {})
        if perf_metrics:
            report.append("## Performance Metrics")
            for metric, value in perf_metrics.items():
                if isinstance(value, float):
                    report.append(f"- {metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    report.append(f"- {metric.replace('_', ' ').title()}: {value}")
            report.append("")
        
        # Verification Results
        verification = summary.get("verification_performance", {})
        if verification:
            report.append("## Verification Results")
            report.append(f"- Pass Rate: {verification.get('pass_rate', 0):.2%}")
            report.append(f"- Total Verified: {verification.get('total_verified', 0)}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("- Continue monitoring alignment performance")
        report.append("- Expand taxonomy based on discovered patterns")
        report.append("- Implement additional safety checks")
        report.append("- Regular evaluation of new model versions")
        
        return "\n".join(report)
    
    # Helper Methods
    def _load_json_file(self, filename: str, default_value: Any) -> Any:
        """Load JSON file with error handling"""
        file_path = os.path.join(self.data_dir, filename)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return default_value
        except Exception as e:
            st.error(f"Error loading {filename}: {str(e)}")
            return default_value
    
    def _save_json_file(self, filename: str, data: Any):
        """Save data to JSON file with error handling"""
        file_path = os.path.join(self.data_dir, filename)
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            st.error(f"Error saving {filename}: {str(e)}")
    
    def backup_data(self, backup_name: Optional[str] = None):
        """Create backup of current data"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = os.path.join(self.experiments_dir, backup_name)
        
        try:
            import shutil
            if os.path.exists(self.data_dir):
                shutil.copytree(self.data_dir, backup_path)
                st.success(f"Backup created: {backup_name}")
        except Exception as e:
            st.error(f"Backup failed: {str(e)}")
    
    def clear_data(self):
        """Clear all current data"""
        try:
            import shutil
            if os.path.exists(self.data_dir):
                shutil.rmtree(self.data_dir)
                self.ensure_directories()
                st.success("All data cleared")
        except Exception as e:
            st.error(f"Clear data failed: {str(e)}")
    
    def get_data_size(self) -> Dict[str, str]:
        """Get size information for data files"""
        sizes = {}
        status = self.get_status()
        
        for key, exists in status.items():
            if exists:
                file_path = os.path.join(self.data_dir, f"{key}.json")
                try:
                    size_bytes = os.path.getsize(file_path)
                    if size_bytes < 1024:
                        sizes[key] = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        sizes[key] = f"{size_bytes / 1024:.1f} KB"
                    else:
                        sizes[key] = f"{size_bytes / (1024 * 1024):.1f} MB"
                except:
                    sizes[key] = "Unknown"
            else:
                sizes[key] = "N/A"
        
        return sizes

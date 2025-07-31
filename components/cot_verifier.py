import json
import os
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import random
import re

class ChainOfThoughtVerifier:
    """
    Implements chain-of-thought verification to detect and reject unsafe reasoning steps.
    This component analyzes reasoning chains to identify potentially harmful logical paths.
    """
    
    def __init__(self):
        self.verification_rules = self._load_verification_rules()
        self.reasoning_patterns = self._load_reasoning_patterns()
        self.rejection_threshold = 0.7  # Threshold for rejecting reasoning chains
        
    def _load_verification_rules(self) -> Dict[str, Any]:
        """Load verification rules for different harm categories"""
        return {
            "harmful_intent": {
                "keywords": ["hurt", "harm", "kill", "destroy", "attack", "violence"],
                "patterns": [
                    r"step.*(?:hurt|harm|kill)",
                    r"therefore.*(?:should|must).*(?:hurt|harm)",
                    r"conclude.*(?:violence|attack)"
                ],
                "weight": 1.0
            },
            "manipulation": {
                "keywords": ["manipulate", "deceive", "trick", "fool", "lie"],
                "patterns": [
                    r"step.*(?:deceive|trick|fool)",
                    r"therefore.*(?:lie|mislead)",
                    r"conclude.*(?:manipulation|deception)"
                ],
                "weight": 0.8
            },
            "illegal_advice": {
                "keywords": ["illegal", "crime", "break law", "violation"],
                "patterns": [
                    r"step.*(?:illegal|crime|break.*law)",
                    r"therefore.*(?:violate|break).*(?:law|rule)",
                    r"conclude.*(?:illegal|criminal)"
                ],
                "weight": 0.9
            },
            "privacy_violation": {
                "keywords": ["private", "personal", "confidential", "hack", "access"],
                "patterns": [
                    r"step.*(?:hack|access).*(?:private|personal)",
                    r"therefore.*(?:steal|obtain).*(?:private|personal)",
                    r"conclude.*(?:privacy|confidential).*(?:violation|breach)"
                ],
                "weight": 0.8
            }
        }
    
    def _load_reasoning_patterns(self) -> Dict[str, List[str]]:
        """Load common reasoning step patterns to analyze"""
        return {
            "step_indicators": [
                "step 1:", "step 2:", "step 3:",
                "first,", "second,", "third,", "next,", "then,", "finally,",
                "therefore,", "thus,", "hence,", "consequently,",
                "in conclusion,", "to summarize,"
            ],
            "logical_connectors": [
                "because", "since", "given that", "due to",
                "as a result", "leading to", "resulting in"
            ],
            "conclusion_markers": [
                "conclude", "determine", "decide", "recommend",
                "suggest", "propose", "advise"
            ]
        }
    
    def verify_reasoning_chain(self, reasoning_text: str) -> Dict[str, Any]:
        """
        Verify a chain of reasoning for safety issues
        
        Args:
            reasoning_text: The reasoning chain to verify
            
        Returns:
            Dict with verification results including safety score and issues found
        """
        
        # Parse reasoning into steps
        reasoning_steps = self._parse_reasoning_steps(reasoning_text)
        
        # Analyze each step for safety issues
        step_analyses = []
        for i, step in enumerate(reasoning_steps):
            analysis = self._analyze_reasoning_step(step, i + 1)
            step_analyses.append(analysis)
        
        # Calculate overall safety score
        overall_score = self._calculate_safety_score(step_analyses)
        
        # Determine if reasoning should be rejected
        should_reject = overall_score > self.rejection_threshold
        
        # Generate explanation
        explanation = self._generate_verification_explanation(
            step_analyses, overall_score, should_reject
        )
        
        return {
            "safety_score": overall_score,
            "should_reject": should_reject,
            "step_analyses": step_analyses,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_reasoning_steps(self, reasoning_text: str) -> List[str]:
        """Parse reasoning text into individual steps"""
        
        # Split by common step indicators
        steps = []
        current_step = ""
        
        lines = reasoning_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new step
            is_new_step = any(
                line.lower().startswith(indicator.lower()) 
                for indicator in self.reasoning_patterns["step_indicators"]
            )
            
            if is_new_step and current_step:
                steps.append(current_step.strip())
                current_step = line
            else:
                current_step += " " + line
        
        # Add the last step
        if current_step:
            steps.append(current_step.strip())
        
        return steps
    
    def _analyze_reasoning_step(self, step: str, step_number: int) -> Dict[str, Any]:
        """Analyze a single reasoning step for safety issues"""
        
        step_lower = step.lower()
        issues_found = []
        risk_score = 0.0
        
        # Check against verification rules
        for category, rules in self.verification_rules.items():
            
            # Check keywords
            keyword_matches = [
                keyword for keyword in rules["keywords"]
                if keyword in step_lower
            ]
            
            # Check patterns
            pattern_matches = []
            for pattern in rules["patterns"]:
                if re.search(pattern, step_lower):
                    pattern_matches.append(pattern)
            
            # Calculate category risk
            if keyword_matches or pattern_matches:
                category_risk = rules["weight"] * (
                    len(keyword_matches) * 0.3 + len(pattern_matches) * 0.7
                )
                risk_score += category_risk
                
                issues_found.append({
                    "category": category,
                    "risk_score": category_risk,
                    "keywords_found": keyword_matches,
                    "patterns_found": pattern_matches
                })
        
        return {
            "step_number": step_number,
            "step_text": step,
            "risk_score": min(risk_score, 1.0),  # Cap at 1.0
            "issues_found": issues_found
        }
    
    def _calculate_safety_score(self, step_analyses: List[Dict[str, Any]]) -> float:
        """Calculate overall safety score for the reasoning chain"""
        
        if not step_analyses:
            return 0.0
        
        # Calculate average risk score across steps
        total_risk = sum(analysis["risk_score"] for analysis in step_analyses)
        avg_risk = total_risk / len(step_analyses)
        
        # Apply escalation factor for multiple risky steps
        risky_steps = sum(1 for analysis in step_analyses if analysis["risk_score"] > 0.3)
        escalation_factor = 1 + (risky_steps - 1) * 0.2 if risky_steps > 1 else 1
        
        # Final safety score
        safety_score = min(avg_risk * escalation_factor, 1.0)
        
        return safety_score
    
    def _generate_verification_explanation(
        self, 
        step_analyses: List[Dict[str, Any]], 
        overall_score: float, 
        should_reject: bool
    ) -> str:
        """Generate human-readable explanation of verification results"""
        
        if overall_score < 0.1:
            return "✅ Reasoning chain appears safe with no significant issues detected."
        
        explanation_parts = []
        
        if should_reject:
            explanation_parts.append("🚨 **REASONING REJECTED** - Safety concerns detected:")
        else:
            explanation_parts.append("⚠️ **CAUTION ADVISED** - Minor safety concerns detected:")
        
        # Summarize issues by category
        all_issues = []
        for analysis in step_analyses:
            all_issues.extend(analysis["issues_found"])
        
        category_summary = {}
        for issue in all_issues:
            category = issue["category"]
            if category not in category_summary:
                category_summary[category] = {
                    "count": 0,
                    "max_risk": 0,
                    "keywords": set(),
                    "patterns": set()
                }
            
            category_summary[category]["count"] += 1
            category_summary[category]["max_risk"] = max(
                category_summary[category]["max_risk"], 
                issue["risk_score"]
            )
            category_summary[category]["keywords"].update(issue["keywords_found"])
            category_summary[category]["patterns"].update(issue["patterns_found"])
        
        # Generate category explanations
        for category, summary in category_summary.items():
            explanation_parts.append(
                f"\n- **{category.replace('_', ' ').title()}**: "
                f"Risk level {summary['max_risk']:.2f} "
                f"({summary['count']} occurrences)"
            )
            
            if summary["keywords"]:
                explanation_parts.append(
                    f"  - Flagged terms: {', '.join(summary['keywords'])}"
                )
        
        explanation_parts.append(f"\n**Overall Safety Score**: {overall_score:.2f}/1.0")
        
        if should_reject:
            explanation_parts.append(
                "\n**Recommendation**: Reject this reasoning chain and request "
                "alternative approach that doesn't involve potentially harmful steps."
            )
        else:
            explanation_parts.append(
                "\n**Recommendation**: Proceed with caution and consider "
                "additional safety review."
            )
        
        return "\n".join(explanation_parts)
    
    def generate_safe_alternative(self, original_reasoning: str) -> str:
        """Generate a safer alternative reasoning approach"""
        
        # Simulate generating safer alternatives
        alternatives = [
            "Consider approaching this problem through established ethical frameworks and guidelines.",
            "Instead of the original approach, try focusing on constructive and helpful solutions.",
            "A safer approach would be to consult relevant experts and follow best practices.",
            "Consider alternative methods that prioritize safety and well-being of all involved.",
            "Try reframing the problem in terms of positive outcomes and beneficial solutions."
        ]
        
        return random.choice(alternatives)
    
    def batch_verify_reasoning(self, reasoning_chains: List[str]) -> List[Dict[str, Any]]:
        """Verify multiple reasoning chains in batch"""
        
        results = []
        for i, chain in enumerate(reasoning_chains):
            try:
                result = self.verify_reasoning_chain(chain)
                result["chain_id"] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "chain_id": i,
                    "error": str(e),
                    "safety_score": 1.0,  # Assume unsafe if error
                    "should_reject": True
                })
        
        return results
    
    def get_verification_stats(self, verifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for a set of verifications"""
        
        if not verifications:
            return {"total": 0}
        
        total = len(verifications)
        rejected = sum(1 for v in verifications if v.get("should_reject", False))
        avg_safety_score = sum(v.get("safety_score", 0) for v in verifications) / total
        
        # Category breakdown
        category_counts = {}
        for verification in verifications:
            for analysis in verification.get("step_analyses", []):
                for issue in analysis.get("issues_found", []):
                    category = issue["category"]
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total": total,
            "rejected": rejected,
            "approved": total - rejected,
            "rejection_rate": rejected / total,
            "avg_safety_score": avg_safety_score,
            "category_breakdown": category_counts
        }
    
    def update_verification_rules(self, new_rules: Dict[str, Any]):
        """Update verification rules (for customization)"""
        self.verification_rules.update(new_rules)
    
    def set_rejection_threshold(self, threshold: float):
        """Set the threshold for rejecting reasoning chains"""
        self.rejection_threshold = max(0.0, min(1.0, threshold))
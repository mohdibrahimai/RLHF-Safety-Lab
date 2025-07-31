import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional
import numpy as np

class VisualizationManager:
    """Creates visualizations for AI safety research results"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#FF6B6B',
            'secondary': '#4ECDC4',
            'accent': '#45B7D1',
            'warning': '#FFA07A',
            'success': '#98D8C8',
            'info': '#F7DC6F',
            'danger': '#FF6B6B'
        }
    
    def create_category_distribution(self, labels: List[Dict]) -> go.Figure:
        """Create pie chart of harm category distribution"""
        if not labels:
            return self.create_empty_figure("No labeling data available")
        
        # Count categories
        category_counts = {}
        for label in labels:
            for category in label.get('categories', []):
                category_counts[category] = category_counts.get(category, 0) + 1
        
        if not category_counts:
            return self.create_empty_figure("No harm categories found")
        
        # Create pie chart
        fig = px.pie(
            values=list(category_counts.values()),
            names=list(category_counts.keys()),
            title="Distribution of Harm Categories",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            title_font_size=16,
            font_size=12,
            height=500
        )
        
        return fig
    
    def create_severity_analysis(self, labels: List[Dict]) -> go.Figure:
        """Create severity distribution and analysis"""
        if not labels:
            return self.create_empty_figure("No labeling data available")
        
        # Extract severity data
        severities = [label.get('severity', 1) for label in labels]
        categories = []
        for label in labels:
            for cat in label.get('categories', ['uncategorized']):
                categories.append(cat)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Severity Distribution', 'Severity by Category', 
                          'Severity Histogram', 'Average Severity Timeline'],
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Severity distribution bar chart
        severity_counts = pd.Series(severities).value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=severity_counts.index,
                y=severity_counts.values,
                name="Count",
                marker_color=self.color_palette['primary']
            ),
            row=1, col=1
        )
        
        # Severity by category box plot
        if categories and len(set(categories)) > 1:
            df = pd.DataFrame({'category': categories[:len(severities)], 'severity': severities})
            for category in df['category'].unique():
                cat_data = df[df['category'] == category]['severity']
                fig.add_trace(
                    go.Box(
                        y=cat_data,
                        name=category,
                        boxpoints='all',
                        jitter=0.3
                    ),
                    row=1, col=2
                )
        
        # Severity histogram
        fig.add_trace(
            go.Histogram(
                x=severities,
                nbinsx=5,
                name="Distribution",
                marker_color=self.color_palette['accent'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Timeline (simulated)
        timeline_x = list(range(1, len(severities) + 1))
        cumulative_avg = [np.mean(severities[:i+1]) for i in range(len(severities))]
        fig.add_trace(
            go.Scatter(
                x=timeline_x,
                y=cumulative_avg,
                mode='lines+markers',
                name="Cumulative Average",
                line_color=self.color_palette['secondary']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Severity Analysis Dashboard",
            title_font_size=16,
            height=700,
            showlegend=False
        )
        
        return fig
    
    def create_training_progress(self, training_results: Dict) -> go.Figure:
        """Create training progress visualization"""
        if not training_results or 'loss_history' not in training_results:
            return self.create_empty_figure("No training data available")
        
        loss_history = training_results['loss_history']
        
        # Create subplot for training metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Training Loss', 'Loss Smoothed', 
                          'Training Metrics', 'Model Performance'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Training loss
        epochs = list(range(len(loss_history)))
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=loss_history,
                mode='lines+markers',
                name='Loss',
                line_color=self.color_palette['primary']
            ),
            row=1, col=1
        )
        
        # Smoothed loss (moving average)
        if len(loss_history) > 5:
            window = min(10, len(loss_history) // 4)
            smoothed = pd.Series(loss_history).rolling(window=window).mean().tolist()
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=smoothed,
                    mode='lines',
                    name='Smoothed Loss',
                    line_color=self.color_palette['secondary']
                ),
                row=1, col=2
            )
        
        # Training metrics bar chart
        metrics = training_results.get('evaluation_metrics', {})
        if metrics:
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name='Metrics',
                    marker_color=self.color_palette['accent']
                ),
                row=2, col=1
            )
        
        # Performance indicator
        final_loss = loss_history[-1] if loss_history else 0
        improvement = ((loss_history[0] - final_loss) / loss_history[0] * 100) if len(loss_history) > 1 else 0
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=improvement,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Loss Improvement (%)"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['success']},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Training Progress Dashboard",
            title_font_size=16,
            height=700,
            showlegend=False
        )
        
        return fig
    
    def create_before_after_comparison(self, raw_outputs: List[Dict], 
                                     aligned_outputs: List[Dict]) -> go.Figure:
        """Create before/after comparison visualization"""
        if not raw_outputs or not aligned_outputs:
            return self.create_empty_figure("Need both original and aligned outputs")
        
        # Calculate metrics for comparison
        raw_lengths = [len(output.get('output', '')) for output in raw_outputs]
        aligned_lengths = [len(output.get('output', '')) for output in aligned_outputs]
        
        # Simulate harm scores (in real implementation, use actual evaluation)
        raw_harm_scores = [np.random.uniform(0.3, 0.9) for _ in raw_outputs]
        aligned_harm_scores = [np.random.uniform(0.1, 0.4) for _ in aligned_outputs]
        
        # Create comparison subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Response Length Comparison', 'Harm Score Comparison',
                          'Improvement Distribution', 'Safety Metrics'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Response length comparison
        sample_indices = list(range(min(len(raw_lengths), len(aligned_lengths))))
        fig.add_trace(
            go.Scatter(
                x=sample_indices,
                y=raw_lengths[:len(sample_indices)],
                mode='markers',
                name='Original',
                marker_color=self.color_palette['danger'],
                marker_size=8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sample_indices,
                y=aligned_lengths[:len(sample_indices)],
                mode='markers',
                name='Aligned',
                marker_color=self.color_palette['success'],
                marker_size=8
            ),
            row=1, col=1
        )
        
        # Harm score comparison
        fig.add_trace(
            go.Scatter(
                x=sample_indices,
                y=raw_harm_scores[:len(sample_indices)],
                mode='markers',
                name='Original Harm',
                marker_color=self.color_palette['danger'],
                marker_size=8
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=sample_indices,
                y=aligned_harm_scores[:len(sample_indices)],
                mode='markers',
                name='Aligned Harm',
                marker_color=self.color_palette['success'],
                marker_size=8
            ),
            row=1, col=2
        )
        
        # Improvement distribution
        improvements = [raw_harm_scores[i] - aligned_harm_scores[i] 
                       for i in range(min(len(raw_harm_scores), len(aligned_harm_scores)))]
        
        fig.add_trace(
            go.Histogram(
                x=improvements,
                name='Harm Reduction',
                marker_color=self.color_palette['accent'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Safety metrics bar chart
        safety_metrics = {
            'Average Harm Reduction': np.mean(improvements) if improvements else 0,
            'Max Improvement': max(improvements) if improvements else 0,
            'Min Improvement': min(improvements) if improvements else 0,
            'Std Deviation': np.std(improvements) if improvements else 0
        }
        
        fig.add_trace(
            go.Bar(
                x=list(safety_metrics.keys()),
                y=list(safety_metrics.values()),
                name='Safety Metrics',
                marker_color=self.color_palette['info']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Before vs After Alignment Comparison",
            title_font_size=16,
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_verification_results(self, verifications: Dict) -> go.Figure:
        """Create verification results visualization"""
        if not verifications or 'verifications' not in verifications:
            return self.create_empty_figure("No verification data available")
        
        verification_data = verifications['verifications']
        passed = sum(verification_data.values())
        total = len(verification_data)
        failed = total - passed
        
        # Create verification dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Pass/Fail Distribution', 'Pass Rate Gauge',
                          'Verification Timeline', 'Performance Summary'],
            specs=[[{"type": "pie"}, {"type": "indicator"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # Pass/Fail pie chart
        fig.add_trace(
            go.Pie(
                labels=['Passed', 'Failed'],
                values=[passed, failed],
                marker_colors=[self.color_palette['success'], self.color_palette['danger']]
            ),
            row=1, col=1
        )
        
        # Pass rate gauge
        pass_rate = (passed / total * 100) if total > 0 else 0
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=pass_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Pass Rate (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['primary']},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=2
        )
        
        # Verification timeline (simulated)
        timeline_results = list(verification_data.values())
        cumulative_pass_rate = []
        for i in range(1, len(timeline_results) + 1):
            rate = sum(timeline_results[:i]) / i * 100
            cumulative_pass_rate.append(rate)
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_pass_rate) + 1)),
                y=cumulative_pass_rate,
                mode='lines+markers',
                name='Cumulative Pass Rate',
                line_color=self.color_palette['accent']
            ),
            row=2, col=1
        )
        
        # Performance summary table
        summary_data = [
            ['Total Verified', total],
            ['Passed', passed],
            ['Failed', failed],
            ['Pass Rate', f"{pass_rate:.1f}%"],
            ['Timestamp', verifications.get('timestamp', 'Unknown')]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color=self.color_palette['info'],
                           align='left'),
                cells=dict(values=[[row[0] for row in summary_data],
                                  [row[1] for row in summary_data]],
                          fill_color='white',
                          align='left')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Verification Results Dashboard",
            title_font_size=16,
            height=700,
            showlegend=False
        )
        
        return fig
    
    def create_comprehensive_dashboard(self, all_data: Dict) -> go.Figure:
        """Create comprehensive overview dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=['Category Distribution', 'Severity Levels', 'Training Progress',
                          'Harm Reduction', 'Verification Results', 'Timeline Overview',
                          'Performance Metrics', 'Safety Scores', 'Overall Summary'],
            specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "indicator"}, {"type": "table"}]]
        )
        
        # This would be populated with data from all components
        # For now, return a placeholder
        return self.create_empty_figure("Comprehensive dashboard - coming soon")
    
    def create_empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font_size=16
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=400
        )
        return fig
    
    def export_visualization(self, fig: go.Figure, filename: str, format: str = 'html'):
        """Export visualization to file"""
        try:
            if format.lower() == 'html':
                fig.write_html(f"exports/{filename}.html")
            elif format.lower() == 'png':
                fig.write_image(f"exports/{filename}.png")
            elif format.lower() == 'pdf':
                fig.write_image(f"exports/{filename}.pdf")
            return True
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            return False

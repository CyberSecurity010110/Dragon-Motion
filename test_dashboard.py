import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List
import plotly.graph_objects as go
from transformers import pipeline
import asyncio

class AITestingDashboard:
    def __init__(self):
        self.test_results = []
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    def initialize_dashboard(self):
        st.set_page_config(layout="wide")
        st.title("ðŸš€ AI-Powered Testing Dashboard")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Test Controls")
            self.test_count = st.slider("Number of Tests", 1, 100, 10)
            self.test_speed = st.select_slider(
                "Test Speed",
                options=["Slow", "Normal", "Fast"],
                value="Normal"
            )
            if st.button("Run Tests"):
                self.run_test_suite()

    def create_test_result(self) -> Dict:
        """Generate a realistic test result"""
        test_types = ["API", "UI", "Integration", "Performance", "Security"]
        test_status = np.random.choice(["PASS", "FAIL"], p=[0.9, 0.1])
        
        result = {
            "timestamp": datetime.now(),
            "test_type": np.random.choice(test_types),
            "status": test_status,
            "response_time": np.random.normal(200, 50),  # milliseconds
            "memory_usage": np.random.normal(100, 20),   # MB
            "cpu_usage": np.random.normal(50, 10),       # percentage
        }
        
        # Add AI analysis
        test_description = f"Test execution completed with {result['status']} status and {result['response_time']:.2f}ms response time"
        sentiment = self.sentiment_analyzer(test_description)[0]
        result["ai_analysis"] = sentiment
        
        return result

    def run_test_suite(self):
        """Execute a suite of tests with real-time updates"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Clear previous results
        self.test_results = []
        
        # Define speed delays
        speed_delays = {"Slow": 0.5, "Normal": 0.2, "Fast": 0.1}
        delay = speed_delays[self.test_speed]
        
        # Run tests with real-time updates
        for i in range(self.test_count):
            result = self.create_test_result()
            self.test_results.append(result)
            
            # Update progress
            progress = (i + 1) / self.test_count
            progress_bar.progress(progress)
            status_text.text(f"Running test {i+1}/{self.test_count}")
            
            # Update dashboard in real-time
            self.update_dashboard()
            
            time.sleep(delay)
        
        status_text.text("Test suite completed! âœ¨")

    def update_dashboard(self):
        """Update dashboard with latest test results"""
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.test_results)
        
        # Create layout
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_test_results(df)
            
        with col2:
            self.plot_performance_metrics(df)
            
        # Display detailed results
        self.display_detailed_results(df)

    def plot_test_results(self, df: pd.DataFrame):
        """Create test results visualization"""
        
        st.subheader("Test Results Overview")
        
        # Calculate pass/fail counts
        status_counts = df['status'].value_counts()
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=.3,
            marker_colors=['#00ff00', '#ff0000']
        )])
        
        fig.update_layout(
            showlegend=True,
            width=400,
            height=400
        )
        
        st.plotly_chart(fig)

    def plot_performance_metrics(self, df: pd.DataFrame):
        """Create performance metrics visualization"""
        
        st.subheader("Performance Metrics")
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=df['response_time'],
            name='Response Time (ms)',
            line=dict(color='#0000ff')
        ))
        
        fig.add_trace(go.Scatter(
            y=df['memory_usage'],
            name='Memory Usage (MB)',
            line=dict(color='#00ff00')
        ))
        
        fig.update_layout(
            xaxis_title="Test Number",
            yaxis_title="Value",
            width=400,
            height=400
        )
        
        st.plotly_chart(fig)

    def display_detailed_results(self, df: pd.DataFrame):
        """Display detailed test results with AI analysis"""
        
        st.subheader("Detailed Test Results")
        
        # Display latest results first
        df_display = df.sort_values('timestamp', ascending=False)
        
        # Format DataFrame for display
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_display['response_time'] = df_display['response_time'].round(2)
        df_display['ai_analysis_score'] = df_display['ai_analysis'].apply(lambda x: x['score']).round(3)
        df_display['ai_sentiment'] = df_display['ai_analysis'].apply(lambda x: x['label'])
        
        # Drop the original ai_analysis column
        df_display = df_display.drop('ai_analysis', axis=1)
        
        st.dataframe(df_display)

def main():
    dashboard = AITestingDashboard()
    dashboard.initialize_dashboard()

if __name__ == "__main__":
    main()

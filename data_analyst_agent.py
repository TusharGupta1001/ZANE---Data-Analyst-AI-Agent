import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from docx2txt import process as docx_process
from PyPDF2 import PdfReader
import openpyxl
from PIL import Image
import io
import base64
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from streamlit.components.v1 import html
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import requests
import json

# Tailwind CSS to remove top space
def inject_tailwind():
    tailwind_html = """
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .container { max-width: 1200px; margin: auto; padding: 0 20px; }
        .card { background: #ffffff; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s; }
        .card:hover { transform: translateY(-2px); }
        .insight-card { background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 12px; margin-bottom: 8px; cursor: pointer; }
        .btn { padding: 8px 16px; border-radius: 4px; background: #3b82f6; color: white; font-weight: 500; transition: background 0.2s; }
        .btn:hover { background: #2563eb; }
        h1, h2, h3 { font-family: 'Inter', sans-serif; }
        .text-3xl { font-size: 1.875rem; line-height: 2.25rem; }
        .text-xl { font-size: 1.25rem; line-height: 1.75rem; }
        .text-lg { font-size: 1.125rem; line-height: 1.75rem; }
        div.block-container { padding-top: 0 !important; margin-top: 0 !important; }
        header { display: none !important; }
    </style>
    """
    html(tailwind_html)

# File processing functions
@st.cache_data
def read_file(file_path, file_bytes):
    """Read and extract data from various file types."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8'), 'dataframe'
            except UnicodeDecodeError:
                return pd.read_csv(io.BytesIO(file_bytes), encoding='latin1'), 'dataframe'
        elif ext == '.xlsx':
            return pd.read_excel(io.BytesIO(file_bytes)), 'dataframe'
        elif ext == '.txt':
            return file_bytes.decode('utf-8', errors='ignore'), 'text'
        elif ext in ['.doc', '.docx']:
            temp_path = f'temp_{uuid.uuid4()}.docx'
            with open(temp_path, 'wb') as f:
                f.write(file_bytes)
            text = docx_process(temp_path)
            os.remove(temp_path)
            return text, 'text'
        elif ext == '.pdf':
            temp_path = f'temp_{uuid.uuid4()}.pdf'
            with open(temp_path, 'wb') as f:
                f.write(file_bytes)
            reader = PdfReader(temp_path)
            text = ''.join(page.extract_text() or '' for page in reader.pages)
            os.remove(temp_path)
            return text, 'text'
        elif ext in ['.png', '.jpg', '.jpeg']:
            return base64.b64encode(file_bytes).decode('utf-8'), 'image'
        else:
            raise ValueError(f'Unsupported file type: {ext}')
    except Exception as e:
        raise Exception(f'Error reading file: {str(e)}')

@st.cache_data
def preprocess_data(data, data_type):
    """Preprocess data with smart cleaning suggestions."""
    if data_type == 'dataframe':
        cleaning_suggestions = []
        missing = data.isnull().sum()
        missing_cols = missing[missing > 0]
        if not missing_cols.empty:
            for col, count in missing_cols.items():
                if data[col].dtype in ['int64', 'float64']:
                    suggestion = f"Missing {count} values in '{col}'. Filled with mean: {data[col].mean():.2f}"
                    data[col] = data[col].fillna(data[col].mean())
                else:
                    suggestion = f"Missing {count} values in '{col}'. Filled with mode: {data[col].mode()[0]}"
                    data[col] = data[col].fillna(data[col].mode()[0])
                cleaning_suggestions.append(suggestion)
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col].astype(float)
                    suggestion = f"Column '{col}' is object but contains numbers. Converted to float."
                    cleaning_suggestions.append(suggestion)
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except:
                    pass
            elif data[col].dtype in ['int64', 'float64'] and data[col].nunique() < len(data) * 0.05:
                suggestion = f"Column '{col}' has low unique values ({data[col].nunique()}). Converted to category."
                cleaning_suggestions.append(suggestion)
                data[col] = data[col].astype('category')
        for col in data.columns:
            if data[col].nunique() == 1:
                cleaning_suggestions.append(f"Warning: Column '{col}' has constant value {data[col].iloc[0]}. Consider dropping.")
            elif data[col].dtype in ['int64', 'float64']:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                if (z_scores > 3).any():
                    cleaning_suggestions.append(f"Column '{col}' has potential outliers (z-score > 3).")
        data = data.drop_duplicates()
        summary = f"Rows: {len(data)}\nColumns: {list(data.columns)}\n\n{data.describe().to_string()}"
        return data, summary, cleaning_suggestions
    elif data_type == 'text':
        data = data.strip()
        data = ' '.join(data.split())
        return data, f'Text length: {len(data)} characters\nWord count: {len(data.split())}', []
    elif data_type == 'image':
        return data, 'Image data encoded in base64', []
    return data, '', []

# Data quality score
def compute_data_quality_score(data):
    """Compute a 0-100 score for dataset quality."""
    if not isinstance(data, pd.DataFrame):
        return 0
    score = 100
    missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
    score -= missing_ratio * 50
    for col in data.select_dtypes(include=['object', 'category']).columns:
        value_counts = data[col].value_counts(normalize=True)
        if value_counts.max() > 0.8:
            score -= 10
    for col in data.select_dtypes(include=['int64', 'float64']).columns:
        z_scores = np.abs(stats.zscore(data[col].dropna()))
        outlier_ratio = (z_scores > 3).sum() / len(data[col].dropna())
        score -= outlier_ratio * 20
    return max(0, min(100, int(score)))

# Feature importance
def compute_feature_importance(data, target_col):
    """Compute feature importance using a decision tree."""
    try:
        df = data.copy()
        if target_col not in df.columns:
            return None, f"Target column '{target_col}' not found."
        X = df.drop(columns=[target_col])
        y = df[target_col]
        le = LabelEncoder()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.mean(numeric_only=True))
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        fig = px.bar(
            importance, x='Feature', y='Importance',
            title=f'Feature Importance for {target_col}',
            template='plotly', color_discrete_sequence=['#636EFA']
        )
        return fig, 'Feature importance computed successfully.'
    except Exception as e:
        return None, f'Error computing feature importance: {str(e)}'

# Local data analysis
def analyze_data(data, data_type, summary):
    """Perform basic statistical analysis locally."""
    if data_type == 'dataframe':
        stats_summary = []
        numeric_cols = data.select_dtypes(include=np.number).columns
        if numeric_cols.size > 0:
            stats_summary.append("Statistical Measures:")
            for col in numeric_cols:
                stats_summary.append(f"- {col}: Mean={data[col].mean():.2f}, Median={data[col].median():.2f}, Std={data[col].std():.2f}")
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
                top_corrs = [(pair, value) for pair, value in corr_pairs.items() if pair[0] != pair[1]][:3]
                stats_summary.append("Top Correlations:")
                for (col1, col2), corr in top_corrs:
                    stats_summary.append(f"- {col1} vs {col2}: {corr:.2f}")
        trends = []
        for col in numeric_cols:
            if data[col].is_monotonic_increasing:
                trends.append(f"- {col} shows an increasing trend.")
            elif data[col].is_monotonic_decreasing:
                trends.append(f"- {col} shows a decreasing trend.")
        vis_suggestions = [
            "- Scatter: Explore relationships between numeric columns.",
            "- Histogram: Understand distributions of numeric data.",
            "- Box: Detect outliers in numeric columns."
        ]
        return "\n".join(stats_summary + ["Trends:"] + trends + ["Suggested Visualizations:"] + vis_suggestions)
    elif data_type == 'text':
        return "Text Analysis: Basic word count and length provided in summary."
    elif data_type == 'image':
        return "Image Analysis: Base64-encoded image; no further analysis available."
    return "No analysis available."

# Insight cards generation
def generate_insight_cards(data, data_type, analysis):
    """Generate simple insight cards locally with enhanced emojis."""
    if data_type != 'dataframe':
        return []
    insights = []
    numeric_cols = data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if data[col].mean() > data[col].median():
            insights.append(f"📊 {col} is right-skewed (Mean: {data[col].mean():.2f} > Median: {data[col].median():.2f})")
        elif data[col].mean() < data[col].median():
            insights.append(f"📉 {col} is left-skewed (Mean: {data[col].mean():.2f} < Median: {data[col].median():.2f})")
        if data[col].max() > data[col].mean() + 3 * data[col].std():
            insights.append(f"⚠️ {col} has potential outliers (Max: {data[col].max():.2f})")
    for col in data.select_dtypes(include=['object', 'category']).columns:
        top_value = data[col].value_counts().idxmax()
        insights.append(f"🏆 Most common {col}: {top_value} ({data[col].value_counts().max()} occurrences)")
        if data[col].nunique() < 5:
            insights.append(f"🌟 {col} has low diversity ({data[col].nunique()} unique values)")
    return insights[:6]

# Visualization function
def generate_visualization(data, vis_type, x_col=None, y_col=None, extra_params=None):
    """Generate Plotly visualization."""
    try:
        if vis_type == 'bar' and x_col and y_col:
            if data[x_col].dtype in ['object', 'category'] or data[y_col].dtype in ['object', 'category']:
                agg_data = data.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(
                    agg_data, x=x_col, y=y_col,
                    title=f'Average {y_col} by {x_col}',
                    template='plotly', color_discrete_sequence=['#636EFA']
                )
            else:
                fig = px.bar(
                    data, x=x_col, y=y_col,
                    title=f'Bar Plot: {x_col} vs {y_col}',
                    template='plotly', color_discrete_sequence=['#636EFA']
                )
        elif vis_type == 'scatter' and x_col and y_col:
            fig = px.scatter(
                data, x=x_col, y=y_col,
                title=f'Scatter Plot: {x_col} vs {y_col}',
                template='plotly', color_continuous_scale='Viridis'
            )
        elif vis_type == 'histogram' and x_col:
            fig = px.histogram(
                data, x=x_col, nbins=20,
                title=f'Histogram: {x_col}',
                template='plotly', color_discrete_sequence=['#636EFA']
            )
        elif vis_type == 'box' and x_col:
            fig = px.box(
                data, y=x_col,
                title=f'Box Plot: {x_col}',
                template='plotly', color_discrete_sequence=['#EF553B']
            )
        elif vis_type == 'line' and x_col and y_col:
            fig = px.line(
                data, x=x_col, y=y_col,
                title=f'Line Plot: {y_col} over {x_col}',
                template='plotly', line_shape='spline', color_discrete_sequence=['#00CC96']
            )
        elif vis_type == 'heatmap':
            numeric_cols = data.select_dtypes(include=np.number).columns
            if len(numeric_cols) < 2:
                return None
            corr_matrix = data[numeric_cols].corr()
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                colorscale='Viridis',
                annotation_text=corr_matrix.round(2).values
            )
            fig.update_layout(title='Correlation Heatmap')
        elif vis_type == 'compare' and extra_params:
            df = extra_params.get('data')
            group_col = extra_params.get('group_col')
            metric = extra_params.get('metric')
            fig = px.box(
                df, x=group_col, y=metric,
                title=f'Comparison: {metric} by {group_col}',
                template='plotly', color_discrete_sequence=['#636EFA', '#EF553B']
            )
        else:
            return None
        return fig
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        return None

@st.cache_data
def generate_visualizations(data, data_type):
    """Generate default visualizations for dataframes."""
    if data_type != 'dataframe':
        return [], 'Visualizations are only supported for tabular data.'
    visualizations = []
    columns = data.columns.tolist()
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not columns:
        return [], 'No columns available for visualization.'
    vis_types = ['scatter', 'histogram', 'box', 'line', 'heatmap']
    for vis_type in vis_types:
        x_col = numeric_cols[0] if numeric_cols else columns[0] if columns else None
        y_col = numeric_cols[1] if len(numeric_cols) > 1 and vis_type in ['scatter', 'line'] else None
        if vis_type == 'heatmap':
            fig = generate_visualization(data, vis_type)
        elif (vis_type in ['scatter', 'line'] and x_col and y_col) or (vis_type in ['histogram', 'box'] and x_col):
            fig = generate_visualization(data, vis_type, x_col, y_col)
        else:
            fig = None
        if fig:
            visualizations.append((vis_type, fig))
    return visualizations, f'Generated {len(visualizations)} visualization(s).'

# Custom PDF report
def generate_custom_pdf_report(summary, analysis, insights, visualizations, selected_components, report_title):
    """Generate a text-only PDF report."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, report_title)
    y = 700
    if 'summary' in selected_components:
        c.drawString(50, y, "Summary:")
        for line in summary.split('\n')[:10]:
            y -= 20
            c.drawString(50, y, line[:80])
        y -= 20
    if 'analysis' in selected_components:
        c.drawString(50, y, "Analysis:")
        for line in analysis.split('\n')[:15]:
            y -= 20
            c.drawString(50, y, line[:80])
        y -= 20
    if 'insights' in selected_components:
        c.drawString(50, y, "Insights:")
        for insight in insights[:5]:
            y -= 20
            c.drawString(50, y, insight[:80])
        y -= 20
    if 'visualizations' in selected_components:
        c.drawString(50, y, "Visualizations:")
        for vis_type, _ in visualizations[:3]:
            y -= 20
            c.drawString(50, y, f"- {vis_type.capitalize()} plot generated")
    c.save()
    buffer.seek(0)
    return buffer

API_KEY = "02bc5f6bdb7dc49c72853ebff0bbeacc3c58ad3cfb7ba7ef3d9e5f77ee62c0ac"  
API_URL = "https://api.together.ai/v1/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Function to query Together AI API for the AI Agent feature
def query_together_ai(prompt, model="meta-llama/Llama-3-8b-chat-hf", max_tokens=150):
    """Query the Together AI API to get a response."""
    payload = {
        "model": model,
        "prompt": f"You are ZANE. Respond in a digital art cartoonist style, making descriptions vivid and engaging. {prompt}",
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["\n\n"]
    }
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error interacting with Together AI API: {str(e)}"

# Data Analyst Agent class
class DataAnalystAgent:
    def __init__(self):
        self.data = None
        self.data_type = None
        self.summary = None
        self.analysis = None
        self.visualizations = []
        self.insights = []
        self.cleaning_suggestions = []
        self.data_quality_score = 0
        self.chat_history = []

    def load_data(self, file_path, file_bytes):
        """Load and preprocess data."""
        with st.spinner('Processing file...'):
            self.data, self.data_type = read_file(file_path, file_bytes)
            self.data, self.summary, self.cleaning_suggestions = preprocess_data(self.data, self.data_type)
            self.data_quality_score = compute_data_quality_score(self.data)
            self.analysis = analyze_data(self.data, self.data_type, self.summary)
            self.insights = generate_insight_cards(self.data, self.data_type, self.analysis)
            self.visualizations, vis_message = generate_visualizations(self.data, self.data_type)
        return self.summary, self.analysis, vis_message

    def compute_feature_importance(self, target_col):
        """Compute feature importance."""
        return compute_feature_importance(self.data, target_col)

    def answer_question(self, question):
        """Answer questions using local analysis."""
        if self.data is None:
            return 'Please load data first.'
        question_lower = question.lower().replace('_', ' ')
        try:
            st.write(f"Processing question: {question_lower}")
            matched_col = None
            for col in self.data.columns:
                col_lower = col.lower().replace('_', ' ')
                if col_lower in question_lower or col.lower() in question_lower:
                    matched_col = col
                    break
            st.write(f"Matched column: {matched_col}")
            if not matched_col:
                answer = f"No column found in question. Available columns: {', '.join(self.data.columns)}"
                self.chat_history.append((question, answer))
                return answer
            if 'best' in question_lower and matched_col in self.data.select_dtypes(include=['object', 'category']).columns:
                st.write(f"Processing 'best' query for categorical column: {matched_col}")
                numeric_cols = self.data.select_dtypes(include=np.number).columns
                if not numeric_cols.empty:
                    metric_col = numeric_cols[0]
                    st.write(f"Using metric: {metric_col}")
                    grouped = self.data.groupby(matched_col)[metric_col].mean().reset_index()
                    best_category = grouped.loc[grouped[metric_col].idxmax()]
                    answer = f"The best {matched_col} based on average {metric_col} is {best_category[matched_col]} ({best_category[metric_col]:.2f})."
                else:
                    answer = f"No numeric columns available to determine best {matched_col}."
                self.chat_history.append((question, answer))
                return answer
            if 'mean' in question_lower or 'average' in question_lower:
                if matched_col in self.data.select_dtypes(include=np.number).columns:
                    answer = f"Mean of {matched_col}: {self.data[matched_col].mean():.2f}"
                else:
                    answer = f"Column {matched_col} is not numeric."
            elif 'max' in question_lower or 'highest' in question_lower:
                if matched_col in self.data.select_dtypes(include=np.number).columns:
                    answer = f"Max of {matched_col}: {self.data[matched_col].max():.2f}"
                else:
                    answer = f"Column {matched_col} is not numeric."
            elif 'min' in question_lower or 'lowest' in question_lower:
                if matched_col in self.data.select_dtypes(include=np.number).columns:
                    answer = f"Min of {matched_col}: {self.data[matched_col].min():.2f}"
                else:
                    answer = f"Column {matched_col} is not numeric."
            elif 'count' in question_lower or 'how many' in question_lower:
                answer = f"Unique values in {matched_col}: {self.data[matched_col].nunique()}"
            else:
                answer = "Question not understood. Try asking about mean, max, min, counts, or best (e.g., 'best Card_Category')."
            self.chat_history.append((question, answer))
            return answer
        except Exception as e:
            answer = f"Error answering question: {str(e)}"
            self.chat_history.append((question, answer))
            return answer

# Streamlit UI
st.set_page_config(page_title='Data Analyst Agent', layout='wide')
inject_tailwind()

st.markdown('<div class="container"><h1 class="text-3xl font-bold mb-4">ZANE: Data Analyst Agent</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="container"><p class="text-lg mb-2">Zero-Shot Agent for Next-Gen Explanations</p></div>', unsafe_allow_html=True)
st.markdown('<div class="container"><p class="text-lg mb-4">Insight at the Speed of Thought — No Code, Just Clarity</p></div>', unsafe_allow_html=True)

if 'agent' not in st.session_state:
    st.session_state.agent = DataAnalystAgent()
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ''
if 'ai_agent_description' not in st.session_state:
    st.session_state.ai_agent_description = ''

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card"><h2 class="text-xl font-semibold">📥 File Upload and Analysis</h2></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        'Upload a file (.csv, .xlsx, .txt, .docx, .pdf, .png, .jpg)',
        type=['csv', 'xlsx', 'txt', 'docx', 'pdf', 'png', 'jpg'],
        key='file_uploader'
    )

    if uploaded_file:
        try:
            file_bytes = uploaded_file.read()
            summary, analysis, vis_message = st.session_state.agent.load_data(uploaded_file.name, file_bytes)
            st.success('✅ File processed successfully!')
            
            st.markdown('<div class="card"><h3 class="text-lg font-semibold">🌟 Data Quality Score</h3></div>', unsafe_allow_html=True)
            score = st.session_state.agent.data_quality_score
            emoji = "🥳" if score >= 80 else "😊" if score >= 60 else "⚠️"
            st.write(f"{emoji} Score: {score}/100")
            
            if st.session_state.agent.cleaning_suggestions:
                st.markdown('<div class="card"><h3 class="text-lg font-semibold">🧹 Data Cleaning Suggestions</h3></div>', unsafe_allow_html=True)
                for suggestion in st.session_state.agent.cleaning_suggestions:
                    st.write(f"- {suggestion}")
            
            st.markdown('<div class="card"><h3 class="text-lg font-semibold">📈 Interactive Data Explorer</h3></div>', unsafe_allow_html=True)
            edited_df = st.data_editor(st.session_state.agent.data, use_container_width=True, num_rows="dynamic")
            if not edited_df.equals(st.session_state.agent.data):
                selected_rows = edited_df.index.tolist()
                if selected_rows:
                    selected_data = st.session_state.agent.data.loc[selected_rows]
                    st.write("📋 Select columns for custom plot:")
                    x_col = st.selectbox('X-Axis', selected_data.columns, key='explorer_x')
                    y_col = st.selectbox('Y-Axis', selected_data.columns, key='explorer_y')
                    if st.button('✨ Generate Custom Plot', key='custom_plot'):
                        fig = generate_visualization(selected_data, 'scatter', x_col, y_col)
                        if fig:
                            st.write("🎉 Plot generated!")
                            st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.agent.insights:
                st.markdown('<div class="card"><h3 class="text-lg font-semibold">🌟 Insight Cards</h3></div>', unsafe_allow_html=True)
                for insight in st.session_state.agent.insights:
                    st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card"><h3 class="text-lg font-semibold">📝 Summary</h3></div>', unsafe_allow_html=True)
            st.text_area('Data Summary', summary, height=200)
            st.markdown('<div class="card"><h3 class="text-lg font-semibold">🔍 Analysis</h3></div>', unsafe_allow_html=True)
            st.text_area('Data Analysis', analysis, height=300)
            
            st.markdown('<div class="card"><h3 class="text-lg font-semibold">📊 Visualizations</h3></div>', unsafe_allow_html=True)
            st.write(f"🎨 {vis_message}")
            for vis_type, fig in st.session_state.agent.visualizations:
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="card"><h3 class="text-lg font-semibold">🎯 Feature Importance</h3></div>', unsafe_allow_html=True)
            target_col = st.selectbox('Select Target Column', st.session_state.agent.data.columns, key='feature_target')
            if st.button('🔍 Compute Feature Importance', key='feature_importance'):
                fig, message = st.session_state.agent.compute_feature_importance(target_col)
                st.write(f"✅ {message}")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="card"><h3 class="text-lg font-semibold">📊 Comparative Analysis</h3></div>', unsafe_allow_html=True)
            st.write("🎯 Select grouping and metric:")
            group_col = st.selectbox('Grouping Column', st.session_state.agent.data.columns, key='compare_group')
            metric = st.selectbox('Metric to Compare', st.session_state.agent.data.select_dtypes(include=np.number).columns, key='compare_metric')
            group_values = st.session_state.agent.data[group_col].unique()
            selected_groups = st.multiselect('Groups to Compare', group_values, default=group_values[:2], key='compare_groups')
            if st.button('🔍 Compare Groups', key='compare_button'):
                compare_data = st.session_state.agent.data[st.session_state.agent.data[group_col].isin(selected_groups)]
                fig = generate_visualization(
                    compare_data, 'compare', extra_params={'data': compare_data, 'group_col': group_col, 'metric': metric}
                )
                if fig:
                    st.write("✅ Comparison generated!")
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="card"><h3 class="text-lg font-semibold">📄 Custom Report Builder</h3></div>', unsafe_allow_html=True)
            report_title = st.text_input('Report Title', 'Data Analysis Report', key='report_title')
            selected_components = st.multiselect(
                'Select Report Components',
                ['summary', 'analysis', 'insights', 'visualizations'],
                default=['summary', 'analysis', 'insights'],
                key='report_components'
            )
            if st.button('📝 Generate Custom Report', key='custom_report'):
                pdf_buffer = generate_custom_pdf_report(
                    summary, analysis, st.session_state.agent.insights,
                    st.session_state.agent.visualizations, selected_components, report_title
                )
                st.download_button(
                    label='⬇️ Download Custom Report as PDF',
                    data=pdf_buffer,
                    file_name=f'{report_title.lower().replace(" ", "_")}.pdf',
                    mime='application/pdf',
                    key='download_custom_pdf'
                )

            # AI Agent Feature with Updated Button Label
            st.markdown('<div class="card"><h3 class="text-lg font-semibold">🤖 AI Agent: ZANE</h3></div>', unsafe_allow_html=True)
            description_button = st.button('🎨 Data Description', key='generate_ai_description')
            
            if description_button:
                if st.session_state.agent.data_type == 'dataframe':
                    # Prepare a summary of the data for the API
                    data_summary = f"Dataset has {len(st.session_state.agent.data)} rows and columns: {list(st.session_state.agent.data.columns)}. "
                    numeric_cols = st.session_state.agent.data.select_dtypes(include=np.number).columns
                    categorical_cols = st.session_state.agent.data.select_dtypes(include=['object', 'category']).columns
                    for col in numeric_cols:
                        data_summary += f"{col} (numeric): Mean={st.session_state.agent.data[col].mean():.2f}, Max={st.session_state.agent.data[col].max():.2f}. "
                    for col in categorical_cols:
                        top_value = st.session_state.agent.data[col].value_counts().idxmax()
                        data_summary += f"{col} (categorical): Most common={top_value}. "
                    # Query Together AI for a creative description
                    prompt = f"Provide a creative description of the following dataset in a digital art cartoonist style: {data_summary}"
                    description = query_together_ai(prompt, max_tokens=200)
                    st.session_state.ai_agent_description = description
                else:
                    st.session_state.ai_agent_description = "ZANE can only describe tabular datasets. Please upload a CSV or Excel file."

            if st.session_state.ai_agent_description:
                st.markdown('<div class="card"><h3 class="text-lg font-semibold">✨ ZANE’s Description</h3></div>', unsafe_allow_html=True)
                st.write(st.session_state.ai_agent_description)

        except Exception as e:
            st.error(f'Error processing file: {str(e)}')

with col2:
    st.markdown('<div class="card"><h2 class="text-xl font-semibold">💬 Interact with Data</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><h3 class="text-lg font-semibold">❓ Ask Questions</h3></div>', unsafe_allow_html=True)
    question = st.text_input('Ask a question about the data (e.g., "What is the mean of Annual_Fees?")', key='question_input', value=st.session_state.selected_question)
    
    if st.session_state.agent.data is not None:
        suggestions = []
        query_types = ['mean', 'max', 'min']
        numeric_cols = st.session_state.agent.data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = st.session_state.agent.data.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = st.session_state.agent.data.columns.tolist()
        for col in numeric_cols:
            for q_type in query_types:
                suggestions.append(f"{q_type} {col}")
        for col in all_cols:
            suggestions.append(f"count {col}")
        for col in categorical_cols:
            suggestions.append(f"best {col}")
        suggestions = suggestions[:10]
        
        if suggestions:
            st.markdown('<div class="card"><h3 class="text-lg font-semibold">💡 Suggested Questions</h3></div>', unsafe_allow_html=True)
            selected_suggestion = st.selectbox('Select a suggested question or type your own above', [''] + suggestions, key='suggestion_select')
            if selected_suggestion and selected_suggestion != st.session_state.selected_question:
                st.session_state.selected_question = selected_suggestion
                st.rerun()
    
    if question:
        answer = st.session_state.agent.answer_question(question)
        st.markdown('<div class="card"><h3 class="text-lg font-semibold">✅ Answer</h3></div>', unsafe_allow_html=True)
        st.write(answer)

    if st.session_state.agent.chat_history:
        st.markdown('<div class="card"><h3 class="text-lg font-semibold">🕒 Chat History</h3></div>', unsafe_allow_html=True)
        for i, (q, a) in enumerate(st.session_state.agent.chat_history[::-1]):
            with st.expander(f'Q: {q[:50]}...', expanded=i == 0):
                st.write(f'**Question:** {q}')
                st.write(f'**Answer:** {a}')

if st.button('🗑️ Clear Session', key='clear_session'):
    st.session_state.agent = DataAnalystAgent()
    st.session_state.file_uploader = None
    st.session_state.selected_question = ''
    st.session_state.ai_agent_description = ''
    st.success('✅ Session cleared!')
"""
SentinelAI - Network Intrusion Detection System
Streamlit Application

This module provides the Streamlit-based web UI for:
1. Loading network flow data
2. Visualizing network traffic
3. Detecting and visualizing anomalies
4. Explaining anomalies to security analysts
5. Real-time simulation of network traffic

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import json
import io
import os
import base64
import matplotlib.pyplot as plt
import altair as alt
import joblib
import threading
import queue
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Import the data generator
from data_generator import DataGenerator, explain_anomaly, extract_key_insights

# Set up page
st.set_page_config(page_title="SentinelAI - Network Intrusion Detection", layout="wide", initial_sidebar_state="expanded")

# Create a global queue for real-time flow updates
flow_queue = queue.Queue(maxsize=100)

# Custom CSS
st.markdown("""
<style>
    .anomaly-high {
        color: #F63366;
        font-weight: bold;
        background-color: rgba(246, 51, 102, 0.1);
        padding: 2px 6px;
        border-radius: 3px;
    }
    .anomaly-medium {
        color: #FF9F1C;
        font-weight: bold;
        background-color: rgba(255, 159, 28, 0.1);
        padding: 2px 6px;
        border-radius: 3px;
    }
    .alert-box {
        background-color: #F8F9FA;
        border-left: 5px solid #F63366;
        padding: 15px;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .metrics-box {
        background-color: #F8F9FA;
        border-left: 5px solid #4D96FF;
        padding: 15px;
        border-radius: 3px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 15px;
    }
</style>
""", unsafe_allow_html=True)

def create_flow_simulator(df, speed_factor, flow_queue):
    """Create a flow generator that simulates real-time flow data"""
    def simulate_flows():
        # Get flow data from dataframe
        flows = df.copy()
        flows = flows.sort_values('timestamp')

        # Get the first timestamp
        start_time = flows.iloc[0]['timestamp']

        # Get the current time as the base
        real_start_time = datetime.datetime.now()

        # Process each flow
        for i, flow in flows.iterrows():
            # Calculate the time to wait
            time_diff = (flow['timestamp'] - start_time).total_seconds() / speed_factor

            # Sleep until it's time to send this flow
            elapsed = (datetime.datetime.now() - real_start_time).total_seconds()
            if time_diff > elapsed:
                time.sleep(time_diff - elapsed)

            # Add to queue
            try:
                flow_queue.put(flow.to_dict(), block=False)
            except queue.Full:
                # If queue is full, remove oldest item
                try:
                    flow_queue.get(block=False)
                    flow_queue.put(flow.to_dict(), block=False)
                except queue.Empty:
                    pass

    return threading.Thread(target=simulate_flows)

def get_download_link(df, filename="sentinel_ai_data.csv", text="Download CSV"):
    """Generate a link to download the dataframe as a CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def plot_roc_curve(y_true, y_scores_dict):
    """Plot ROC curve for multiple models"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, y_score in y_scores_dict.items():
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    return fig

def plot_precision_recall_curve(y_true, y_scores_dict):
    """Plot Precision-Recall curve for multiple models"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, y_score in y_scores_dict.items():
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        pr_auc = metrics.auc(recall, precision)
        ax.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")

    return fig

def load_models():
    """Load trained models from disk if available"""
    models = {}
    models_dir = 'models'

    try:
        if os.path.exists(os.path.join(models_dir, 'isolation_forest.joblib')):
            models['iso'] = joblib.load(os.path.join(models_dir, 'isolation_forest.joblib'))

        if (os.path.exists(os.path.join(models_dir, 'ocsvm.joblib')) and
                os.path.exists(os.path.join(models_dir, 'ocsvm_scaler.joblib'))):
            models['ocsvm'] = joblib.load(os.path.join(models_dir, 'ocsvm.joblib'))
            models['ocsvm_scaler'] = joblib.load(os.path.join(models_dir, 'ocsvm_scaler.joblib'))

        if (os.path.exists(os.path.join(models_dir, 'dbscan.joblib')) and
                os.path.exists(os.path.join(models_dir, 'dbscan_scaler.joblib'))):
            models['dbscan'] = joblib.load(os.path.join(models_dir, 'dbscan.joblib'))
            models['dbscan_scaler'] = joblib.load(os.path.join(models_dir, 'dbscan_scaler.joblib'))

        if os.path.exists(os.path.join(models_dir, 'model_metrics.json')):
            with open(os.path.join(models_dir, 'model_metrics.json'), 'r') as f:
                models['metrics'] = json.load(f)

    except Exception as e:
        st.error(f"Error loading models: {e}")

    return models

# Sidebar
with st.sidebar:
    st.title("🔒 SentinelAI")
    st.markdown("### Network Intrusion Detection System")

    # Input methods
    st.header("Data Source")
    data_source = st.radio(
        "Choose data source",
        ["Generate New Data", "Upload CSV", "Use Saved Data"]
    )

    if data_source == "Generate New Data":
        st.header("Data Generation")
        n_normal = st.slider("Normal flows", 100, 10000, 2000, 100)
        n_attack = st.slider("Attack flows", 10, 5000, 500, 10)

        attack_types = st.multiselect(
            "Attack types to generate",
            ["port_scan", "brute_force", "data_exfiltration", "dos"],
            ["port_scan", "brute_force", "data_exfiltration", "dos"]
        )

    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload network flow CSV", type=["csv"])

    # Model settings
    st.header("Model Settings")
    iso_contamination = st.slider("IsolationForest contamination", 0.01, 0.5, 0.1, 0.01)
    ocsvm_nu = st.slider("OneClassSVM ν", 0.01, 0.5, 0.1, 0.01)
    dbscan_eps = st.slider("DBSCAN epsilon", 0.1, 2.0, 0.5, 0.1)
    dbscan_min_samples = st.slider("DBSCAN min samples", 2, 20, 5, 1)

    st.header("Alert Settings")
    anomaly_threshold = st.slider("Anomaly score threshold", 0.0, 1.0, 0.8, 0.05)

    st.header("Simulation Settings")
    enable_simulation = st.checkbox("Enable real-time simulation", value=False)
    if enable_simulation:
        speed_factor = st.slider("Simulation speed factor", 1.0, 100.0, 10.0, 1.0)

    generate_button = st.button("Process Data", use_container_width=True)

# Main content
st.title("🔒 SentinelAI - Network Intrusion Detection")

# Initialize the session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'simulation_running' not in st.session_state:
    st.session_state['simulation_running'] = False
if 'models' not in st.session_state:
    st.session_state['models'] = {}

# Process data when button is clicked
if generate_button:
    if data_source == "Generate New Data":
        with st.spinner("Generating synthetic network flows..."):
            # Create data generator
            generator = DataGenerator()

            # Generate synthetic data
            df = generator.generate_synthetic_flows(n_normal, n_attack, attack_types)

            # Train models
            iso_model = generator.train_isolation_forest(df, iso_contamination)
            ocsvm_model, ocsvm_scaler = generator.train_ocsvm(df, ocsvm_nu)
            dbscan_model, dbscan_scaler = generator.train_dbscan(df, dbscan_eps, dbscan_min_samples)

            # Score flows
            df = generator.score_flows(df, iso_model, ocsvm_model, ocsvm_scaler, dbscan_model, dbscan_scaler)

            # Store models
            st.session_state['models'] = {
                'iso': iso_model,
                'ocsvm': ocsvm_model,
                'ocsvm_scaler': ocsvm_scaler,
                'dbscan': dbscan_model,
                'dbscan_scaler': dbscan_scaler
            }

            # Store dataframe
            st.session_state['df'] = df
            st.session_state['simulation_running'] = False

            st.success(f"Generated {len(df)} flows ({n_normal} normal, {n_attack} attack)")

    elif data_source == "Upload CSV":
        if uploaded_file is not None:
            with st.spinner("Processing uploaded data..."):
                # Load CSV
                df = pd.read_csv(uploaded_file)

                # Check if required columns exist
                required_columns = ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port',
                                    'protocol', 'duration_ms', 'total_bytes', 'label']

                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    # Parse timestamp if needed
                    if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                    # Create data generator
                    generator = DataGenerator()

                    # Apply feature engineering if needed
                    features = generator.get_feature_columns()
                    missing_features = [col for col in features if col not in df.columns]

                    if missing_features:
                        st.info(f"Applying feature engineering to generate {len(missing_features)} missing features...")
                        df = generator._engineer_features(df)

                    # Check if model scores exist
                    if ('score_iso_norm' not in df.columns or
                            'score_ocsvm_norm' not in df.columns or
                            'score_dbscan_norm' not in df.columns):
                        st.info("Training models and scoring flows...")

                        # Train models
                        iso_model = generator.train_isolation_forest(df, iso_contamination)
                        ocsvm_model, ocsvm_scaler = generator.train_ocsvm(df, ocsvm_nu)
                        dbscan_model, dbscan_scaler = generator.train_dbscan(df, dbscan_eps, dbscan_min_samples)

                        # Score flows
                        df = generator.score_flows(df, iso_model, ocsvm_model, ocsvm_scaler, dbscan_model, dbscan_scaler)

                        # Store models
                        st.session_state['models'] = {
                            'iso': iso_model,
                            'ocsvm': ocsvm_model,
                            'ocsvm_scaler': ocsvm_scaler,
                            'dbscan': dbscan_model,
                            'dbscan_scaler': dbscan_scaler
                        }

                    # Store dataframe
                    st.session_state['df'] = df
                    st.session_state['simulation_running'] = False

                    st.success(f"Processed {len(df)} flows")

    elif data_source == "Use Saved Data":
        with st.spinner("Loading saved data..."):
            # Check if network_flows.csv exists
            if os.path.exists('network_flows.csv'):
                # Load CSV
                df = pd.read_csv('network_flows.csv')

                # Parse timestamp
                if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Load models
                models = load_models()

                if models:
                    st.session_state['models'] = models

                # Store dataframe
                st.session_state['df'] = df
                st.session_state['simulation_running'] = False

                st.success(f"Loaded {len(df)} flows from saved data")
            else:
                st.error("No saved data found. Generate new data first.")

# If data is available, show the analysis interface
if st.session_state['df'] is not None:
    df = st.session_state['df']

    # Get feature columns
    generator = DataGenerator()
    features = generator.get_feature_columns()

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Flow Analysis",
        "🚨 Anomaly Detection",
        "🔍 Explanations",
        "⚙️ Simulation"
    ])

    # Tab 1: Overview
    with tab1:
        # Summary statistics
        st.header("Dataset Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Flows", len(df))
        with col2:
            st.metric("Normal Flows", len(df[df['label'] == 'normal']))
        with col3:
            st.metric("Attack Flows", len(df[df['label'] == 'attack']))
        with col4:
            detection_rate = len(df[(df['pred_iso'] == 1) & (df['label'] == 'attack')]) / max(1, len(df[df['label'] == 'attack']))
            st.metric("Detection Rate (IF)", f"{detection_rate:.1%}")

        # Metrics comparison
        st.header("Model Performance")

        # Calculate metrics for each model
        y_true = (df['label'] == 'attack').astype(int)
        metrics_list = []

        for name, pred_col, score_col in [
            ('IsolationForest', 'pred_iso', 'score_iso_norm'),
            ('OneClassSVM', 'pred_ocsvm', 'score_ocsvm_norm'),
            ('DBSCAN', 'pred_dbscan', 'score_dbscan_norm')
        ]:
            try:
                roc = metrics.roc_auc_score(y_true, df[score_col])
            except:
                roc = np.nan

            try:
                prec, rec, f1, _ = metrics.precision_recall_fscore_support(
                    y_true, df[pred_col], average='binary', zero_division=0
                )
                metrics_list.append({
                    'Model': name,
                    'ROC AUC': roc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1 Score': f1
                })
            except:
                metrics_list.append({
                    'Model': name,
                    'ROC AUC': roc,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'F1 Score': np.nan
                })

        metrics_df = pd.DataFrame(metrics_list)

        st.dataframe(metrics_df.style.format({
            'ROC AUC': "{:.3f}", 'Precision': "{:.3f}", 'Recall': "{:.3f}", 'F1 Score': "{:.3f}"
        }))

        # Confusion matrices
        st.header("Confusion Matrices")

        cols = st.columns(3)
        for i, (name, pred_col) in enumerate([
            ('IsolationForest', 'pred_iso'),
            ('OneClassSVM', 'pred_ocsvm'),
            ('DBSCAN', 'pred_dbscan')
        ]):
            with cols[i]:
                st.subheader(name)
                try:
                    cm = metrics.confusion_matrix(y_true, df[pred_col])
                    cm_df = pd.DataFrame(
                        cm,
                        index=['True Normal', 'True Attack'],
                        columns=['Pred Normal', 'Pred Attack']
                    )
                    st.dataframe(cm_df)

                    # Calculate metrics
                    tn, fp, fn, tp = cm.ravel()
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                    st.markdown(f"""
                    * **Accuracy**: {accuracy:.3f}
                    * **Precision**: {precision:.3f}
                    * **Recall**: {recall:.3f}
                    """)
                except:
                    st.warning(f"Could not calculate confusion matrix for {name}")

        # ROC and precision-recall curves
        st.header("Model Comparison Curves")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            y_scores_dict = {
                'IsolationForest': df['score_iso_norm'],
                'OneClassSVM': df['score_ocsvm_norm'],
                'DBSCAN': df['score_dbscan_norm']
            }
            fig_roc = plot_roc_curve(y_true, y_scores_dict)
            st.pyplot(fig_roc)

        with col2:
            st.subheader("Precision-Recall Curve")
            fig_pr = plot_precision_recall_curve(y_true, y_scores_dict)
            st.pyplot(fig_pr)

        # Attack type distribution
        if 'attack_type' in df.columns:
            st.header("Attack Type Distribution")

            attack_counts = df[df['label'] == 'attack']['attack_type'].value_counts().reset_index()
            attack_counts.columns = ['attack_type', 'count']

            bar = alt.Chart(attack_counts).mark_bar().encode(
                x=alt.X('attack_type:N', title='Attack Type'),
                y=alt.Y('count:Q', title='Count'),
                color=alt.Color('attack_type:N', legend=None)
            ).properties(
                width=600,
                height=300
            )

            st.altair_chart(bar, use_container_width=True)

        # Download data
        st.header("Download Dataset")
        st.markdown(get_download_link(df), unsafe_allow_html=True)

    # Tab 2: Flow Analysis
    with tab2:
        st.header("Network Flow Analysis")

        # Flow filtering
        col1, col2 = st.columns(2)
        with col1:
            filter_label = st.selectbox("Filter by label", ["All", "Normal", "Attack"])
        with col2:
            if filter_label == "Attack" and 'attack_type' in df.columns:
                filter_attack = st.selectbox("Filter by attack type", ["All"] + sorted(df['attack_type'].dropna().unique().tolist()))
            else:
                filter_attack = "All"

        # Apply filters
        filtered_df = df.copy()
        if filter_label != "All":
            filtered_df = filtered_df[filtered_df['label'].str.lower() == filter_label.lower()]

        if filter_attack != "All" and filter_label == "Attack" and 'attack_type' in df.columns:
            filtered_df = filtered_df[filtered_df['attack_type'] == filter_attack]

        # Feature distributions
        st.subheader("Feature Distributions")

        dist_feature = st.selectbox(
            "Select feature to visualize",
            ["duration_ms", "total_bytes", "packet_count", "byte_rate", "packets_per_second", "bytes_per_packet"]
        )

        # Scale feature for better visualization
        scale_log = st.checkbox("Log scale", value=True)

        if scale_log and (filtered_df[dist_feature] > 0).all():
            filtered_df[f"{dist_feature}_log"] = np.log1p(filtered_df[dist_feature])
            plot_feature = f"{dist_feature}_log"
            title_suffix = " (log scale)"
        else:
            plot_feature = dist_feature
            title_suffix = ""

        # Create histogram
        hist = alt.Chart(filtered_df).mark_bar().encode(
            alt.X(f"{plot_feature}:Q", title=f"{dist_feature}{title_suffix}", bin=alt.Bin(maxbins=50)),
            alt.Y('count()', title='Count'),
            alt.Color('label:N', title='Label')
        ).properties(
            width=600,
            height=300,
            title=f"Distribution of {dist_feature}{title_suffix}"
        )

        st.altair_chart(hist, use_container_width=True)

        # Protocol distribution
        st.subheader("Protocol Distribution")

        protocol_counts = filtered_df['protocol'].value_counts().reset_index()
        protocol_counts.columns = ['protocol', 'count']

        pie = alt.Chart(protocol_counts).mark_arc().encode(
            theta=alt.Theta('count:Q'),
            color=alt.Color('protocol:N', scale=alt.Scale(scheme='category10')),
            tooltip=['protocol', 'count']
        ).properties(
            width=400,
            height=400,
            title="Protocol Distribution"
        )

        st.altair_chart(pie, use_container_width=True)

        # Time series
        st.subheader("Traffic Over Time")

        # Resample by minute
        filtered_df['minute'] = filtered_df['timestamp'].dt.floor('min')
        time_series = filtered_df.groupby(['minute', 'label']).size().reset_index(name='count')

        line = alt.Chart(time_series).mark_line().encode(
            x=alt.X('minute:T', title='Time'),
            y=alt.Y('count:Q', title='Flow Count'),
            color=alt.Color('label:N', title='Label')
        ).properties(
            width=600,
            height=300,
            title="Flow Count Over Time"
        )

        st.altair_chart(line, use_container_width=True)

        # Port heatmap
        st.subheader("Destination Port Analysis")

        # Get top ports
        top_ports = filtered_df['dst_port'].value_counts().head(20).index.tolist()
        port_df = filtered_df[filtered_df['dst_port'].isin(top_ports)]

        heatmap = alt.Chart(port_df).mark_rect().encode(
            x=alt.X('dst_port:O', title='Destination Port'),
            y=alt.Y('label:N', title='Label'),
            color=alt.Color('count()', title='Count', scale=alt.Scale(scheme='viridis')),
            tooltip=['dst_port', 'label', 'count()']
        ).properties(
            width=600,
            height=200,
            title="Destination Port Usage by Label"
        )

        st.altair_chart(heatmap, use_container_width=True)

    # Tab 3: Anomaly Detection
    with tab3:
        st.header("Anomaly Detection Results")

        # Model selection
        model_choice = st.selectbox(
            "Select anomaly detection model",
            ["IsolationForest", "OneClassSVM", "DBSCAN", "Ensemble (Majority Vote)"]
        )

        if model_choice == "IsolationForest":
            score_col = 'score_iso_norm'
            pred_col = 'pred_iso'
            model = st.session_state['models'].get('iso')
        elif model_choice == "OneClassSVM":
            score_col = 'score_ocsvm_norm'
            pred_col = 'pred_ocsvm'
            model = st.session_state['models'].get('ocsvm')
        elif model_choice == "DBSCAN":
            score_col = 'score_dbscan_norm'
            pred_col = 'pred_dbscan'
            model = st.session_state['models'].get('dbscan')
        else:  # Ensemble
            # Create ensemble prediction (majority vote)
            df['pred_ensemble'] = (
                (df['pred_iso'] + df['pred_ocsvm'] + df['pred_dbscan'] >= 2)
            ).astype(int)

            # Create ensemble score (average of normalized scores)
            df['score_ensemble_norm'] = (
                                                df['score_iso_norm'] + df['score_ocsvm_norm'] + df['score_dbscan_norm']
                                        ) / 3

            score_col = 'score_ensemble_norm'
            pred_col = 'pred_ensemble'
            model = None  # No single model for ensemble

        # Anomaly thresholding
        threshold = st.slider(
            "Anomaly score threshold for alerts",
            min_value=0.0,
            max_value=1.0,
            value=anomaly_threshold,
            step=0.05,
            key="anomaly_threshold_slider"
        )

        # Apply threshold to create alert flag
        df['is_alert'] = (df[score_col] >= threshold).astype(int)

        # Show metrics at current threshold
        st.subheader(f"Performance at threshold {threshold:.2f}")

        col1, col2, col3, col4 = st.columns(4)

        # Calculate metrics
        y_pred = df['is_alert']
        try:
            prec = metrics.precision_score(y_true, y_pred)
            rec = metrics.recall_score(y_true, y_pred)
            f1 = metrics.f1_score(y_true, y_pred)
            acc = metrics.accuracy_score(y_true, y_pred)

            with col1:
                st.metric("Precision", f"{prec:.3f}")
            with col2:
                st.metric("Recall", f"{rec:.3f}")
            with col3:
                st.metric("F1 Score", f"{f1:.3f}")
            with col4:
                st.metric("Accuracy", f"{acc:.3f}")
        except:
            st.warning("Could not calculate metrics at this threshold")

        # Show anomalies
        st.subheader("Detected Anomalies")

        # Apply threshold
        anomalies = df[df[score_col] >= threshold].sort_values(score_col, ascending=False)

        if anomalies.empty:
            st.info("No anomalies detected at the current threshold.")
        else:
            # Style anomalies by severity
            def color_anomaly_score(val):
                if val >= 0.9:
                    return 'background-color: rgba(255,0,0,0.2); color: #B30000; font-weight: bold'
                elif val >= 0.7:
                    return 'background-color: rgba(255,165,0,0.2); color: #CC5500; font-weight: bold'
                else:
                    return 'background-color: rgba(255,255,0,0.1); color: #999900'

            # Show high-level anomaly table
            display_cols = ['flow_id', 'timestamp', 'src_ip', 'dst_ip', 'dst_port', 'protocol',
                            'duration_ms', 'total_bytes', 'label']

            if 'attack_type' in anomalies.columns:
                display_cols.append('attack_type')

            display_cols.append(score_col)

            st.dataframe(
                anomalies[display_cols]
                .rename(columns={score_col: 'anomaly_score'})
                .style.format({'anomaly_score': '{:.3f}'})
                .applymap(color_anomaly_score, subset=['anomaly_score'])
            )

            # Alert summary
            st.subheader("Alert Summary")

            col1, col2 = st.columns(2)

            with col1:
                true_pos = len(anomalies[anomalies['label'] == 'attack'])
                false_pos = len(anomalies[anomalies['label'] == 'normal'])
                st.metric("True Positive Alerts", true_pos)
                st.metric("False Positive Alerts", false_pos)

            with col2:
                # Attack type distribution in alerts
                if 'attack_type' in anomalies.columns:
                    attack_distribution = anomalies[anomalies['label'] == 'attack']['attack_type'].value_counts()
                    st.markdown("**Attack Types Detected:**")
                    for attack_type, count in attack_distribution.items():
                        st.markdown(f"- {attack_type}: {count}")

        # Anomaly score distribution
        st.subheader("Anomaly Score Distribution")

        hist = alt.Chart(df).mark_bar().encode(
            alt.X(f"{score_col}:Q", title="Anomaly Score", bin=alt.Bin(maxbins=50)),
            alt.Y('count()', title="Count"),
            alt.Color('label:N', title="Label")
        ).properties(
            width=600,
            height=300,
            title=f"Distribution of {model_choice} Anomaly Scores"
        )

        # Add threshold line
        threshold_line = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(
            color='red',
            strokeDash=[3, 3]
        ).encode(
            x='threshold:Q'
        )

        st.altair_chart(hist + threshold_line, use_container_width=True)

        # Scatter plot of key features
        st.subheader("Feature Relationships")

        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis feature", features, index=features.index('byte_rate'))
        with col2:
            y_feature = st.selectbox("Y-axis feature", features, index=features.index('duration_ms'))

        scatter = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X(f"{x_feature}:Q", title=x_feature),
            y=alt.Y(f"{y_feature}:Q", title=y_feature),
            color=alt.Color(f"{score_col}:Q", title="Anomaly Score", scale=alt.Scale(scheme='inferno')),
            tooltip=['flow_id', 'src_ip', 'dst_ip', 'protocol', x_feature, y_feature, score_col, 'label']
        ).properties(
            width=700,
            height=400,
            title=f"Relationship between {x_feature} and {y_feature} with Anomaly Scores"
        )

        st.altair_chart(scatter, use_container_width=True)

    # Tab 4: Explanations
    with tab4:
        st.header("Anomaly Explanations")

        # Select model for explanations
        model_choice_exp = st.selectbox(
            "Select model for explanations",
            ["IsolationForest", "OneClassSVM", "DBSCAN"],
            key="explanation_model"
        )

        if model_choice_exp == "IsolationForest":
            score_col_exp = 'score_iso_norm'
            model_exp = st.session_state['models'].get('iso')
        elif model_choice_exp == "OneClassSVM":
            score_col_exp = 'score_ocsvm_norm'
            model_exp = st.session_state['models'].get('ocsvm')
        else:  # DBSCAN
            score_col_exp = 'score_dbscan_norm'
            model_exp = st.session_state['models'].get('dbscan')

        # Get top anomalies
        top_anomalies = df.sort_values(score_col_exp, ascending=False).head(10)

        if top_anomalies.empty:
            st.info("No anomalies to explain.")
        else:
            # Select an anomaly to explain
            selected_flow_id = st.selectbox(
                "Select a flow to explain",
                top_anomalies['flow_id'].tolist(),
                format_func=lambda x: f"Flow {x} (Score: {float(top_anomalies[top_anomalies['flow_id'] == x][score_col_exp].iloc[0]):.3f})"
            )

            # Get the selected flow
            selected_flow = df[df['flow_id'] == selected_flow_id].iloc[0]

            # Display flow details
            st.subheader("Flow Details")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Source:** {selected_flow['src_ip']}:{selected_flow['src_port']}")
                st.markdown(f"**Destination:** {selected_flow['dst_ip']}:{selected_flow['dst_port']}")
                st.markdown(f"**Protocol:** {selected_flow['protocol']}")

            with col2:
                st.markdown(f"**Duration:** {selected_flow['duration_ms']:.2f} ms")
                st.markdown(f"**Bytes:** {selected_flow['total_bytes']} bytes")
                if 'packet_count' in selected_flow:
                    st.markdown(f"**Packets:** {selected_flow['packet_count']} packets")

            with col3:
                st.markdown(f"**Byte Rate:** {selected_flow['byte_rate']:.2f} bytes/ms")
                if 'packets_per_second' in selected_flow:
                    st.markdown(f"**Packet Rate:** {selected_flow['packets_per_second']:.2f} packets/sec")
                st.markdown(f"**Anomaly Score:** {selected_flow[score_col_exp]:.3f}")

            # Generate explanation
            if model_exp is not None:
                explanation = explain_anomaly(model_exp, selected_flow, features)

                # Extract insights
                insights = extract_key_insights(explanation, top_n=5)

                # Display insights
                st.subheader("Key Insights")

                for i, insight in enumerate(insights):
                    st.markdown(f"{i+1}. {insight}")

                # Feature importance visualization
                st.subheader("Feature Contributions")

                # Get top contributing features
                top_features = explanation.head(10)

                # Create bar chart
                bar = alt.Chart(top_features).mark_bar().encode(
                    x=alt.X('contribution:Q', title='Contribution to Anomaly'),
                    y=alt.Y('feature:N', title='Feature', sort='-x'),
                    color=alt.Color('contribution:Q', scale=alt.Scale(scheme='reds')),
                    tooltip=['feature', 'value', 'contribution']
                ).properties(
                    width=700,
                    height=400,
                    title="Top Features Contributing to Anomaly"
                )

                st.altair_chart(bar, use_container_width=True)

                # Comparison to normal traffic
                st.subheader("Comparison to Normal Traffic")

                # Select top features to compare
                compare_features = explanation.head(5)['feature'].tolist()

                # Prepare data for comparison
                compare_data = []
                for feature in compare_features:
                    # Get stats for normal traffic
                    normal_mean = df[df['label'] == 'normal'][feature].mean()
                    normal_std = df[df['label'] == 'normal'][feature].std()

                    # Get value for anomaly
                    anomaly_value = selected_flow[feature]

                    # Calculate z-score
                    z_score = (anomaly_value - normal_mean) / max(normal_std, 1e-5)  # Avoid division by zero

                    compare_data.append({
                        'feature': feature,
                        'normal_mean': normal_mean,
                        'anomaly_value': anomaly_value,
                        'z_score': z_score
                    })

                compare_df = pd.DataFrame(compare_data)

                # Create comparison chart
                comparison = alt.Chart(compare_df).mark_bar().encode(
                    x=alt.X('z_score:Q', title='Standard Deviations from Mean'),
                    y=alt.Y('feature:N', title='Feature', sort='-x'),
                    color=alt.Color('z_score:Q', scale=alt.Scale(domain=[-3, 3], scheme='redblue')),
                    tooltip=['feature', 'normal_mean', 'anomaly_value', 'z_score']
                ).properties(
                    width=700,
                    height=300,
                    title="How Anomalous Features Compare to Normal Traffic (Z-Score)"
                )

                st.altair_chart(comparison, use_container_width=True)
            else:
                st.warning("Model not available for explanation. Please retrain models.")

    # Tab 5: Simulation
    with tab5:
        st.header("Real-Time Network Simulation")

        if enable_simulation:
            # Start the simulation if not already running
            if not st.session_state.get('simulation_running', False):
                # Create and start the simulator
                simulator = create_flow_simulator(df, speed_factor, flow_queue)
                simulator.daemon = True
                simulator.start()
                st.session_state['simulation_running'] = True
                st.success("Simulation started! Showing real-time network flows.")

            # Container for real-time updates
            flow_container = st.empty()
            alert_container = st.empty()
            chart_container = st.empty()

            # Simulation settings
            col1, col2 = st.columns(2)
            with col1:
                auto_scroll = st.checkbox("Auto-scroll to latest", value=True)
            with col2:
                max_flows = st.number_input("Maximum flows to display", min_value=10, max_value=1000, value=50, step=10)

            # Initialize session state for real-time data
            if 'real_time_flows' not in st.session_state:
                st.session_state['real_time_flows'] = []

            if 'alert_count' not in st.session_state:
                st.session_state['alert_count'] = 0

            # Get new flows from queue
            new_flows = []
            while not flow_queue.empty():
                try:
                    flow = flow_queue.get_nowait()
                    new_flows.append(flow)
                except queue.Empty:
                    break

            # Add to session state
            st.session_state['real_time_flows'].extend(new_flows)

            # Keep only the latest flows
            if len(st.session_state['real_time_flows']) > max_flows:
                st.session_state['real_time_flows'] = st.session_state['real_time_flows'][-max_flows:]

            # Convert to DataFrame
            if st.session_state['real_time_flows']:
                live_df = pd.DataFrame(st.session_state['real_time_flows'])

                # Score the flows using the selected model
                if 'iso' in st.session_state['models'] and model_choice == "IsolationForest":
                    iso_model = st.session_state['models']['iso']
                    live_features = live_df[features]
                    live_df['score'] = -iso_model.decision_function(live_features)

                    # Normalize score
                    min_val = df['score_iso'].min()
                    max_val = df['score_iso'].max()
                    if max_val > min_val:
                        live_df['score_norm'] = (live_df['score'] - min_val) / (max_val - min_val)
                    else:
                        live_df['score_norm'] = 0

                    # Apply threshold
                    live_df['is_alert'] = (live_df['score_norm'] >= threshold).astype(int)

                elif 'ocsvm' in st.session_state['models'] and 'ocsvm_scaler' in st.session_state['models'] and model_choice == "OneClassSVM":
                    ocsvm_model = st.session_state['models']['ocsvm']
                    ocsvm_scaler = st.session_state['models']['ocsvm_scaler']

                    live_features = live_df[features]
                    X_scaled = ocsvm_scaler.transform(live_features)
                    live_df['score'] = -ocsvm_model.decision_function(X_scaled)

                    # Normalize score
                    min_val = df['score_ocsvm'].min()
                    max_val = df['score_ocsvm'].max()
                    if max_val > min_val:
                        live_df['score_norm'] = (live_df['score'] - min_val) / (max_val - min_val)
                    else:
                        live_df['score_norm'] = 0

                    # Apply threshold
                    live_df['is_alert'] = (live_df['score_norm'] >= threshold).astype(int)

                elif 'dbscan' in st.session_state['models'] and 'dbscan_scaler' in st.session_state['models'] and model_choice == "DBSCAN":
                    dbscan_model = st.session_state['models']['dbscan']
                    dbscan_scaler = st.session_state['models']['dbscan_scaler']

                    live_features = live_df[features]
                    X_scaled = dbscan_scaler.transform(live_features)
                    live_df['pred'] = dbscan_model.fit_predict(X_scaled)
                    live_df['score_norm'] = (live_df['pred'] == -1).astype(float)

                    # Apply threshold
                    live_df['is_alert'] = (live_df['score_norm'] >= threshold).astype(int)

                else:  # Ensemble or fallback
                    # Use a simple heuristic if models aren't available
                    if 'byte_rate' in live_df.columns and 'duration_ms' in live_df.columns:
                        # Flag very high byte rates or very short/long durations
                        byte_rate_threshold = df['byte_rate'].quantile(0.95)
                        duration_low = df['duration_ms'].quantile(0.05)
                        duration_high = df['duration_ms'].quantile(0.95)

                        live_df['score_norm'] = 0.0
                        # Set high score for unusual byte rates or durations
                        live_df.loc[live_df['byte_rate'] > byte_rate_threshold, 'score_norm'] = 0.8
                        live_df.loc[live_df['duration_ms'] < duration_low, 'score_norm'] = 0.8
                        live_df.loc[live_df['duration_ms'] > duration_high, 'score_norm'] = 0.7

                        live_df['is_alert'] = (live_df['score_norm'] >= threshold).astype(int)
                    else:
                        # If no features available, don't score
                        live_df['score_norm'] = 0.0
                        live_df['is_alert'] = 0

                # Update alert count
                new_alerts = live_df['is_alert'].sum()
                st.session_state['alert_count'] += new_alerts

                # Display alerts
                with alert_container:
                    alert_col1, alert_col2 = st.columns(2)
                    with alert_col1:
                        st.metric("Total Alerts", st.session_state['alert_count'])
                    with alert_col2:
                        # Get most recent alert
                        if new_alerts > 0:
                            last_alert = live_df[live_df['is_alert'] == 1].iloc[-1]
                            st.markdown(f"""
                            <div class="alert-box">
                                <strong>⚠️ New Alert!</strong><br>
                                Source: {last_alert['src_ip']}:{last_alert['src_port']}<br>
                                Destination: {last_alert['dst_ip']}:{last_alert['dst_port']}<br>
                                Score: {last_alert['score_norm']:.3f}
                            </div>
                            """, unsafe_allow_html=True)

                # Display flows
                with flow_container:
                    st.subheader("Live Network Flows")

                    # Style function for highlighting anomalies
                    def highlight_anomalies(row):
                        if row['is_alert'] == 1:
                            if row['score_norm'] >= 0.9:
                                return ['background-color: rgba(255,0,0,0.2)'] * len(row)
                            elif row['score_norm'] >= 0.7:
                                return ['background-color: rgba(255,165,0,0.2)'] * len(row)
                            else:
                                return ['background-color: rgba(255,255,0,0.1)'] * len(row)
                        return [''] * len(row)

                    # Format score
                    def format_score(val):
                        if val >= 0.9:
                            return f'<span class="anomaly-high">{val:.3f}</span>'
                        elif val >= 0.7:
                            return f'<span class="anomaly-medium">{val:.3f}</span>'
                        else:
                            return f'{val:.3f}'

                    # Display dataframe with styling
                    display_cols = ['timestamp', 'src_ip', 'dst_ip', 'dst_port', 'protocol',
                                    'duration_ms', 'total_bytes']

                    if 'packet_count' in live_df.columns:
                        display_cols.append('packet_count')

                    display_cols.extend(['score_norm', 'label'])

                    if 'attack_type' in live_df.columns:
                        display_cols.append('attack_type')

                    st.dataframe(
                        live_df[display_cols]
                        .sort_values('timestamp', ascending=False)
                        .reset_index(drop=True)
                        .style
                        .apply(highlight_anomalies, axis=1)
                        .format({'score_norm': '{:.3f}', 'duration_ms': '{:.2f}'})
                    )

                # Plot real-time metrics
                with chart_container:
                    st.subheader("Live Metrics")

                    # Prepare time series data
                    live_df['minute'] = pd.to_datetime(live_df['timestamp']).dt.floor('min')
                    time_series = live_df.groupby('minute').agg({
                        'flow_id': 'count',
                        'is_alert': 'sum'
                    }).reset_index()
                    time_series.columns = ['minute', 'total_flows', 'alerts']

                    # Melt for plotting
                    plot_df = pd.melt(
                        time_series,
                        id_vars=['minute'],
                        value_vars=['total_flows', 'alerts'],
                        var_name='metric',
                        value_name='count'
                    )

                    # Create line chart
                    line = alt.Chart(plot_df).mark_line().encode(
                        x=alt.X('minute:T', title='Time'),
                        y=alt.Y('count:Q', title='Count'),
                        color=alt.Color('metric:N', title='Metric')
                    ).properties(
                        width=700,
                        height=300,
                        title="Flow and Alert Rates"
                    )

                    st.altair_chart(line, use_container_width=True)

            else:
                st.info("Waiting for network flows...")

        else:
            st.info("Enable real-time simulation in the sidebar to see live network flows.")

        # Alert configuration
        st.header("Alert Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Alert Thresholds")

            st.markdown("""
            Configure anomaly score thresholds for different severity levels:
            - **High**: Immediately actionable, potential active threat
            - **Medium**: Suspicious activity requiring investigation
            - **Low**: Unusual but potentially benign
            """)

            high_threshold = st.slider("High severity threshold", 0.0, 1.0, 0.9, 0.05)
            medium_threshold = st.slider("Medium severity threshold", 0.0, high_threshold, 0.7, 0.05)
            low_threshold = st.slider("Low severity threshold", 0.0, medium_threshold, 0.5, 0.05)

        with col2:
            st.subheader("Alert Actions")

            st.markdown("Configure actions to take when alerts are triggered:")

            st.checkbox("Log all alerts to file", value=True)
            st.checkbox("Send email for high severity alerts", value=False)
            st.checkbox("Send webhook notifications", value=False)
            webhook_url = st.text_input("Webhook URL (if enabled)")

            st.markdown("Alert retention policy:")
            st.radio(
                "Retention period",
                ["24 hours", "7 days", "30 days", "90 days", "Indefinite"],
                index=2
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>🔒 <b>SentinelAI</b> - Real-Time Explainable Network Intrusion Detection</p>
    <p style="font-size: 0.8em;">Built with Streamlit • Scikit-learn • Pandas • Altair</p>
</div>
""", unsafe_allow_html=True)
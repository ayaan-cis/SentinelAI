"""
SentinelAI - Generate Mock Network Flow Data

This script generates a small sample dataset of network flows
including both normal traffic and various attack patterns.
"""

import pandas as pd
import numpy as np
import datetime
import uuid
import json
import os
from faker import Faker

def generate_mock_data(output_dir='data', models_dir='models'):
    """Generate mock network flow data and metrics"""

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)
    faker = Faker()
    Faker.seed(42)

    # Current timestamp base
    now = datetime.datetime.now()
    records = []

    # Normal flows (50)
    print("Generating normal flows...")
    for i in range(50):
        ts = now + datetime.timedelta(seconds=np.random.randint(0, 3600))
        flags = np.random.choice(['SYN', 'ACK', 'PSH-ACK', 'SYN-ACK', 'RST', 'FIN', 'FIN-ACK', ''])
        proto = np.random.choice(['TCP', 'UDP', 'ICMP'], p=[0.7, 0.25, 0.05])

        # Select realistic destination ports
        if proto == 'TCP':
            dst_port = np.random.choice([80, 443, 8080, 22, 25, 143, 3306, 5432],
                                        p=[0.3, 0.3, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
        elif proto == 'UDP':
            dst_port = np.random.choice([53, 123, 161, 1900, 5353],
                                        p=[0.5, 0.2, 0.1, 0.1, 0.1])
        else:  # ICMP
            dst_port = 0

        # Adjust parameters for different connection types
        if dst_port in [80, 443, 8080]:  # Web traffic
            duration = np.random.exponential(scale=200)
            bytes_count = np.random.randint(100, 15000)
            packet_count = max(1, int(bytes_count / np.random.randint(40, 1500)))
        elif dst_port == 53:  # DNS
            duration = np.random.exponential(scale=20)
            bytes_count = np.random.randint(50, 300)
            packet_count = np.random.randint(1, 3)
        else:  # Other protocols
            duration = np.random.exponential(scale=100)
            bytes_count = np.random.randint(40, 1500)
            packet_count = max(1, int(bytes_count / np.random.randint(40, 100)))

        records.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': faker.ipv4_private(),
            'dst_ip': faker.ipv4_private() if np.random.random() < 0.7 else faker.ipv4_public(),
            'src_port': np.random.randint(1024, 65535),
            'dst_port': dst_port,
            'protocol': proto,
            'duration_ms': duration,
            'total_bytes': bytes_count,
            'packet_count': packet_count,
            'tcp_flags': flags if proto == 'TCP' else '',
            'label': 'normal',
            'attack_type': None,
            'flow_id': str(uuid.uuid4())[:8]
        })

    # Attack flows (10 of each type)
    print("Generating attack flows...")
    attack_types = ['port_scan', 'brute_force', 'data_exfiltration', 'dos']

    # Port scan attacks
    for i in range(3):
        ts = now + datetime.timedelta(seconds=np.random.randint(0, 3600))
        records.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': faker.ipv4_private(),
            'dst_ip': faker.ipv4_private(),
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.randint(1, 1024),  # Scanning low ports
            'protocol': 'TCP',
            'duration_ms': np.random.exponential(scale=5),  # Very short duration
            'total_bytes': np.random.randint(40, 60),  # Small packets
            'packet_count': 1,
            'tcp_flags': 'SYN',  # Typical for port scans
            'label': 'attack',
            'attack_type': 'port_scan',
            'flow_id': str(uuid.uuid4())[:8]
        })

    # Brute force attacks
    for i in range(2):
        ts = now + datetime.timedelta(seconds=np.random.randint(0, 3600))
        records.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': faker.ipv4_private(),
            'dst_ip': faker.ipv4_private(),
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.choice([22, 23, 3389, 5900]),  # SSH, Telnet, RDP, VNC
            'protocol': 'TCP',
            'duration_ms': np.random.exponential(scale=50),
            'total_bytes': np.random.randint(60, 200),
            'packet_count': np.random.randint(1, 3),
            'tcp_flags': np.random.choice(['SYN', 'SYN-ACK', 'ACK']),
            'label': 'attack',
            'attack_type': 'brute_force',
            'flow_id': str(uuid.uuid4())[:8]
        })

    # Data exfiltration attacks
    for i in range(3):
        ts = now + datetime.timedelta(seconds=np.random.randint(0, 3600))
        records.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': faker.ipv4_private(),
            'dst_ip': faker.ipv4_public(),  # External destination
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.choice([80, 443, 53]),  # Common ports for hiding exfiltration
            'protocol': np.random.choice(['TCP', 'UDP']),
            'duration_ms': np.random.exponential(scale=300),
            'total_bytes': np.random.randint(10000, 100000),  # Large data transfer
            'packet_count': np.random.randint(10, 100),
            'tcp_flags': 'PSH-ACK' if np.random.random() < 0.8 else 'ACK',
            'label': 'attack',
            'attack_type': 'data_exfiltration',
            'flow_id': str(uuid.uuid4())[:8]
        })

    # DoS attacks
    for i in range(2):
        ts = now + datetime.timedelta(seconds=np.random.randint(0, 3600))
        records.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': faker.ipv4_private(),
            'dst_ip': faker.ipv4_private(),
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.choice([80, 443, 8080, 25, 53]),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], p=[0.6, 0.2, 0.2]),
            'duration_ms': np.random.exponential(scale=10),  # Very short connections
            'total_bytes': np.random.randint(40, 100),  # Small packets
            'packet_count': np.random.randint(1, 3),
            'tcp_flags': 'SYN' if np.random.random() < 0.7 else np.random.choice(['RST', 'FIN', '']),
            'label': 'attack',
            'attack_type': 'dos',
            'flow_id': str(uuid.uuid4())[:8]
        })

    # Create DataFrame
    print("Creating DataFrame and engineering features...")
    df = pd.DataFrame(records)

    # Add basic derived features
    # Byte rate
    df['byte_rate'] = df['total_bytes'] / (df['duration_ms'] + 1e-3)  # Avoid division by zero
    df['packets_per_second'] = df['packet_count'] / (df['duration_ms'] + 1e-3) * 1000
    df['bytes_per_packet'] = df['total_bytes'] / (df['packet_count'] + 1e-3)

    # Protocol and flag features
    df['is_tcp'] = (df['protocol'] == 'TCP').astype(int)
    df['is_udp'] = (df['protocol'] == 'UDP').astype(int)
    df['is_icmp'] = (df['protocol'] == 'ICMP').astype(int)
    df['is_syn'] = df['tcp_flags'].str.contains('SYN').fillna(False).astype(int)
    df['is_ack'] = df['tcp_flags'].str.contains('ACK').fillna(False).astype(int)
    df['is_rst'] = df['tcp_flags'].str.contains('RST').fillna(False).astype(int)
    df['is_fin'] = df['tcp_flags'].str.contains('FIN').fillna(False).astype(int)
    df['is_psh'] = df['tcp_flags'].str.contains('PSH').fillna(False).astype(int)

    # Port categories
    df['is_web_port'] = df['dst_port'].isin([80, 443, 8080]).astype(int)
    df['is_db_port'] = df['dst_port'].isin([3306, 5432, 1433, 1521]).astype(int)
    df['is_mail_port'] = df['dst_port'].isin([25, 143, 465, 587, 993]).astype(int)
    df['is_file_port'] = df['dst_port'].isin([20, 21, 22, 139, 445]).astype(int)
    df['is_dns_port'] = (df['dst_port'] == 53).astype(int)

    # Add time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Add mock anomaly scores (for demonstration)
    print("Adding mock anomaly scores...")
    # These would normally be calculated by models, but we'll add preset scores for this sample

    # IsolationForest scores - higher for attacks
    df['score_iso'] = np.random.random(len(df)) * 0.4  # Base scores
    df.loc[df['label'] == 'attack', 'score_iso'] += 0.4  # Higher for attacks
    df['score_iso_norm'] = df['score_iso']  # Already in 0-1 range
    df['pred_iso'] = (df['score_iso_norm'] > 0.6).astype(int)

    # OCSVM scores - higher for attacks
    df['score_ocsvm'] = np.random.random(len(df)) * 0.5
    df.loc[df['label'] == 'attack', 'score_ocsvm'] += 0.3
    df['score_ocsvm_norm'] = df['score_ocsvm']
    df['pred_ocsvm'] = (df['score_ocsvm_norm'] > 0.6).astype(int)

    # DBSCAN scores - higher for attacks
    df['score_dbscan'] = np.random.random(len(df)) * 0.5
    df.loc[df['label'] == 'attack', 'score_dbscan'] += 0.3
    df['score_dbscan_norm'] = df['score_dbscan']
    df['pred_dbscan'] = (df['score_dbscan_norm'] > 0.6).astype(int)

    # Ensemble score
    df['score_ensemble_norm'] = (df['score_iso_norm'] + df['score_ocsvm_norm'] + df['score_dbscan_norm']) / 3
    df['pred_ensemble'] = (df['score_ensemble_norm'] > 0.6).astype(int)

    # Save to CSV
    print(f"Saving dataset to {output_dir}/sample_network_flows.csv...")
    df.to_csv(f"{output_dir}/sample_network_flows.csv", index=False)

    # Create mock model metrics
    print("Creating mock model metrics...")
    model_metrics = {
        'isolation_forest': {
            'roc_auc': 0.92,
            'precision': 0.85,
            'recall': 0.80,
            'f1': 0.82,
            'accuracy': 0.88
        },
        'ocsvm': {
            'roc_auc': 0.88,
            'precision': 0.82,
            'recall': 0.75,
            'f1': 0.78,
            'accuracy': 0.86
        },
        'dbscan': {
            'roc_auc': 0.85,
            'precision': 0.78,
            'recall': 0.70,
            'f1': 0.74,
            'accuracy': 0.82
        },
        'ensemble': {
            'roc_auc': 0.94,
            'precision': 0.87,
            'recall': 0.82,
            'f1': 0.84,
            'accuracy': 0.89
        }
    }

    # Save mock model metrics
    print(f"Saving model metrics to {models_dir}/model_metrics.json...")
    with open(f"{models_dir}/model_metrics.json", 'w') as f:
        json.dump(model_metrics, f, indent=4)

    print(f"Generated sample dataset with {len(df)} flows ({len(df[df['label'] == 'normal'])} normal, {len(df[df['label'] == 'attack'])} attack)")

    return df

if __name__ == "__main__":
    df = generate_mock_data()

    # Print a sample of the data
    print("\nSample of generated network flows:")
    print(df[['timestamp', 'src_ip', 'dst_ip', 'protocol', 'duration_ms', 'total_bytes', 'label', 'attack_type', 'score_iso_norm']].head(5))
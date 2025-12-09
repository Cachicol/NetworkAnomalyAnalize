from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
import numpy as np

app = FastAPI(
    title="Network Anomaly Detection API",
    description="API para análise de flows de rede (CICFlowMeter → UNSW-NB15 → Modelo RF)",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/supervised_rf_unsw.pkl"
META_PATH = "models/supervised_rf_unsw_meta.pkl"

model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)

def clean_and_prepare_features(df_flows, meta):
    if df_flows.empty:
        return pd.DataFrame()

    # Primeiro, vamos mapear as colunas do CICFlowMeter para o formato UNSW
    cic_to_unsw_mapping = {
        # Informações de conexão
        'Src IP': 'src_ip',
        'Src Port': 'sport', 
        'Dst IP': 'dst_ip',
        'Dst Port': 'dport',
        'Protocol': 'proto',
        
        # Features principais
        'Flow Duration': 'dur',
        'Total Fwd Packet': 'spkts',
        'Total Bwd packets': 'dpkts',
        'Total Length of Fwd Packet': 'sbytes',
        'Total Length of Bwd Packet': 'dbytes',
        'Flow Bytes/s': 'rate',
        'Flow Packets/s': 'rate',
        
        # Features de tempo
        'Flow IAT Mean': 'stime',
        'Flow IAT Std': 'ltime',
        'Flow IAT Max': 'Sintpkt',
        'Flow IAT Min': 'Dintpkt',
        
        # Features de tamanho de pacote
        'Fwd Packet Length Max': 'sload',
        'Bwd Packet Length Max': 'dload',
        'Fwd Packet Length Min': 'Sjit',
        'Bwd Packet Length Min': 'Djit',
        'Fwd Packet Length Mean': 'swin',
        'Bwd Packet Length Mean': 'dwin',
        'Fwd Packet Length Std': 'stcpb',
        'Bwd Packet Length Std': 'dtcpb',
        
        # Flags TCP
        'FIN Flag Count': 'ct_flw_http_mthd',
        'SYN Flag Count': 'is_sm_ips_ports',
        'RST Flag Count': 'ct_ftp_cmd',
        'ACK Flag Count': 'ct_srv_src',
        'PSH Flag Count': 'ct_srv_dst',
        'URG Flag Count': 'ct_dst_ltm',
    }
    
    # Criar dataframe vazio com as colunas do UNSW
    df_feats = pd.DataFrame(columns=meta["train_columns"])
    
    # Preencher com valores padrão primeiro
    for col in meta["train_columns"]:
        df_feats[col] = 0
    
    # Mapear as colunas disponíveis
    for cic_col, unsw_col in cic_to_unsw_mapping.items():
        if cic_col in df_flows.columns and unsw_col in meta["train_columns"]:
            try:
                # Converter para numérico se não for coluna categórica
                if unsw_col not in ["proto", "service", "state"]:
                    df_feats[unsw_col] = pd.to_numeric(df_flows[cic_col], errors='coerce').fillna(0)
                else:
                    df_feats[unsw_col] = df_flows[cic_col].astype(str)
            except Exception as e:
                print(f"Erro ao mapear {cic_col} para {unsw_col}: {e}")
                continue
    
    # Garantir tipos numéricos para colunas não categóricas
    categorical_cols = ["proto", "service", "state"]
    for col in df_feats.columns:
        if col not in categorical_cols:
            df_feats[col] = pd.to_numeric(df_feats[col], errors='coerce').fillna(0)
    
    # Processamento de colunas categóricas
    for col in categorical_cols:
        if col in df_feats.columns:
            df_feats[col] = (
                df_feats[col]
                .astype(str)
                .str.strip()
                .replace([
                    "nan", "None", "null", "<NA>", "NaN", "",
                    "NONE", "NULL", "N/A", "n/a"
                ], "missing")
            )
            df_feats[col] = df_feats[col].apply(
                lambda x: f"cat_{x}" if x.replace(".", "").isdigit() else x
            )
    
    return df_feats

def tail_csv(file_path, n=200):
    """VERSÃO DEFINITIVA - Novo padrão de 85 colunas"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if len(lines) <= 1:
            return pd.DataFrame()
        
        header = lines[0].strip().split(',')
        
        # Processamento eficiente para o novo padrão
        data = []
        for line in lines[1:]:
            if line.strip():
                values = line.strip().split(',')
                # SEMPRE assume 85 colunas e remove a última
                if len(values) >= 84:  # Aceita 84 ou mais
                    data.append(values[:84])
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=header)
        
        # Converte colunas numéricas
        numeric_columns = [col for col in df.columns if col not in ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label']]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ordena por timestamp
        if 'Timestamp' in df.columns:
            df['Timestamp_dt'] = pd.to_datetime(
                df['Timestamp'], 
                format='%d/%m/%Y %I:%M:%S %p',
                errors='coerce'
            )
            df = df[df['Timestamp_dt'].notna()]
            df = df.sort_values('Timestamp_dt', ascending=False)
        
        print(f"✅ {len(df)} flows processados | Retornando {min(n, len(df))} mais recentes")
        
        return df.head(n)
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return pd.DataFrame()

@app.get("/")
def root():
    return {"message": "API operacional."}

@app.get("/debug_csv_raw")
def debug_csv_raw():
    """Mostra o conteúdo cru do CSV"""
    try:
        with open("data/flows.csv", "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        return {
            "total_lines": len(lines),
            "first_5_lines": lines[:5],
            "last_5_lines": lines[-5:] if len(lines) > 5 else lines,
            "file_size": os.path.getsize("data/flows.csv")
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/predict_latest")
def predict_latest():
    """
    Lê as últimas 200 linhas do flows.csv,
    converte para formato UNSW e retorna previsões.
    """
    try:
        df_flows = tail_csv("data/flows.csv", 200)
        
        if df_flows.empty:
            return {"error": "Nenhum dado encontrado no arquivo flows.csv"}

        # Usa a mesma função que funciona no Streamlit
        df_feats = clean_and_prepare_features(df_flows, meta)

        if df_feats.empty:
            return {"error": "Não foi possível preparar features para análise"}

        # Faz a predição
        preds = model.predict(df_feats)
        
        # Adicionar as predições ao dataframe convertido
        df_feats["prediction"] = preds
        df_feats["label"] = df_feats["prediction"].map({1: "Ataque", 0: "Normal"})
        
        # Adicionar informações de conexão do flow original para o retorno
        connection_mapping = {
            'Src IP': 'src_ip',
            'Src Port': 'sport', 
            'Dst IP': 'dst_ip',
            'Dst Port': 'dport',
            'Protocol': 'proto'
        }
        
        for cic_col, unsw_col in connection_mapping.items():
            if cic_col in df_flows.columns:
                df_feats[unsw_col] = df_flows[cic_col].values

        # Adicionar timestamp se disponível
        if 'Timestamp' in df_flows.columns:
            df_feats['timestamp'] = df_flows['Timestamp'].values
        elif 'ts_collected' in df_flows.columns:
            df_feats['timestamp'] = df_flows['ts_collected'].values
        else:
            df_feats['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        # Preparar dados para retorno
        attacks = df_feats[df_feats["prediction"] == 1]
        
        # Estatísticas dos ataques
        attack_stats = {}
        if not attacks.empty:
            if 'proto' in attacks.columns:
                proto_counts = attacks['proto'].value_counts().head(5).to_dict()
                attack_stats["top_protocols"] = proto_counts
            
            if 'src_ip' in attacks.columns:
                src_counts = attacks['src_ip'].value_counts().head(5).to_dict()
                attack_stats["top_source_ips"] = src_counts

        return {
            "total_flows": len(df_feats),
            "attacks": int((preds == 1).sum()),
            "normal": int((preds == 0).sum()),
            "attack_rate": float((preds.mean() * 100)),
            "attack_stats": attack_stats,
            "data": df_feats.to_dict(orient="records")
        }
        
    except Exception as e:
        return {"error": f"Erro durante a predição: {str(e)}"}

@app.get("/debug_data")
def debug_data():
    """
    Endpoint para debug dos dados
    """
    try:
        df = tail_csv("data/flows.csv", 10)
        
        if df.empty:
            return {"error": "Arquivo vazio ou não encontrado"}
            
        result = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(3).to_dict(orient="records"),
            "null_counts": df.isnull().sum().to_dict(),
            "shape": df.shape
        }
        
        # Verifica valores únicos por coluna
        for col in df.columns[:5]:  # Apenas primeiras 5 colunas para não ficar muito grande
            unique_vals = df[col].unique()[:3]
            result[f"unique_{col}"] = [str(val) for val in unique_vals]
            
        return result
        
    except Exception as e:
        return {"error": str(e)}
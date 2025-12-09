# config.py
MODEL_PATHS = {
    "rf_model": "models/supervised_rf_unsw.pkl",
    "meta": "models/supervised_rf_unsw_meta.pkl"
}

DATA_PATHS = {
    "flows": "data/flows.csv"
}

CIC_TO_UNSW_MAPPING = {
    'Src IP': 'src_ip',
    'Src Port': 'sport', 
    'Dst IP': 'dst_ip',
    'Dst Port': 'dport',
    'Protocol': 'proto',
    'Flow Duration': 'dur',
    'Total Fwd Packet': 'spkts',
    # ... resto do mapping
}

API_CONFIG = {
    "max_rows": 200,
    "default_tail": 50
}
import os
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import re

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)

# ==============================================
# CONFIGURAÇÃO
# ==============================================
st.set_page_config(page_title="Model Analysis - Saved Models", layout="wide")
st.title("📊 Análise com Modelos Salvos")

# ==============================================
# INICIALIZAÇÃO DO SESSION STATE
# ==============================================
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'display_df' not in st.session_state:
    st.session_state.display_df = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'conf_matrix_model' not in st.session_state:
    st.session_state.conf_matrix_model = None
if 'selected_model_names' not in st.session_state:
    st.session_state.selected_model_names = []

# ==============================================
# CONFIGURAÇÃO DE PATHS
# ==============================================
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def get_saved_models():
    """Lista todos os modelos salvos na pasta models/"""
    models = []
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if file.endswith("_model.pkl"):
                # Extrair nome amigável
                model_name = file.replace("_model.pkl", "").replace("_", " ").title()
                models.append({
                    'file': file,
                    'name': model_name,
                    'full_path': os.path.join(MODELS_DIR, file)
                })
    return models

def load_model_and_meta(model_file):
    """Carrega modelo e seus metadados"""
    try:
        base_name = model_file.replace("_model.pkl", "")
        model_path = os.path.join(MODELS_DIR, f"{base_name}_model.pkl")
        meta_path = os.path.join(MODELS_DIR, f"{base_name}_meta.pkl")
        
        if os.path.exists(model_path) and os.path.exists(meta_path):
            model = joblib.load(model_path)
            meta = joblib.load(meta_path)
            return model, meta
    except Exception as e:
        st.error(f"Erro ao carregar modelo {model_file}: {e}")
    return None, None

# ==============================================
# INTERFACE - SIDEBAR
# ==============================================
st.sidebar.title("⚙️ Configurações")

# Verificar modelos existentes
st.sidebar.subheader("📁 Modelos Salvos")
saved_models = get_saved_models()

if not saved_models:
    st.sidebar.error("❌ Nenhum modelo salvo encontrado!")
    st.sidebar.info("Primeiro treine e salve modelos na página de treinamento.")
else:
    st.sidebar.success(f"✅ {len(saved_models)} modelo(s) encontrado(s)")
    
    # Seleção de modelos para análise
    model_names = [m['name'] for m in saved_models]
    
    # Usar valor do session state como padrão se existir
    default_models = st.session_state.selected_model_names if st.session_state.selected_model_names else (model_names[:3] if len(model_names) >= 3 else model_names)
    
    selected_model_names = st.sidebar.multiselect(
        "Selecione modelos para analisar:",
        model_names,
        default=default_models
    )
    
    # Atualizar session state com a seleção atual
    st.session_state.selected_model_names = selected_model_names
    
    # Configurações de teste
    st.sidebar.subheader("🔧 Configurações de Teste")
    test_size = st.sidebar.slider("Tamanho do teste (%):", 10, 50, 30)
    random_state = st.sidebar.number_input("Random State:", 0, 100, 42)
    
    # Botão para analisar
    analyze_button = st.sidebar.button("🔍 Analisar Modelos Selecionados", type="primary")

# ==============================================
# CARREGAR DATASET PARA TESTE
# ==============================================
@st.cache_data
def load_test_data():
    """Carrega dados para teste"""
    DATA_PATH = r"C:\Users\pichau\.cache\kagglehub\datasets\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_testing-set.csv"
    
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        df.columns = [c.strip().replace("ï»¿", "") for c in df.columns]
        
        # Features básicas que nossos modelos usam
        BASIC_FEATURES = [
            'src_ip', 'sport', 'dst_ip', 'dport', 'proto',
            'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes'
        ]
        
        # Verificar quais features existem
        available_features = [f for f in BASIC_FEATURES if f in df.columns]
        
        # Manter apenas features disponíveis + label
        features_to_keep = available_features + ['label']
        df = df[features_to_keep]
        
        # Remover nulos
        df = df.dropna()
        
        return df, available_features
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

# ==============================================
# PAINEL PRINCIPAL
# ==============================================
st.write("## 📊 Análise de Modelos Salvos")

if not saved_models:
    st.error("""
    ❌ **Nenhum modelo salvo encontrado!**
    
    Para usar esta página:
    1. Vá para a página de treinamento
    2. Treine alguns modelos
    3. Marque "Salvar automaticamente após treino"
    4. Volte aqui para analisar
    """)
    st.stop()

# Mostrar modelos disponíveis
st.write(f"### 📁 Modelos Disponíveis ({len(saved_models)})")

col1, col2, col3 = st.columns(3)
model_stats = {'total': len(saved_models), 'loaded': 0, 'failed': 0}

for idx, model_info in enumerate(saved_models):
    model, meta = load_model_and_meta(model_info['file'])
    
    if model and meta:
        model_stats['loaded'] += 1
        
        # Mostrar informações básicas
        if idx % 3 == 0:
            with col1:
                with st.expander(f"🤖 {model_info['name']}", expanded=False):
                    st.write(f"**Treinado em:** {meta.get('training_date', 'N/A')}")
                    st.write(f"**Features:** {len(meta.get('features', []))}")
                    
                    if 'metrics' in meta:
                        st.write("**Métricas originais:**")
                        st.write(f"Acurácia: {meta['metrics'].get('accuracy', 0):.3f}")
                        st.write(f"F1-Score: {meta['metrics'].get('f1_score', 0):.3f}")
        elif idx % 3 == 1:
            with col2:
                with st.expander(f"🤖 {model_info['name']}", expanded=False):
                    st.write(f"**Treinado em:** {meta.get('training_date', 'N/A')}")
                    st.write(f"**Features:** {len(meta.get('features', []))}")
                    
                    if 'metrics' in meta:
                        st.write("**Métricas originais:**")
                        st.write(f"Acurácia: {meta['metrics'].get('accuracy', 0):.3f}")
                        st.write(f"F1-Score: {meta['metrics'].get('f1_score', 0):.3f}")
        else:
            with col3:
                with st.expander(f"🤖 {model_info['name']}", expanded=False):
                    st.write(f"**Treinado em:** {meta.get('training_date', 'N/A')}")
                    st.write(f"**Features:** {len(meta.get('features', []))}")
                    
                    if 'metrics' in meta:
                        st.write("**Métricas originais:**")
                        st.write(f"Acurácia: {meta['metrics'].get('accuracy', 0):.3f}")
                        st.write(f"F1-Score: {meta['metrics'].get('f1_score', 0):.3f}")
    else:
        model_stats['failed'] += 1

st.success(f"✅ {model_stats['loaded']}/{model_stats['total']} modelos carregados com sucesso")

# ==============================================
# FUNÇÃO PARA EXECUTAR ANÁLISE
# ==============================================
def run_analysis(selected_model_names, test_size, random_state):
    """Executa análise dos modelos selecionados"""
    # Carregar dados de teste
    df, available_features = load_test_data()
    
    if df is None:
        st.error("Não foi possível carregar dados para teste.")
        return None, None, None
    
    # Preparar dados
    X = df[available_features]
    y = df['label']
    
    # Split para teste
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state, stratify=y
    )
    
    # Armazenar resultados
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model_name in enumerate(selected_model_names):
        status_text.text(f"Testando {model_name}...")
        
        # Encontrar o modelo correspondente
        model_info = next((m for m in saved_models if m['name'] == model_name), None)
        if not model_info:
            st.warning(f"Modelo {model_name} não encontrado na lista.")
            continue
            
        model, meta = load_model_and_meta(model_info['file'])
        if not model:
            st.error(f"Não foi possível carregar o modelo {model_name}.")
            continue
            
        try:
            # Fazer previsões
            y_pred = model.predict(X_test)
            
            # Métricas básicas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            
            # AUC-ROC (se disponível)
            roc_auc = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    roc_auc = None
            
            # Salvar resultados
            results.append({
                'Modelo': model_name,
                'Acurácia': accuracy,
                'Precisão': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC-ROC': roc_auc,
                'Modelo Obj': model,
                'Meta': meta
            })
            
            st.success(f"✅ {model_name} testado!")
            
        except Exception as e:
            st.error(f"❌ Erro ao testar {model_name}: {e}")
        
        progress_bar.progress((idx + 1) / len(selected_model_names))
    
    status_text.text("Análise concluída!")
    
    # Criar DataFrame para exibição
    if results:
        results_df = pd.DataFrame(results)
        display_df = results_df.drop(columns=['Modelo Obj', 'Meta'])
        display_df = display_df.sort_values('F1-Score', ascending=False)
        
        # Salvar no session state
        st.session_state.results = results
        st.session_state.display_df = display_df
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.analysis_complete = True
        
        # Definir modelo padrão para matriz de confusão
        if display_df is not None and not display_df.empty:
            st.session_state.conf_matrix_model = display_df['Modelo'].iloc[0]
    
    return results, display_df, (X_test, y_test)

# ==============================================
# EXECUTAR ANÁLISE QUANDO BOTÃO CLICADO
# ==============================================
if 'analyze_button' in locals() and analyze_button and st.session_state.selected_model_names:
    st.session_state.analysis_complete = False
    results, display_df, test_data = run_analysis(st.session_state.selected_model_names, test_size, random_state)

# ==============================================
# MOSTRAR RESULTADOS DA ANÁLISE
# ==============================================
if st.session_state.analysis_complete and st.session_state.results:
    st.write("## 🔍 Análise Comparativa")
    
    # Obter dados do session state
    results = st.session_state.results
    display_df = st.session_state.display_df
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # RESULTADOS - TABELA COMPARATIVA
    st.write("### 📊 Resultados no Conjunto de Teste")
    st.dataframe(
        display_df.style.format({
            'Acurácia': '{:.3f}',
            'Precisão': '{:.3f}', 
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}',
            'AUC-ROC': '{:.3f}'
        }).highlight_max(subset=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC'], 
                       color='lightgreen')
        .highlight_min(subset=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC'], 
                     color='lightcoral'),
        use_container_width=True
    )
    
    # ==============================================
    # GRÁFICOS DE COMPARAÇÃO
    # ==============================================
    st.write("### 📈 Visualização dos Resultados")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Métricas", "🎯 Curvas ROC", "📉 Matriz Confusão", "📋 Detalhes"])
    
    with tab1:
        # Gráfico de barras das métricas
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics_to_plot):
            axes[idx].bar(display_df['Modelo'], display_df[metric], color='skyblue', edgecolor='black')
            axes[idx].set_title(f'{metric} por Modelo')
            axes[idx].set_ylabel(metric)
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Adicionar valores no topo das barras
            for j, v in enumerate(display_df[metric]):
                axes[idx].text(j, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        # Curvas ROC
        models_with_proba = [r for r in results if r['AUC-ROC'] is not None]
        
        if models_with_proba:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for result in models_with_proba:
                model = result['Modelo Obj']
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, lw=2, 
                           label=f"{result['Modelo']} (AUC = {roc_auc:.3f})")
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Aleatório (AUC = 0.5)')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Taxa de Falsos Positivos')
            ax.set_ylabel('Taxa de Verdadeiros Positivos')
            ax.set_title('Curvas ROC - Comparação de Modelos')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        else:
            st.info("⚠️ Nenhum modelo com probabilidades disponível para curva ROC.")
    
    with tab3:
        # Matrizes de Confusão
        st.write("Selecione um modelo para ver sua matriz de confusão:")
        
        # Usar session state para armazenar a seleção
        if 'conf_matrix_model' not in st.session_state:
            st.session_state.conf_matrix_model = display_df['Modelo'].iloc[0] if not display_df.empty else None
        
        # Selectbox que atualiza o session state
        if not display_df.empty:
            model_options = display_df['Modelo'].tolist()
            current_index = model_options.index(st.session_state.conf_matrix_model) if st.session_state.conf_matrix_model in model_options else 0
            
            selected_for_conf_matrix = st.selectbox(
                "Modelo:",
                model_options,
                key="conf_matrix_model_select",
                index=current_index
            )
            
            # Atualizar session state quando mudar
            if selected_for_conf_matrix != st.session_state.conf_matrix_model:
                st.session_state.conf_matrix_model = selected_for_conf_matrix
            
            # Encontrar o modelo selecionado
            selected_result = next((r for r in results if r['Modelo'] == st.session_state.conf_matrix_model), None)
            
            if selected_result:
                model = selected_result['Modelo Obj']
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Normal', 'Ataque'],
                           yticklabels=['Normal', 'Ataque'],
                           ax=ax)
                ax.set_title(f'Matriz de Confusão - {st.session_state.conf_matrix_model}')
                ax.set_xlabel('Predito')
                ax.set_ylabel('Real')
                
                st.pyplot(fig)
                
                # Relatório de classificação
                st.write("**Relatório de Classificação:**")
                report = classification_report(y_test, y_pred, target_names=['Normal', 'Ataque'])
                st.code(report)
    
    with tab4:
        # Detalhes dos modelos
        st.write("### 📋 Detalhes dos Modelos")
        
        for result in results:
            with st.expander(f"🔍 {result['Modelo']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Métricas Atuais:**")
                    st.write(f"Acurácia: {result['Acurácia']:.3f}")
                    st.write(f"Precisão: {result['Precisão']:.3f}")
                    st.write(f"Recall: {result['Recall']:.3f}")
                    st.write(f"F1-Score: {result['F1-Score']:.3f}")
                    if result['AUC-ROC']:
                        st.write(f"AUC-ROC: {result['AUC-ROC']:.3f}")
                
                with col2:
                    meta = result['Meta']
                    st.write("**Informações do Treino:**")
                    st.write(f"Data: {meta.get('training_date', 'N/A')}")
                    st.write(f"Features: {len(meta.get('features', []))}")
                    
                    if 'metrics' in meta:
                        st.write("**Métricas Originais:**")
                        st.write(f"Acurácia: {meta['metrics'].get('accuracy', 0):.3f}")
                        st.write(f"F1: {meta['metrics'].get('f1_score', 0):.3f}")

else:
    # Mostrar mensagem apropriada baseada na seleção
    if st.session_state.selected_model_names:
        st.info("👆 **Clique em 'Analisar Modelos Selecionados' na sidebar para começar**")
    else:
        st.info("👈 **Selecione modelos na sidebar para análise**")

# ==============================================
# RODAPÉ
# ==============================================
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **📊 Análise de Modelos Salvos**  
    
    • Testa modelos já treinados  
    • Compara performance  
    • Gera métricas completas  
    • Visualiza resultados
    """
)

st.sidebar.write(f"**🕒 {datetime.now().strftime('%H:%M:%S')}**")

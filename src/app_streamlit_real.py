import os
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import re  # Adicionar para limpar nomes de arquivos

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)

# Importar vários modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# ==============================================
# CONFIGURAÇÃO
# ==============================================
st.set_page_config(page_title="Model Comparison - Mapped Features", layout="wide")
st.title("🔬 Comparação de Modelos - Features Mapeadas")

# ==============================================
# CONFIGURAÇÃO DE PATHS
# ==============================================
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)  # ⭐ CRIAR PASTA LOGO NO INÍCIO ⭐

def clean_filename(name):
    """Limpa nome para ser usado como nome de arquivo"""
    # Substituir caracteres inválidos
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remover espaços extras
    name = name.strip()
    # Substituir espaços por underscores
    name = name.replace(' ', '_')
    # Converter para minúsculas
    name = name.lower()
    return name

def save_model_pipeline(model, model_name, meta_info):
    """Salva modelo e metadados de forma segura"""
    try:
        # Limpar nome do modelo
        safe_name = clean_filename(model_name)
        
        # Caminhos dos arquivos
        model_path = os.path.join(MODELS_DIR, f"{safe_name}_model.pkl")
        meta_path = os.path.join(MODELS_DIR, f"{safe_name}_meta.pkl")
        
        # Salvar modelo
        joblib.dump(model, model_path)
        
        # Salvar metadados
        joblib.dump(meta_info, meta_path)
        
        return model_path, meta_path
    except Exception as e:
        st.error(f"Erro ao salvar modelo {model_name}: {e}")
        return None, None

def load_model_pipeline(model_name):
    """Carrega modelo e metadados"""
    try:
        safe_name = clean_filename(model_name)
        model_path = os.path.join(MODELS_DIR, f"{safe_name}_model.pkl")
        meta_path = os.path.join(MODELS_DIR, f"{safe_name}_meta.pkl")
        
        if os.path.exists(model_path) and os.path.exists(meta_path):
            model = joblib.load(model_path)
            meta = joblib.load(meta_path)
            return model, meta
        return None, None
    except:
        return None, None

# ==============================================
# FEATURES MAPEADAS (CONSTANTE)
# ==============================================
BASIC_FEATURES = [
    'src_ip', 'sport', 'dst_ip', 'dport', 'proto',  # Conexão
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes'     # Estatísticas
]

# Mapeamento para exibição
FEATURE_DISPLAY_NAMES = {
    'src_ip': 'IP Origem',
    'sport': 'Porta Origem',
    'dst_ip': 'IP Destino',
    'dport': 'Porta Destino',
    'proto': 'Protocolo',
    'dur': 'Duração',
    'spkts': 'Pacotes Enviados',
    'dpkts': 'Pacotes Recebidos',
    'sbytes': 'Bytes Enviados',
    'dbytes': 'Bytes Recebidos'
}

# ==============================================
# CARREGAR DATASET
# ==============================================
@st.cache_data
def load_data():
    """Carrega e prepara o dataset UNSW-NB15"""
    DATA_PATH = r"C:\Users\pichau\.cache\kagglehub\datasets\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_testing-set.csv"
    
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        df.columns = [c.strip().replace("ï»¿", "") for c in df.columns]
        
        # Verificar quais features mapeadas existem
        available_features = [f for f in BASIC_FEATURES if f in df.columns]
        
        if not available_features:
            st.error("Nenhuma feature mapeada encontrada no dataset!")
            return None, None
        
        # Manter apenas features mapeadas + label
        features_to_keep = available_features + ['label']
        df = df[features_to_keep]
        
        # Remover linhas com valores nulos
        df = df.dropna()
        
        return df, available_features
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

# ==============================================
# INTERFACE - SIDEBAR
# ==============================================
st.sidebar.title("⚙️ Configurações")

# Verificar modelos existentes
st.sidebar.subheader("📁 Modelos Existentes")
existing_models = []
if os.path.exists(MODELS_DIR):
    for file in os.listdir(MODELS_DIR):
        if file.endswith("_model.pkl"):
            # Extrair nome do modelo do arquivo
            model_name = file.replace("_model.pkl", "").replace("_", " ").title()
            existing_models.append(model_name)

if existing_models:
    st.sidebar.success(f"{len(existing_models)} modelo(s) já treinado(s)")
    for model in existing_models[:5]:  # Mostrar apenas 5
        st.sidebar.write(f"• {model}")
else:
    st.sidebar.info("Nenhum modelo treinado ainda")

# Seleção de modelos
st.sidebar.subheader("Seleção de Modelos")
model_options = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB()
}

selected_models = st.sidebar.multiselect(
    "Escolha os modelos para comparar:",
    list(model_options.keys()),
    default=["Random Forest", "Logistic Regression", "Decision Tree"]
)

# Configurações de treino
st.sidebar.subheader("Configurações de Treino")
test_size = st.sidebar.slider("Tamanho do teste (%):", 10, 50, 30)
random_state = st.sidebar.number_input("Random State:", 0, 100, 42)
use_cross_val = st.sidebar.checkbox("Usar Cross-Validation", value=False)
cv_folds = st.sidebar.slider("Número de folds (CV):", 3, 10, 5) if use_cross_val else 5

# Configurações de salvamento
st.sidebar.subheader("💾 Configurações de Salvamento")
auto_save = st.sidebar.checkbox("Salvar automaticamente após treino", value=True)
overwrite_existing = st.sidebar.checkbox("Sobrescrever modelos existentes", value=True)

# Botão para treinar
train_button = st.sidebar.button("🚀 Treinar Modelos Selecionados", type="primary")

# ==============================================
# CARREGAR DADOS
# ==============================================
st.write("## 📊 Carregamento de Dados")

df, available_features = load_data()

if df is None:
    st.error("Não foi possível carregar os dados. Verifique o caminho do arquivo.")
    st.stop()

# Informações do dataset
st.success(f"✅ Dataset carregado com sucesso!")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total de Amostras", len(df))
with col2:
    normal = (df['label'] == 0).sum()
    st.metric("Normal", normal)
with col3:
    attack = (df['label'] == 1).sum()
    st.metric("Ataque", attack)

# Mostrar features disponíveis
st.write(f"**Features mapeadas disponíveis:** {len(available_features)}/{len(BASIC_FEATURES)}")
for i, feat in enumerate(available_features):
    st.write(f"{i+1}. {FEATURE_DISPLAY_NAMES.get(feat, feat)} (`{feat}`)")

# Mostrar primeiras linhas
with st.expander("👀 Visualizar dados (primeiras 10 linhas)"):
    st.dataframe(df.head(10))

# ==============================================
# TREINAR MODELOS
# ==============================================
if train_button and selected_models:
    st.write("## 🤖 Treinamento de Modelos")
    
    # Separar features e target
    X = df[available_features]
    y = df['label']
    
    # Identificar tipos de colunas
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    # Criar pré-processador
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state, stratify=y
    )
    
    # Treinar cada modelo selecionado
    results = []
    trained_models = {}
    saved_models_info = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model_name in enumerate(selected_models):
        status_text.text(f"Treinando {model_name}...")
        
        # Criar pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model_options[model_name])
        ])
        
        try:
            # Treinar modelo
            pipeline.fit(X_train, y_train)
            
            # Previsões
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            
            # Cross-validation se solicitado
            cv_score = None
            if use_cross_val:
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='accuracy')
                cv_score = cv_scores.mean()
            
            # AUC-ROC se disponível
            roc_auc = None
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Salvar resultados
            results.append({
                'Modelo': model_name,
                'Acurácia': accuracy,
                'Precisão': precision,
                'Recall': recall,
                'F1-Score': f1,
                'CV Score': cv_score,
                'AUC-ROC': roc_auc
            })
            
            # Salvar modelo treinado
            trained_models[model_name] = pipeline
            
            # ⭐⭐ SALVAR AUTOMATICAMENTE SE CONFIGURADO ⭐⭐
            if auto_save:
                # Verificar se já existe
                existing_model, existing_meta = load_model_pipeline(model_name)
                
                if existing_model and not overwrite_existing:
                    st.warning(f"Modelo '{model_name}' já existe. Pulando salvamento.")
                else:
                    # Criar metadados
                    meta_info = {
                        'model_name': model_name,
                        'features': available_features,
                        'num_cols': num_cols,
                        'cat_cols': cat_cols,
                        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'test_size': test_size,
                        'random_state': random_state,
                        'metrics': {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'cv_score': cv_score,
                            'roc_auc': roc_auc
                        },
                        'dataset_info': {
                            'total_samples': len(df),
                            'train_samples': len(X_train),
                            'test_samples': len(X_test),
                            'normal_samples': normal,
                            'attack_samples': attack
                        }
                    }
                    
                    # Salvar modelo
                    model_path, meta_path = save_model_pipeline(pipeline, model_name, meta_info)
                    
                    if model_path and meta_path:
                        saved_models_info.append({
                            'name': model_name,
                            'model_path': model_path,
                            'meta_path': meta_path
                        })
                        st.success(f"✅ {model_name} treinado e salvo!")
                    else:
                        st.success(f"✅ {model_name} treinado! (não salvo)")
            else:
                st.success(f"✅ {model_name} treinado! (não salvo)")
            
        except Exception as e:
            st.error(f"❌ Erro ao treinar {model_name}: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        progress_bar.progress((idx + 1) / len(selected_models))
    
    status_text.text("Treinamento concluído!")
    
    # Mostrar modelos salvos
    if saved_models_info:
        st.write("### 💾 Modelos Salvos")
        for info in saved_models_info:
            st.write(f"• **{info['name']}**:")
            st.write(f"  Modelo: `{info['model_path']}`")
            st.write(f"  Metadados: `{info['meta_path']}`")
    
    # ==============================================
    # RESULTADOS - TABELA COMPARATIVA
    # ==============================================
    st.write("## 📈 Resultados - Comparação")
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Ordenar por F1-Score
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        # Exibir tabela
        st.dataframe(
            results_df.style.format({
                'Acurácia': '{:.3f}',
                'Precisão': '{:.3f}', 
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}',
                'CV Score': '{:.3f}' if use_cross_val else None,
                'AUC-ROC': '{:.3f}'
            }).highlight_max(subset=['Acurácia', 'Precisão', 'Recall', 'F1-Score'], color='lightgreen')
            .highlight_min(subset=['Acurácia', 'Precisão', 'Recall', 'F1-Score'], color='lightcoral'),
            use_container_width=True
        )
        
        # ==============================================
        # GRÁFICOS DE COMPARAÇÃO
        # ==============================================
        st.write("### 📊 Visualização dos Resultados")
        
        # Criar tabs para diferentes visualizações
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Métricas Principais", "📊 Barras", "🎯 AUC-ROC", "📉 Matriz de Confusão"])
        
        with tab1:
            # Gráfico de radar/spider (simplificado)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Preparar dados para o gráfico
            metrics_to_plot = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
            x = np.arange(len(metrics_to_plot))
            width = 0.8 / len(results_df)
            
            for i, (_, row) in enumerate(results_df.iterrows()):
                values = [row[metric] for metric in metrics_to_plot]
                offset = width * i - (width * len(results_df) / 2) + width/2
                ax.bar(x + offset, values, width, label=row['Modelo'])
            
            ax.set_xlabel('Métricas')
            ax.set_ylabel('Score')
            ax.set_title('Comparação de Métricas por Modelo')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_to_plot)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_ylim([0, 1])
            
            st.pyplot(fig)
        
        with tab2:
            # Gráfico de barras agrupadas
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, metric in enumerate(['Acurácia', 'Precisão', 'Recall', 'F1-Score']):
                axes[idx].bar(results_df['Modelo'], results_df[metric], color='skyblue', edgecolor='black')
                axes[idx].set_title(f'{metric} por Modelo')
                axes[idx].set_ylabel(metric)
                axes[idx].set_ylim([0, 1])
                axes[idx].tick_params(axis='x', rotation=45)
                
                # Adicionar valores no topo das barras
                for j, v in enumerate(results_df[metric]):
                    axes[idx].text(j, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            # Curvas ROC
            if any(results_df['AUC-ROC'].notna()):
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for model_name in results_df['Modelo']:
                    if results_df.loc[results_df['Modelo'] == model_name, 'AUC-ROC'].values[0] is not None:
                        pipeline = trained_models[model_name]
                        if hasattr(pipeline, 'predict_proba'):
                            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                            roc_auc = auc(fpr, tpr)
                            
                            ax.plot(fpr, tpr, lw=2, 
                                   label=f'{model_name} (AUC = {roc_auc:.3f})')
                
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
                st.info("AUC-ROC não disponível para os modelos selecionados.")
        
        with tab4:
            # Matrizes de Confusão
            st.write("Selecione um modelo para ver sua matriz de confusão:")
            
            selected_model_conf = st.selectbox(
                "Modelo:",
                results_df['Modelo'].tolist(),
                key="conf_matrix_select"
            )
            
            if selected_model_conf in trained_models:
                pipeline = trained_models[selected_model_conf]
                y_pred = pipeline.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Normal', 'Ataque'],
                           yticklabels=['Normal', 'Ataque'],
                           ax=ax)
                ax.set_title(f'Matriz de Confusão - {selected_model_conf}')
                ax.set_xlabel('Predito')
                ax.set_ylabel('Real')
                
                st.pyplot(fig)
                
                # Relatório de classificação
                st.write("**Relatório de Classificação:**")
                report = classification_report(y_test, y_pred, target_names=['Normal', 'Ataque'])
                st.code(report)
        
        # ==============================================
        # ANÁLISE DAS FEATURES
        # ==============================================
        st.write("## 🔍 Análise das Features")
        
        # Mostrar importância das features para modelos baseados em árvores
        tree_based_models = ['Random Forest', 'Gradient Boosting', 'Decision Tree']
        available_tree_models = [m for m in tree_based_models if m in trained_models]
        
        if available_tree_models:
            st.write("### Importância das Features (Modelos baseados em árvores)")
            
            for model_name in available_tree_models:
                pipeline = trained_models[model_name]
                
                # Tentar obter importância das features
                try:
                    classifier = pipeline.named_steps['classifier']
                    
                    if hasattr(classifier, 'feature_importances_'):
                        # Obter nomes das features após pré-processamento
                        preprocessor = pipeline.named_steps['preprocessor']
                        preprocessor.fit(X_train)
                        
                        # Obter nomes das features
                        feature_names = []
                        
                        # Features numéricas
                        feature_names.extend(num_cols)
                        
                        # Features categóricas (one-hot encoded)
                        if 'cat' in preprocessor.named_transformers_:
                            ohe = preprocessor.named_transformers_['cat']
                            if hasattr(ohe, 'categories_'):
                                for i, col in enumerate(cat_cols):
                                    if i < len(ohe.categories_):
                                        for cat in ohe.categories_[i]:
                                            feature_names.append(f"{col}_{cat}")
                        
                        importances = classifier.feature_importances_
                        
                        if len(feature_names) == len(importances):
                            # Criar DataFrame
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False).head(15)
                            
                            # Plotar
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
                            ax.set_xlabel('Importância')
                            ax.set_title(f'Top 15 Features - {model_name}')
                            ax.invert_yaxis()  # Maior importância no topo
                            
                            # Adicionar valores
                            for bar in bars:
                                width = bar.get_width()
                                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                                       f'{width:.4f}', ha='left', va='center')
                            
                            st.pyplot(fig)
                            
                            # Mostrar tabela
                            with st.expander(f"Ver todas as importâncias - {model_name}"):
                                st.dataframe(importance_df)
                except Exception as e:
                    st.warning(f"Não foi possível obter importância das features para {model_name}: {e}")
        
        # Estatísticas básicas das features
        with st.expander("📊 Estatísticas das Features"):
            st.write("**Estatísticas descritivas das features numéricas:**")
            numeric_stats = X[num_cols].describe().T if num_cols else pd.DataFrame()
            st.dataframe(numeric_stats)
            
            if cat_cols:
                st.write("**Distribuição das features categóricas:**")
                for col in cat_cols:
                    st.write(f"**{col}:**")
                    value_counts = X[col].value_counts().head(10)
                    for val, count in value_counts.items():
                        st.write(f"  - {val}: {count} ({count/len(X)*100:.1f}%)")
    
    else:
        st.warning("Nenhum modelo foi treinado com sucesso.")

else:
    if not selected_models:
        st.info("👈 **Selecione pelo menos um modelo na sidebar e clique em 'Treinar Modelos'**")
    else:
        st.info("👆 **Clique no botão 'Treinar Modelos Selecionados' para iniciar**")

# ==============================================
# RODAPÉ
# ==============================================
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **🔬 Comparador de Modelos**  
    
    • Treina múltiplos modelos ML  
    • Usa apenas features mapeadas  
    • Compara métricas de performance  
    • Salva automaticamente em `models/`
    """
)

st.sidebar.write(f"**Última execução:**")
st.sidebar.write(datetime.now().strftime("%H:%M:%S"))
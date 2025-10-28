import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np

st.set_page_config(page_title="EDA", layout="wide")

st.title("Exploration des Données (EDA)")
st.markdown("### Analyse complète des données avant preprocessing")

# Vérifier que les données sont chargées
if st.session_state['df_raw'] is None:
    st.warning("Aucune donnée chargée. Allez d'abord à la page **Chargement**")
    st.stop()

df = st.session_state['df_raw']

# Onglets pour différentes analyses
tab1, tab2, tab3= st.tabs([
    "Vue d'ensemble",
    "Analyse univariée",
    "Analyse bivariée",
])

# ============================================================
# TAB 1: VUE D'ENSEMBLE
# ============================================================
with tab1:
    st.subheader("Vue d'ensemble des données")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Lignes", f"{df.shape[0]:,}")
    with col2:
        st.metric("Colonnes", df.shape[1])
    with col3:
        null_count = df.isnull().sum().sum()
        null_pct = (null_count / (df.shape[0] * df.shape[1])) * 100
        st.metric("Valeurs manquantes", f"{null_count:,}", f"{null_pct:.1f}%")
    with col4:
        size_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Taille", f"{size_mb:.2f} MB")
    
    st.markdown("---")
    
    # Types de colonnes
    st.markdown("### 📋 Répartition des types de données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        type_counts = df.dtypes.value_counts()
        
        fig = px.pie(
            values=type_counts.values,
            names=[str(t) for t in type_counts.index],
            title='Distribution des types de colonnes'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Détails des types:**")
        for dtype, count in type_counts.items():
            st.write(f"• {dtype}: {count} colonnes")
        
        st.write(f"\n**Colonnes numériques:** {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
        st.write(f"**Colonnes textuelles:** {len(df.select_dtypes(include=['object']).columns)}")
    
    st.markdown("---")
    
    # Statistiques descriptives
    st.markdown("### Statistiques descriptives")
    
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if len(numeric_df.columns) > 0:
        st.dataframe(numeric_df.describe().T, use_container_width=True)
    else:
        st.warning("Aucune colonne numérique trouvée")
    
    # Valeurs manquantes détaillées
    st.markdown("### Analyse des valeurs manquantes")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Colonne': missing.index,
        'Valeurs manquantes': missing.values,
        'Pourcentage': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Valeurs manquantes'] > 0].sort_values('Valeurs manquantes', ascending=False)
    
    if len(missing_df) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                missing_df,
                x='Colonne',
                y='Pourcentage',
                title='Pourcentage de valeurs manquantes par colonne',
                color='Pourcentage',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(missing_df, use_container_width=True, height=400)
    else:
        st.success("Aucune valeur manquante dans le dataset")

# ============================================================
# TAB 2: ANALYSE UNIVARIÉE
# ============================================================
with tab2:
    st.subheader("Analyse univariée")
    
    st.info("Explore la distribution de chaque variable individuellement")
    
    # Sélection de colonne
    col = st.selectbox("Choisir une variable à analyser", df.columns)
    
    if df[col].dtype in ['int64', 'float64']:
        # Variable numérique
        st.markdown(f"### Analyse de **{col}** (numérique)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogramme
            fig = px.histogram(
                df, 
                x=col, 
                title=f"Distribution de {col}",
                nbins=30,
                marginal='box'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boxplot
            fig = px.box(
                df, 
                y=col, 
                title=f"Boxplot de {col}",
                points='outliers'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques
        st.markdown("### Statistiques")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Moyenne", f"{df[col].mean():.2f}")
        with col2:
            st.metric("Médiane", f"{df[col].median():.2f}")
        with col3:
            st.metric("Écart-type", f"{df[col].std():.2f}")
        with col4:
            st.metric("Min", f"{df[col].min():.2f}")
        with col5:
            st.metric("Max", f"{df[col].max():.2f}")
        
        # Détection des outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            st.warning(f"{len(outliers)} outliers détectés ({len(outliers)/len(df)*100:.1f}%)")
            
            with st.expander("Voir les outliers"):
                st.dataframe(outliers[[col]], use_container_width=True)
        else:
            st.success("Aucun outlier détecté")
    
    else:
        # Variable catégorielle
        st.markdown(f"### Analyse de **{col}** (catégorielle)")
        
        value_counts = df[col].value_counts()
        value_pct = (value_counts / len(df) * 100).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution de {col}",
                labels={'x': col, 'y': 'Nombre'},
                color=value_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Répartition de {col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau des fréquences
        st.markdown("### Fréquences")
        
        freq_df = pd.DataFrame({
            'Valeur': value_counts.index,
            'Nombre': value_counts.values,
            'Pourcentage': value_pct.values
        })
        
        st.dataframe(freq_df, use_container_width=True)

# ============================================================
# TAB 3: ANALYSE BIVARIÉE
# ============================================================
with tab3:
    st.subheader("Analyse bivariée")
    
    st.info("Compare les variables entre elles")
    
    if 'ChurnLabel' in df.columns:
        # Sélection de variable
        col = st.selectbox(
            "Variable à analyser par rapport au Churn",
            [c for c in df.columns if c != 'ChurnLabel']
        )
        
        if df[col].dtype in ['int64', 'float64']:
            # Variable numérique vs Churn
            st.markdown(f"### {col} par Churn")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot par churn
                fig = px.box(
                    df, 
                    x='ChurnLabel', 
                    y=col,
                    title=f"{col} par Churn",
                    color='ChurnLabel',
                    labels={'ChurnLabel': 'Churn (0=Non, 1=Oui)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Violin plot
                fig = px.violin(
                    df,
                    x='ChurnLabel',
                    y=col,
                    title=f"Distribution de {col} par Churn",
                    box=True,
                    color='ChurnLabel'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Test statistique
            st.markdown("### Test de Student")
            
            churn_0 = df[df['ChurnLabel'] == 0][col].dropna()
            churn_1 = df[df['ChurnLabel'] == 1][col].dropna()
            
            t_stat, p_value = stats.ttest_ind(churn_0, churn_1)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("t-statistique", f"{t_stat:.4f}")
            with col2:
                st.metric("p-value", f"{p_value:.4f}")
            with col3:
                if p_value < 0.05:
                    st.success("Différence significative")
                else:
                    st.info("Pas de différence significative")
            
            # Statistiques par groupe
            st.markdown("### Statistiques par groupe")
            
            stats_df = df.groupby('ChurnLabel')[col].agg(['mean', 'median', 'std']).round(2)
            stats_df.index = ['Non-Churn', 'Churn']
            
            st.dataframe(stats_df, use_container_width=True)
        
        else:
            # Variable catégorielle vs Churn
            st.markdown(f"### {col} par Churn")
            
            # Table croisée
            crosstab = pd.crosstab(df[col], df['ChurnLabel'], normalize='index') * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Grouped bar chart
                ct_counts = pd.crosstab(df[col], df['ChurnLabel'])
                
                fig = px.bar(
                    ct_counts,
                    title=f"Distribution de {col} par Churn",
                    barmode='group',
                    labels={'value': 'Nombre', 'variable': 'Churn'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Heatmap des pourcentages
                fig = px.imshow(
                    crosstab,
                    title=f"Taux de churn par {col} (%)",
                    text_auto='.1f',
                    aspect='auto',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        # st.warning("La variable ChurnLabel n'est pas présente dans les données")
        
        st.markdown("### Analyse bivariée générale")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1_select = st.selectbox("Variable X", numeric_cols, index=0)
            col2_select = st.selectbox("Variable Y", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            fig = px.scatter(
                df,
                x=col1_select,
                y=col2_select,
                title=f"{col1_select} vs {col2_select}",
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.success("Analyse exploratoire terminée ! Vous pouvez maintenant passer au **Prétraitement**")

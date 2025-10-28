import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Dashboard Analytique")
st.markdown("### Analyse détaillée des prédictions")

# Vérifier que les prédictions existent
if st.session_state['df_with_predictions'] is None:
    st.warning("Aucune prédiction. Allez à la page de **Prédiction**")
    st.stop()

df = st.session_state['df_with_predictions']

# KPIs Globaux
st.markdown("### KPIs Globaux")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Clients", f"{len(df):,}")

with col2:
    churn_rate = (df['Prediction'] == 1).mean() * 100
    st.metric("Taux de Churn", f"{churn_rate:.1f}%")

with col3:
    avg_prob = df['Probability_Churn'].mean()
    st.metric("Probabilité Moyenne", f"{avg_prob:.2%}")

with col4:
    high_risk = (df['Risk_Level'] == 'Élevé').sum()
    st.metric("Clients à Risque Élevé", f"{high_risk:,}")

with col5:
    if 'ChurnLabel' in df.columns:
        accuracy = (df['Prediction'] == df['ChurnLabel']).mean() * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")

st.markdown("---")

# Distribution des probabilités
st.markdown("### Distribution des Probabilités de Churn")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df, 
        x='Probability_Churn',
        nbins=50,
        title='Distribution des probabilités',
        color_discrete_sequence=['#FF6B6B']
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        df,
        y='Probability_Churn',
        title='Boxplot des probabilités',
        color_discrete_sequence=['#4ECDC4']
    )
    st.plotly_chart(fig, use_container_width=True)

# Analyse par segment
if 'Segment' in df.columns:
    st.markdown("---")
    st.markdown("### Analyse par Segment")
    
    segment_analysis = df.groupby('Segment').agg({
        'Prediction': ['count', 'sum'],
        'Probability_Churn': 'mean'
    }).round(2)
    
    segment_analysis.columns = ['Total', 'Churn_Prédit', 'Prob_Moyenne']
    segment_analysis['Taux_Churn_%'] = (
        segment_analysis['Churn_Prédit'] / segment_analysis['Total'] * 100
    ).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(segment_analysis, use_container_width=True)
    
    with col2:
        fig = px.bar(
            segment_analysis.reset_index(),
            x='Segment',
            y='Taux_Churn_%',
            title='Taux de churn par segment',
            color='Taux_Churn_%',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

# Distribution des niveaux de risque
st.markdown("---")
st.markdown("### Distribution des Niveaux de Risque")

col1, col2 = st.columns(2)

with col1:
    risk_counts = df['Risk_Level'].value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Répartition des niveaux de risque',
        color=risk_counts.index,
        color_discrete_map={
            'Faible': 'green',
            'Moyen': 'orange',
            'Élevé': 'red'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        title='Nombre de clients par niveau de risque',
        labels={'x': 'Niveau de risque', 'y': 'Nombre de clients'},
        color=risk_counts.index,
        color_discrete_map={
            'Faible': 'green',
            'Moyen': 'orange',
            'Élevé': 'red'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# Top features (si NPS et Age disponibles)
if 'NPS' in df.columns and 'Age' in df.columns:
    st.markdown("---")
    st.markdown("### Analyse des Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df,
            x='NPS',
            y='Probability_Churn',
            color='Risk_Level',
            title='NPS vs Probabilité de Churn',
            color_discrete_map={
                'Faible': 'green',
                'Moyen': 'orange',
                'Élevé': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df,
            x='Age',
            y='Probability_Churn',
            color='Risk_Level',
            title='Age vs Probabilité de Churn',
            color_discrete_map={
                'Faible': 'green',
                'Moyen': 'orange',
                'Élevé': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

# Matrice de confusion (si ChurnLabel disponible)
if 'ChurnLabel' in df.columns:
    st.markdown("---")
    st.markdown("### Matrice de Confusion")
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(df['ChurnLabel'], df['Prediction'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Afficher la matrice
        cm_df = pd.DataFrame(
            cm,
            index=['Réel: Non-Churn', 'Réel: Churn'],
            columns=['Prédit: Non-Churn', 'Prédit: Churn']
        )
        st.dataframe(cm_df, use_container_width=True)
    
    with col2:
        # Métriques de performance
        accuracy = (df['Prediction'] == df['ChurnLabel']).mean() * 100
        precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
        
        st.metric("Accuracy", f"{accuracy:.1f}%")
        st.metric("Precision", f"{precision:.2%}")
        st.metric("Recall", f"{recall:.2%}")

st.markdown("---")
st.success("Utilisez ces insights pour prendre des décisions stratégiques !")

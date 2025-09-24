import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Fonction pour charger les données d'accidentologie
@st.cache_data
def load_accident_data():
    """Charger et prétraiter les données d'accidentologie"""
    try:
        # Vérifier si le fichier Parquet existe
        parquet_file = 'accidentologie.parquet'
        
        # Lecture du fichier Parquet
        df = pd.read_parquet(parquet_file)
        
        # Création d'une colonne de gravité combinée si elle n'existe pas déjà
        if 'gravite_combinee' not in df.columns:
            df['gravite_combinee'] = 'Blessé léger'
            df.loc[df['Blessés hospitalisés'] > 0, 'gravite_combinee'] = 'Blessé hospitalisé'
            df.loc[df['Tué'] > 0, 'gravite_combinee'] = 'Tué'
        
        # Renommage des colonnes pour correspondre à nos besoins
        column_mapping = {
            'Date': 'date',
            'Latitude': 'latitude',
            'Longitude': 'longitude',
            'Mode': 'type_usager',
            'Arrondissement': 'arrondissement',
            'Id accident': 'id_accident',
            'Gravité': 'gravite'
        }
        
        # Renommage des colonnes si nécessaire
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Extraction des informations temporelles
        df['date_heure'] = pd.to_datetime(df['date'])
        df['heure'] = df['date_heure'].dt.hour
        df['mois'] = df['date_heure'].dt.month
        df['annee'] = df['date_heure'].dt.year
        df['mois_annee'] = df['date_heure'].dt.strftime('%Y-%m')
        df['jour_semaine'] = df['date_heure'].dt.day_name()
        df['mois_nom'] = df['date_heure'].dt.strftime('%B')
        
        # Filtrage des lignes avec des coordonnées valides
        df = df.dropna(subset=['latitude', 'longitude'])
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        return None

def run_simple_accidentologie():
    """Version simplifiée de l'application d'accidentologie"""
    
    # Chargement des données
    df = load_accident_data()
    
    if df is not None:
        st.success(f"✅ Données chargées avec succès : {len(df):,} accidents")
        
        # Sidebar pour les filtres
        st.sidebar.header("Filtres")
        
        # Sélection de la période
        mois_annees = sorted(df['mois_annee'].unique())
        periode_selectionnee = st.sidebar.select_slider(
            "Sélectionner la période",
            options=mois_annees,
            value=(mois_annees[0], mois_annees[-1])
        )
        
        # Filtrage par période
        mask_periode = (df['mois_annee'] >= periode_selectionnee[0]) & (df['mois_annee'] <= periode_selectionnee[1])
        df_periode = df[mask_periode]
        
        # Affichage de la période sélectionnée
        st.sidebar.info(f"Période sélectionnée : de {periode_selectionnee[0]} à {periode_selectionnee[1]}")
        
        # Sélection du type d'analyse
        analysis_type = st.sidebar.selectbox(
            "Type d'analyse",
            ["Évolution temporelle", "Analyse par arrondissement", "Statistiques générales"]
        )

        if analysis_type == "Évolution temporelle":
            st.header("Évolution temporelle des accidents")
            
            # Préparation des données pour l'évolution temporelle
            df_evolution = df_periode.copy()
            df_evolution['mois_annee'] = df_evolution['date'].dt.strftime('%Y-%m')
            
            # Groupement par mois et type de gravité
            evolution_data = df_evolution.groupby(['mois_annee', 'gravite_combinee']).size().reset_index(name='count')
            evolution_data = evolution_data.sort_values('mois_annee')
            
            # Création du graphique d'évolution
            fig_evolution = go.Figure()
            
            # Ajout des lignes pour chaque niveau de gravité
            for gravite, color in [('Tué', 'red'), ('Blessé hospitalisé', 'orange'), ('Blessé léger', 'yellow')]:
                data_gravite = evolution_data[evolution_data['gravite_combinee'] == gravite]
                
                fig_evolution.add_trace(go.Scatter(
                    x=data_gravite['mois_annee'],
                    y=data_gravite['count'],
                    name=gravite,
                    mode='lines+markers',
                    line=dict(width=3, color=color),
                    marker=dict(size=8, color=color)
                ))
            
            # Mise à jour de la mise en page
            fig_evolution.update_layout(
                title="Évolution mensuelle des accidents par gravité",
                xaxis_title="Mois",
                yaxis_title="Nombre d'accidents",
                height=500,
                showlegend=True,
                plot_bgcolor='white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Ajout d'une grille
            fig_evolution.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            fig_evolution.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)

        elif analysis_type == "Analyse par arrondissement":
            st.header("Analyse par arrondissement")

            # Nettoyage des numéros d'arrondissements
            def clean_arrondissement(arr):
                if isinstance(arr, str):
                    arr = arr.lstrip('0')
                    return arr if arr else '1'
                return str(arr)

            # Nettoyage des arrondissements dans le DataFrame
            df_periode['arrondissement'] = df_periode['arrondissement'].apply(clean_arrondissement)
            
            # Sélection de l'arrondissement (en haut de la page)
            arr_analysis = st.selectbox(
                "Sélectionner un arrondissement",
                options=sorted(df_periode['arrondissement'].unique(), key=int),
                format_func=lambda x: f"Arrondissement {x}"
            )
            
            # Filtrage des données pour l'arrondissement sélectionné
            df_filtered = df_periode[df_periode['arrondissement'] == arr_analysis]
            
            if not df_filtered.empty:
                # Métriques principales pour l'arrondissement
                st.subheader(f"Statistiques - Arrondissement {arr_analysis}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total accidents",
                        len(df_filtered)
                    )
                with col2:
                    morts = len(df_filtered[df_filtered['gravite_combinee'] == 'Tué'])
                    st.metric(
                        "Accidents mortels",
                        morts
                    )
                with col3:
                    blesses = len(df_filtered[df_filtered['gravite_combinee'] == 'Blessé hospitalisé'])
                    st.metric(
                        "Blessés hospitalisés",
                        blesses
                    )
                
                # Graphique de répartition par gravité
                st.subheader(f"Répartition par gravité - Arrondissement {arr_analysis}")
                
                gravity_counts = df_filtered['gravite_combinee'].value_counts()
                fig_gravity = px.pie(
                    values=gravity_counts.values,
                    names=gravity_counts.index,
                    title=f"Répartition des accidents par gravité - Arrondissement {arr_analysis}",
                    color_discrete_map={
                        'Tué': 'red',
                        'Blessé hospitalisé': 'orange',
                        'Blessé léger': 'yellow'
                    }
                )
                st.plotly_chart(fig_gravity, use_container_width=True)
                
            else:
                st.warning(f"Aucun accident trouvé dans l'arrondissement {arr_analysis} avec les filtres sélectionnés.")

        elif analysis_type == "Statistiques générales":
            st.header("Statistiques générales")
            
            # Métriques globales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total accidents", len(df_periode))
            with col2:
                morts = len(df_periode[df_periode['gravite_combinee'] == 'Tué'])
                st.metric("Accidents mortels", morts)
            with col3:
                blesses = len(df_periode[df_periode['gravite_combinee'] == 'Blessé hospitalisé'])
                st.metric("Blessés hospitalisés", blesses)
            with col4:
                legers = len(df_periode[df_periode['gravite_combinee'] == 'Blessé léger'])
                st.metric("Blessés légers", legers)
            
            # Graphique de répartition par type d'usager
            st.subheader("Répartition par type d'usager")
            usager_counts = df_periode['type_usager'].value_counts()
            fig_usager = px.bar(
                x=usager_counts.index,
                y=usager_counts.values,
                title="Nombre d'accidents par type d'usager",
                labels={'x': 'Type d\'usager', 'y': 'Nombre d\'accidents'}
            )
            fig_usager.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_usager, use_container_width=True)
            
            # Graphique de répartition par gravité
            st.subheader("Répartition par gravité")
            gravity_counts = df_periode['gravite_combinee'].value_counts()
            fig_gravity = px.pie(
                values=gravity_counts.values,
                names=gravity_counts.index,
                title="Répartition des accidents par gravité",
                color_discrete_map={
                    'Tué': 'red',
                    'Blessé hospitalisé': 'orange',
                    'Blessé léger': 'yellow'
                }
            )
            st.plotly_chart(fig_gravity, use_container_width=True)
    else:
        st.error("Impossible de charger les données d'accidentologie.")

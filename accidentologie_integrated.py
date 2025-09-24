import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import HeatMap
import geopandas as gpd
import json
import requests
from folium.plugins import MarkerCluster
import re
import os
import time
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX
import traceback

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

# Fonction pour charger les données météo
@st.cache_data
def load_weather_data():
    """Charge les données météo depuis le fichier CSV"""
    try:
        weather_df = pd.read_csv('data_meteo.csv')
        # Conversion de la colonne date en datetime64[ns]
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        return weather_df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données météo : {str(e)}")
        return None

def create_accident_map(df):
    """Crée une carte des accidents"""
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=12,
                    tiles='cartodbpositron')
        
    # Création d'un cluster de marqueurs
    marker_cluster = MarkerCluster(
        options={
            'maxClusterRadius': 50,
            'disableClusteringAtZoom': 15
        }
    )

    # Ajout de la carte de chaleur
    heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
    if heat_data:
        HeatMap(
            heat_data,
            name="Carte de chaleur",
            min_opacity=0.3,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={
                0.4: 'blue',
                0.6: 'yellow',
                0.8: 'orange',
                1.0: 'red'
            }
        ).add_to(m)
    
    # Définition des couleurs pour la gravité
    colors = {
        'Tué': 'red',
        'Blessé hospitalisé': 'orange',
        'Blessé léger': 'yellow'
    }

    # Regrouper les accidents très proches pour réduire le nombre de marqueurs
    df_grouped = df.round({'latitude': 4, 'longitude': 4}).groupby(['latitude', 'longitude']).agg({
        'date': 'count',
        'type_usager': lambda x: ', '.join(x.unique()),
        'gravite_combinee': lambda x: x.value_counts().index[0],
        'arrondissement': 'first',
        'Adresse': 'first'
    }).reset_index()
    
    # Limiter le nombre de points si nécessaire
    max_points = 1000  # Nombre maximum de points à afficher
    if len(df_grouped) > max_points:
        # Échantillonnage stratifié par gravité
        df_sample = pd.DataFrame()
        for gravite in colors.keys():
            subset = df_grouped[df_grouped['gravite_combinee'] == gravite]
            sample_size = int(max_points * (len(subset) / len(df_grouped)))
            if not subset.empty:
                df_sample = pd.concat([df_sample, subset.sample(n=min(sample_size, len(subset)))])
        df_grouped = df_sample

    # Ajout des points à la carte
    for idx, row in df_grouped.iterrows():
        # Création du popup avec le nombre d'accidents au même endroit
        popup_html = f"""
        <b>Groupe d'accidents</b><br>
        Adresse: {row['Adresse']}<br>
        Nombre d'accidents: {row['date']}<br>
        Types d'usagers: {row['type_usager']}<br>
        Gravité principale: {row['gravite_combinee']}<br>
        Arrondissement: {row['arrondissement']}
        """

        # Création du marqueur
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color=colors[row['gravite_combinee']],
            fill=True,
            fillOpacity=0.7,
            popup=popup_html
        ).add_to(marker_cluster)
    
    marker_cluster.add_to(m)
    
    # Ajout du contrôle des couches
    folium.LayerControl().add_to(m)
    
    return m

def run_accidentologie_app():
    """Fonction principale pour exécuter l'application d'accidentologie"""
    
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
            ["Carte des accidents", "Évolution temporelle", "Etude par arrondissement"]
        )

        if analysis_type == "Carte des accidents":
            st.header("Cartographie des accidents à Paris")
                
            # Filtres dans la barre latérale
            st.sidebar.subheader("Filtres de la carte")
                
            # Sélection des catégories d'usagers
            categories = sorted(df_periode['type_usager'].unique())
            selected_categories = st.sidebar.multiselect(
                "Types d'usagers",
                options=categories,
                default=categories,
                key='categories_filter'
            )
                
            # Sélection des niveaux de gravité
            gravity_levels = ['Tué', 'Blessé hospitalisé', 'Blessé léger']
            selected_gravity = st.sidebar.multiselect(
                "Niveaux de gravité",
                options=gravity_levels,
                default=gravity_levels,
                key='gravity_filter'
            )

            if not selected_categories or not selected_gravity:
                st.warning("Veuillez sélectionner au moins une catégorie d'usager et un niveau de gravité.")
            else:
                # Filtrage des données
                filtered_data = df_periode[
                    (df_periode['type_usager'].isin(selected_categories)) & 
                    (df_periode['gravite_combinee'].isin(selected_gravity))
                ]

                # Affichage de la carte
                m = create_accident_map(filtered_data)
                st.components.v1.html(m._repr_html_(), height=600)
                    
                # Statistiques de la sélection dans un expander
                with st.expander("Voir les statistiques"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total accidents",
                            len(filtered_data)
                        )
                    with col2:
                        st.metric(
                            "Types d'usagers",
                            len(filtered_data['type_usager'].unique())
                        )
                    with col3:
                        gravity_counts = filtered_data['gravite_combinee'].value_counts()
                        most_common_gravity = gravity_counts.index[0] if not gravity_counts.empty else "N/A"
                        st.metric(
                            "Gravité la plus fréquente",
                            most_common_gravity,
                            f"{gravity_counts.iloc[0] if not gravity_counts.empty else 0} accidents"
                        )

        elif analysis_type == "Évolution temporelle":
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

        elif analysis_type == "Etude par arrondissement":
            st.header("Etude par arrondissement")

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
    else:
        st.error("Impossible de charger les données d'accidentologie.")

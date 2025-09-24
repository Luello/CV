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

# Définition de la fonction prepare_temporal_features au début du fichier, après les imports
def prepare_temporal_features(df):
    """
    Prépare les features temporelles cycliques pour mieux capturer la saisonnalité.
    
    Les features cycliques (sin et cos) permettent de représenter des variables cycliques
    comme les mois ou les jours de manière continue, sans "saut" entre la fin et le début
    du cycle. Par exemple, pour les mois :
    - Décembre (12) et Janvier (1) sont proches dans le temps mais éloignés en valeur numérique
    - La transformation cyclique les rend proches dans l'espace des features
    """
    df = df.copy()
    
    # Features temporelles basiques
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # Features cycliques
    # Pour les mois (période = 12)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Pour les jours du mois (période = 31)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    
    # Pour les jours de la semaine (période = 7)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    return df

def train_prophet_model(daily_data):
    """Entraîne un modèle Prophet sur les données quotidiennes."""
    # Préparation des données pour Prophet
    df_prophet = pd.DataFrame({
        'ds': pd.to_datetime(daily_data['date']),
        'y': daily_data['accidents']
    })
    
    # Configuration du modèle Prophet
    model = Prophet(
        yearly_seasonality=20,
        weekly_seasonality=10,
        daily_seasonality=False,
        holidays_prior_scale=10,
        seasonality_prior_scale=15,
        changepoint_prior_scale=0.05,
        changepoint_range=0.95,
        interval_width=0.95
    )
    
    # Ajout des jours fériés français
    model.add_country_holidays(country_name='FR')
    
    # Copie et préparation des données
    data_for_prophet = daily_data.copy()
    
    # Gestion des valeurs manquantes et ajout des régresseurs
    for col in ['tavg', 'prcp', 'snow', 'wspd', 'trafic_debit', 'trafic_concentration']:
        if col in data_for_prophet.columns:
            # Remplacement des valeurs infinies et traitement spécial pour la neige
            data_for_prophet[col] = data_for_prophet[col].replace([np.inf, -np.inf], np.nan)
            
            # Traitement spécial pour la colonne neige
            if col == 'snow':
                # Remplacer les NaN par 0 pour la neige (absence de neige)
                data_for_prophet[col].fillna(0, inplace=True)
            else:
                # Pour les autres colonnes, utiliser une interpolation plus robuste
                # D'abord essayer une interpolation linéaire
                data_for_prophet[col] = data_for_prophet[col].interpolate(method='linear')
                # Puis utiliser la moyenne de la période pour les valeurs encore manquantes
                if data_for_prophet[col].isna().any():
                    # Calculer les moyennes mensuelles
                    monthly_means = data_for_prophet.groupby(data_for_prophet['date'].dt.month)[col].transform('mean')
                    data_for_prophet[col].fillna(monthly_means, inplace=True)
                    # Si encore des NaN, utiliser la moyenne globale
                    if data_for_prophet[col].isna().any():
                        data_for_prophet[col].fillna(data_for_prophet[col].mean(), inplace=True)
            
            # Vérification finale et ajout du régresseur
            if not data_for_prophet[col].isna().any():
                df_prophet[col] = data_for_prophet[col]
                model.add_regressor(col, mode='multiplicative')
    
    # Ajout de features dérivées
    df_prophet['weekend'] = df_prophet['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    df_prophet['summer_holiday'] = df_prophet['ds'].dt.month.isin([7, 8]).astype(int)
    df_prophet['winter_holiday'] = df_prophet['ds'].dt.month.isin([12, 1]).astype(int)
    
    model.add_regressor('weekend', mode='multiplicative')
    model.add_regressor('summer_holiday', mode='multiplicative')
    model.add_regressor('winter_holiday', mode='multiplicative')
    
    # Entraînement du modèle
    model.fit(df_prophet)
    
    return model, df_prophet

def train_sarima_model(train_data, test_data):
    """Entraîne un modèle SARIMA sur les données quotidiennes."""
    try:
        # Trier les données par date
        train_data = train_data.sort_values('date').copy()
        
        # S'assurer que les dates sont en datetime
        train_data['date'] = pd.to_datetime(train_data['date'])
        test_data['date'] = pd.to_datetime(test_data['date'])
        
        # Créer la série temporelle
        train_ts = pd.Series(
            train_data['accidents'].values,
            index=train_data['date']
        )
        
        st.info("Entraînement du modèle SARIMA en cours...")
        
        # Configuration du modèle SARIMA
        model = SARIMAX(
            train_ts,
            order=(1, 1, 1),  # Simplifier l'ordre du modèle
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Entraînement du modèle
        results = model.fit(disp=False)
        
        st.success("Modèle SARIMA entraîné avec succès!")
        
        # Créer l'index de prédiction
        pred_index = pd.date_range(
            start=test_data['date'].min(),
            periods=len(test_data),
            freq='D'
        )
        
        # Prédictions
        forecast = results.forecast(steps=len(test_data))
        forecast_mean = pd.Series(forecast, index=pred_index)
        
        # Intervalles de confiance
        forecast_ci = pd.DataFrame(
            np.column_stack([
                forecast_mean - 1.96 * results.params['sigma2']**0.5,
                forecast_mean + 1.96 * results.params['sigma2']**0.5
            ]),
            index=pred_index,
            columns=['lower', 'upper']
        )
        
        return forecast_mean, forecast_ci, results
        
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement du modèle SARIMA : {str(e)}")
        st.text("Détails des données :")
        st.text(f"Période d'entraînement : {train_data['date'].min()} à {train_data['date'].max()}")
        st.text(f"Période de test : {test_data['date'].min()} à {test_data['date'].max()}")
        st.text(f"Nombre de données d'entraînement : {len(train_data)}")
        st.text(f"Nombre de données de test : {len(test_data)}")
        raise e

def train_xgboost_model(train_data, test_data_2022, test_data_2023):
    """Entraîne un modèle XGBoost sur les données quotidiennes."""
    # Définition des colonnes pour le modèle
    feature_columns = [
        'tavg', 'prcp', 'snow', 'wspd', 'trafic_debit', 'trafic_concentration',
        'month_sin', 'month_cos', 'day_sin', 'day_cos', 'dayofweek_sin', 'dayofweek_cos'
    ]
    
    # Préparation des données d'entraînement
    train_features = train_data.copy()
    test_features_2022 = test_data_2022.copy()
    test_features_2023 = test_data_2023.copy()
    
    # Gestion des valeurs manquantes
    for col in feature_columns:
        if col in train_features.columns:
            # Traitement spécial pour la neige
            if col == 'snow':
                train_features[col] = train_features[col].fillna(0)
                test_features_2022[col] = test_features_2022[col].fillna(0)
                test_features_2023[col] = test_features_2023[col].fillna(0)
            else:
                # Pour les autres colonnes, utiliser la moyenne mensuelle
                monthly_means = train_features.groupby('month')[col].transform('mean')
                train_features[col] = train_features[col].fillna(monthly_means)
                test_features_2022[col] = test_features_2022[col].fillna(monthly_means)
                test_features_2023[col] = test_features_2023[col].fillna(monthly_means)
    
    # Préparation des données d'entraînement
    X_train = train_features[feature_columns]
    y_train = train_features['accidents']
    X_test_2022 = test_features_2022[feature_columns]
    X_test_2023 = test_features_2023[feature_columns]
    
    # Configuration et entraînement du modèle XGBoost
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions pour 2022 et 2023
    predictions_2022 = model.predict(X_test_2022)
    predictions_2023 = model.predict(X_test_2023)
    
    # Importance des features
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, predictions_2022, predictions_2023, feature_importance

# Fonctions utilitaires pour les cartes
def create_monthly_heatmap(df):
    """Crée une carte de chaleur pour les données mensuelles"""
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=13,
                  tiles='cartodbpositron',
                  max_bounds=True,
                  min_zoom=12,
                  max_zoom=16)
    
    # Ajout de la carte de chaleur
    heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
    if heat_data:
        HeatMap(
            heat_data,
            radius=15,
            blur=20,
            min_opacity=0.4,
            gradient={
                0.4: 'blue',
                0.6: 'yellow',
                0.8: 'orange',
                1.0: 'red'
            }
        ).add_to(m)
    
    return m

def create_yearly_heatmap(df):
    """Crée une carte de chaleur pour les données annuelles"""
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=13,
                  tiles='cartodbpositron',
                  max_bounds=True,
                  min_zoom=12,
                  max_zoom=16)
    
    # Ajout de la carte de chaleur
    heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
    if heat_data:
        HeatMap(
            heat_data,
            radius=15,
            blur=20,
            min_opacity=0.4,
            gradient={
                0.4: 'blue',
                0.6: 'yellow',
                0.8: 'orange',
                1.0: 'red'
            }
        ).add_to(m)
    
    return m

# Configuration de la page
st.set_page_config(
    page_title="Analyse d'Accidentologie à Paris",
    page_icon="🚨",
    layout="wide"
)

# Style personnalisé
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #e63946; color: white; }
    .stSelectbox { background-color: white; }
    .map-container { height: 800px !important; }
    </style>
    """, unsafe_allow_html=True)

# Titre de l'application
st.title("🚨 Analyse d'Accidentologie à Paris")
st.markdown("*Analyse spatiale et temporelle des accidents à Paris*")

# Fonction pour charger les données météo
@st.cache_data
def load_weather_data():
    """
    Charge les données météo depuis le fichier CSV
    """
    try:
        weather_df = pd.read_csv('data_meteo.csv')
        # Conversion de la colonne date en datetime64[ns]
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        return weather_df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données météo : {str(e)}")
        return None

# Fonction pour créer les features cycliques
def create_cyclical_features(df, col_name, period):
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / period)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / period)
    return df

@st.cache_data
def load_data():
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
        
        st.success(f"✅ Données chargées avec succès : {len(df):,} accidents")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        return None

# Chargement des données
df = load_data()

if df is not None:
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
        ["Carte des accidents", "Évolution temporelle animée", "Etude par arrondissement", "Analyses et Prédictions"]
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

            # Paramètres de la heatmap
        st.sidebar.subheader("Paramètres de la carte de chaleur")
        show_heatmap = st.sidebar.checkbox("Afficher la carte de chaleur", value=True)
        
        if show_heatmap:
            heatmap_radius = st.sidebar.slider(
                "Rayon de la zone de chaleur",
                min_value=10,
                max_value=50,
                value=25,
                help="Ajuste la taille des zones de chaleur"
            )
            
            heatmap_blur = st.sidebar.slider(
                "Flou de la carte de chaleur",
                min_value=5,
                max_value=30,
                value=15,
                help="Ajuste le niveau de flou entre les zones"
            )
            
            heatmap_intensity = st.sidebar.slider(
                "Intensité de la carte de chaleur",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Ajuste l'intensité globale de la carte de chaleur"
            )
            
        # Paramètres des marqueurs
        st.sidebar.subheader("Paramètres des marqueurs")
        marker_size = st.sidebar.slider(
            "Taille des marqueurs",
            min_value=3,
            max_value=15,
            value=8,
            help="Ajuste la taille des points sur la carte"
        )
        
        marker_opacity = st.sidebar.slider(
            "Opacité des marqueurs",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Ajuste la transparence des points"
            )

        if not selected_categories or not selected_gravity:
            st.warning("Veuillez sélectionner au moins une catégorie d'usager et un niveau de gravité.")
        else:
            # Filtrage des données
            filtered_data = df_periode[
                (df_periode['type_usager'].isin(selected_categories)) & 
(df_periode['gravite_combinee'].isin(selected_gravity))
]

            # Création de la carte
            def create_accident_map(df):
                m = folium.Map(location=[48.8566, 2.3522], zoom_start=12,
                                tiles='cartodbpositron')
                    
                # Création d'un cluster de marqueurs
                marker_cluster = MarkerCluster(
                    options={
                        'maxClusterRadius': 50,
                        'disableClusteringAtZoom': 15
                    }
                )

                # Ajout de la carte de chaleur si activée
                if show_heatmap:
                    heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
                    if heat_data:
                        HeatMap(
                            heat_data,
                            name="Carte de chaleur",
                            min_opacity=0.3 * heatmap_intensity,
                            max_zoom=18,
                            radius=heatmap_radius,
                            blur=heatmap_blur,
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
                max_points = 2000  # Nombre maximum de points à afficher
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

                    # Ajustement de la taille du marqueur selon le nombre d'accidents
                    size = min(marker_size * (1 + np.log1p(row['date'])), marker_size * 3)

                    # Création du marqueur
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=size,
                        color=colors[row['gravite_combinee']],
                        fill=True,
                        fillOpacity=marker_opacity,
                        popup=popup_html
                    ).add_to(marker_cluster)
                
                marker_cluster.add_to(m)
                
                # Ajout du contrôle des couches
                folium.LayerControl().add_to(m)
                
                return m

            # Affichage de la carte
            m = create_accident_map(filtered_data)
            st.components.v1.html(m._repr_html_(), height=800)
                
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

    elif analysis_type == "Évolution temporelle animée":
        st.header("Évolution temporelle animée des accidents")
        
        # Ajout des filtres dans la barre latérale
        st.sidebar.subheader("Filtres de l'animation")
        
        # Sélection des types d'usagers
        types_usagers = sorted(df_periode['type_usager'].unique())
        selected_types_usagers = st.sidebar.multiselect(
            "Types d'usagers",
            options=types_usagers,
            default=types_usagers,
            key='types_usagers_filter_anim'
        )
        
        # Sélection des niveaux de gravité
        niveaux_gravite = ['Tué', 'Blessé hospitalisé', 'Blessé léger']
        selected_gravite = st.sidebar.multiselect(
            "Niveaux de gravité",
            options=niveaux_gravite,
            default=niveaux_gravite,
            key='gravite_filter_anim'
        )
        
        # Nettoyage et sélection des arrondissements
        def clean_arrondissement(arr):
            if isinstance(arr, str):
                arr = arr.lstrip('0')
                return arr if arr else '1'
            return str(arr)
        
        df_periode['arrondissement'] = df_periode['arrondissement'].apply(clean_arrondissement)
        arrondissements = sorted(df_periode['arrondissement'].unique(), key=int)
        selected_arrondissements = st.sidebar.multiselect(
            "Arrondissements",
            options=arrondissements,
            default=arrondissements,
            key='arrondissements_filter_anim'
        )
        
        # Application des filtres
        if selected_types_usagers and selected_gravite and selected_arrondissements:
            df_filtered = df_periode[
                (df_periode['type_usager'].isin(selected_types_usagers)) &
                (df_periode['gravite_combinee'].isin(selected_gravite)) &
                (df_periode['arrondissement'].isin(selected_arrondissements))
            ]
        else:
            st.warning("Veuillez sélectionner au moins un élément pour chaque filtre.")
            df_filtered = df_periode
        
        # Création des sous-onglets
        tab_mois, tab_annee = st.tabs(["Évolution mensuelle", "Évolution annuelle"])
        
        with tab_mois:
            st.subheader("Évolution moyenne mensuelle (toutes années confondues)")
            
            # Préparation des données mensuelles avec les filtres appliqués
            df_mois = df_filtered.copy()
            df_mois['mois'] = df_mois['date_heure'].dt.month
            df_mois['mois_nom'] = df_mois['date_heure'].dt.strftime('%B')
            df_mois['annee'] = df_mois['date_heure'].dt.year
            
            # Création de l'ordre des mois pour le tri
            mois_ordre = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            df_mois['mois_num'] = df_mois['mois_nom'].map(mois_ordre)
            
            # Liste des mois pour le slider
            mois_list = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
            
            # Initialisation de l'index du mois dans le state si pas déjà fait
            if 'month_index' not in st.session_state:
                st.session_state.month_index = 0
                st.session_state.is_playing_month = False
            
            # Contrôles pour l'animation
            col_slider, col_play = st.columns([4, 1])
            
            with col_slider:
                selected_month = st.select_slider(
                    "Sélectionner le mois",
                    options=mois_list,
                    value=mois_list[st.session_state.month_index]
                )
                st.session_state.month_index = mois_list.index(selected_month)
            
            with col_play:
                if st.button('▶ Lecture' if not st.session_state.is_playing_month else '⏸ Pause', key='play_month'):
                    st.session_state.is_playing_month = not st.session_state.is_playing_month
            
            # Création de la carte
            st.subheader("Carte des accidents")
            df_month = df_mois[df_mois['mois_nom'] == selected_month]
            m = create_monthly_heatmap(df_month)
            st.components.v1.html(m._repr_html_(), height=800)
            
            # Animation
            if st.session_state.is_playing_month:
                st.session_state.month_index = (st.session_state.month_index + 1) % len(mois_list)
                time.sleep(0.5)
                st.rerun()
            
            # Calcul des statistiques mensuelles (tous filtres confondus)
            monthly_stats = df_mois.groupby(['annee', 'mois_nom', 'mois_num']).agg({
                'id_accident': 'count'
            }).reset_index()
            
            # Tri des données
            monthly_stats = monthly_stats.sort_values(['annee', 'mois_num'])
            
            # Création du graphique de comparaison
            st.subheader("Comparaison mensuelle entre les années")
            
            # Définition d'une palette de couleurs distinctes
            color_map = {
                2017: '#1f77b4',  # Bleu foncé
                2018: '#ff7f0e',  # Orange
                2019: '#2ca02c',  # Vert foncé
                2020: '#d62728',  # Rouge
                2021: '#9467bd',  # Violet
                2022: '#8c564b',  # Marron
            }
            
            fig_monthly_comparison = px.line(
                monthly_stats,
                x='mois_nom',
                y='id_accident',
                color='annee',
                title="Évolution mensuelle des accidents par année",
                category_orders={
                    'mois_nom': mois_list
                },
                labels={
                    'mois_nom': 'Mois',
                    'id_accident': "Nombre d'accidents",
                    'annee': 'Année'
                },
                color_discrete_map=color_map
            )
            
            # Personnalisation du graphique
            fig_monthly_comparison.update_layout(
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',  # Fond blanc
                paper_bgcolor='white'  # Fond blanc autour du graphique
            )
            
            # Amélioration de la grille
            fig_monthly_comparison.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            fig_monthly_comparison.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            
            # Ajout des points et personnalisation des lignes
            fig_monthly_comparison.update_traces(
                mode='lines+markers',
                line=dict(width=3),  # Lignes plus épaisses
                marker=dict(size=8)   # Points plus gros
            )
            
            # Affichage du graphique
            st.plotly_chart(fig_monthly_comparison, use_container_width=True)
            
            # Statistiques du mois sélectionné
            st.subheader(f"Statistiques pour {selected_month}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_accidents = len(df_month)
                st.metric("Nombre d'accidents", str(total_accidents))
            
            with col2:
                morts = len(df_month[df_month['gravite_combinee'] == 'Tué'])
                st.metric("Nombre de décès", str(morts))
            
            with col3:
                blesses = len(df_month[df_month['gravite_combinee'] == 'Blessé hospitalisé'])
                st.metric("Nombre de blessés graves", str(blesses))
            
            # Ajout de statistiques détaillées
            with st.expander("Statistiques détaillées par mois"):
                # Création d'un tableau croisé dynamique pour les statistiques
                stats_mensuelles = monthly_stats.pivot_table(
                    index=['mois_nom', 'mois_num'],
                    columns=['annee'],
                    values='id_accident',
                    aggfunc='sum'
                ).reset_index()
                
                # Tri des mois dans l'ordre chronologique
                stats_mensuelles = stats_mensuelles.sort_values('mois_num')
                stats_mensuelles = stats_mensuelles.drop('mois_num', axis=1)
                stats_mensuelles = stats_mensuelles.rename(columns={'mois_nom': 'Mois'})
                
                # Formatage et affichage du tableau
                st.dataframe(
                    stats_mensuelles.style.format({
                        col: '{:.0f}' for col in stats_mensuelles.columns if col != 'Mois'
                    }),
                    use_container_width=True
                )
        
        with tab_annee:
            st.subheader("Évolution annuelle des accidents")
            
            # Préparation des données annuelles avec les filtres appliqués
            df_annee = df_filtered.copy()
            df_annee['annee'] = df_annee['date_heure'].dt.year
            
            # Calcul des statistiques annuelles
            yearly_stats = df_annee.groupby(['annee', 'gravite_combinee']).agg({
                'id_accident': 'count'
            }).reset_index()
            
            # Création des colonnes pour les contrôles
            col_slider, col_play = st.columns([4, 1])
            
            # Initialisation de l'index de l'année dans le state si pas déjà fait
            if 'year_index' not in st.session_state:
                st.session_state.year_index = 0
                st.session_state.is_playing_year = False
            
            # Liste des années
            annees_list = sorted(df_annee['annee'].unique())
            
            with col_slider:
                selected_year = st.select_slider(
                    "Sélectionner l'année",
                    options=annees_list,
                    value=annees_list[st.session_state.year_index]
                )
                # Mise à jour de l'index quand le slider change manuellement
                st.session_state.year_index = annees_list.index(selected_year)
            
            with col_play:
                # Bouton pour démarrer/arrêter l'animation
                if st.button('▶ Lecture' if not st.session_state.is_playing_year else '⏸ Pause', key='play_year'):
                    st.session_state.is_playing_year = not st.session_state.is_playing_year
            
            # Création des conteneurs pour l'affichage dynamique
            map_placeholder = st.empty()
            
            # Création de la carte pour l'année sélectionnée avec les données filtrées
            df_year = df_annee[df_annee['annee'] == selected_year]
            m = create_yearly_heatmap(df_year)
            
            # Affichage de la carte
            with map_placeholder:
                st.components.v1.html(
                    m._repr_html_(),
                    height=800,
                    scrolling=False
                )
            
            # Logique d'animation avec transition plus fluide
            if st.session_state.is_playing_year:
                # Incrémentation de l'index pour la prochaine itération
                st.session_state.year_index = (st.session_state.year_index + 1) % len(annees_list)
                time.sleep(1)
                st.rerun()
            
            # Création des colonnes pour les métriques et le graphique
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Graphique en barres pour la répartition des gravités
                data_annee = yearly_stats[yearly_stats['annee'] == selected_year]
                fig_annee = px.bar(
                    data_annee,
                    x='gravite_combinee',
                    y='id_accident',
                    color='gravite_combinee',
                    title=f"Répartition des accidents en {selected_year}",
                    labels={
                        'id_accident': "Nombre d'accidents",
                        'gravite_combinee': 'Gravité'
                    },
                    color_discrete_map={
                        'Tué': 'red',
                        'Blessé hospitalisé': 'orange',
                        'Blessé léger': 'yellow'
                    }
                )
                st.plotly_chart(fig_annee, use_container_width=True)
            
            with col2:
                # Métriques clés
                total_accidents = data_annee['id_accident'].sum()
                st.metric("Nombre total d'accidents", f"{total_accidents:,.0f}")
                
                morts = data_annee[data_annee['gravite_combinee'] == 'Tué']['id_accident'].iloc[0] if len(data_annee[data_annee['gravite_combinee'] == 'Tué']) > 0 else 0
                st.metric("Nombre de décès", f"{morts:,.0f}")
                
                blesses = data_annee[data_annee['gravite_combinee'] == 'Blessé hospitalisé']['id_accident'].iloc[0] if len(data_annee[data_annee['gravite_combinee'] == 'Blessé hospitalisé']) > 0 else 0
                st.metric("Nombre de blessés graves", f"{blesses:,.0f}")
            
            # Comparaison entre les années
            st.subheader("Comparaison entre les années")
            
            # Préparation des données pour la comparaison annuelle
            yearly_comparison = df_filtered.groupby([
                df_filtered['date_heure'].dt.year,
                'gravite_combinee'
            ]).agg({
                'id_accident': 'count'
            }).reset_index()
            
            # Création du graphique de comparaison
            fig_yearly = px.line(
                yearly_comparison,
                x='date_heure',
                y='id_accident',
                color='gravite_combinee',
                title="Évolution du nombre d'accidents par année",
                labels={
                    'date_heure': 'Année',
                    'id_accident': "Nombre d'accidents",
                    'gravite_combinee': 'Gravité'
                },
                color_discrete_map={
                    'Tué': 'red',
                    'Blessé hospitalisé': 'orange',
                    'Blessé léger': 'yellow'
                }
            )
            
            # Personnalisation du graphique
            fig_yearly.update_layout(
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Ajout des points et personnalisation des lignes
            fig_yearly.update_traces(
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(width=3)
            )
            
            # Ajout des annotations pour les valeurs maximales et minimales
            for gravite in yearly_comparison['gravite_combinee'].unique():
                data_gravite = yearly_comparison[yearly_comparison['gravite_combinee'] == gravite]
                max_point = data_gravite.loc[data_gravite['id_accident'].idxmax()]
                min_point = data_gravite.loc[data_gravite['id_accident'].idxmin()]
                
                # Annotation pour le maximum
                fig_yearly.add_annotation(
                    x=max_point['date_heure'],
                    y=max_point['id_accident'],
                    text=f"Max: {int(max_point['id_accident'])}",
                    showarrow=True,
                    arrowhead=1,
                    yshift=10
                )
                
                # Annotation pour le minimum
                fig_yearly.add_annotation(
                    x=min_point['date_heure'],
                    y=min_point['id_accident'],
                    text=f"Min: {int(min_point['id_accident'])}",
                    showarrow=True,
                    arrowhead=1,
                    yshift=-10
                )
            
            st.plotly_chart(fig_yearly, use_container_width=True)
            
            # Ajout de statistiques détaillées
            with st.expander("Statistiques détaillées par année"):
                stats_annuelles = yearly_comparison.pivot(
                    index='date_heure',
                    columns='gravite_combinee',
                    values='id_accident'
                ).reset_index()
                
                stats_annuelles.columns.name = None
                stats_annuelles = stats_annuelles.rename(columns={'date_heure': 'Année'})
                
                st.dataframe(
                    stats_annuelles.style.format({
                        col: '{:.0f}' for col in stats_annuelles.columns if col != 'Année'
                    }),
                    use_container_width=True
                )

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
        
        # Filtres dans la barre latérale
        st.sidebar.subheader("Filtres")
        
        # Sélection des catégories d'usagers
        categories = sorted(df_periode['type_usager'].unique())
        selected_categories = st.sidebar.multiselect(
            "Types d'usagers",
            options=categories,
            default=categories,
            key='categories_filter_points_noirs'
        )

        # Sélection des niveaux de gravité
        gravity_levels = ['Tué', 'Blessé hospitalisé', 'Blessé léger']
        selected_gravity = st.sidebar.multiselect(
            "Niveaux de gravité",
            options=gravity_levels,
            default=gravity_levels,
            key='gravity_filter_points_noirs'
        )
        
        if not selected_categories or not selected_gravity:
            st.warning("Veuillez sélectionner au moins une catégorie d'usager et un niveau de gravité.")
        else:
            # Filtrage des données
            df_filtered = df_periode[
                (df_periode['type_usager'].isin(selected_categories)) & 
                (df_periode['gravite_combinee'].isin(selected_gravity)) &
                (df_periode['arrondissement'] == arr_analysis)
            ]
            
            # Prétraitement des adresses
            def clean_address(address):
                if pd.isna(address):
                    return address
                # Supprime tout ce qui est avant la virgule (incluse)
                parts = address.split(',', 1)
                if len(parts) > 1:
                    return parts[1].strip()
                return address.strip()
            
            def extract_voie(address):
                if pd.isna(address):
                    return address
                # Supprime les numéros au début de l'adresse
                import re
                # Supprime les numéros et caractères spéciaux au début
                clean = re.sub(r'^[\d\s\-\/\\]+', '', address).strip()
                # Supprime les mentions "bis", "ter", etc.
                clean = re.sub(r'\b(bis|ter|quater)\b', '', clean, flags=re.IGNORECASE).strip()
                return clean

            def normalize_voie(voie):
                if pd.isna(voie):
                    return voie
                # Normalisation basique
                normalized = voie.upper()  # Majuscules
                normalized = re.sub(r'\s+', ' ', normalized)  # Espaces multiples
                normalized = normalized.strip()
                return normalized

            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]

            def find_similar_voies(voies):
                # Normalisation initiale de toutes les voies
                voies_norm = [normalize_voie(v) for v in voies if pd.notna(v)]
                
                # Création des groupes de voies similaires
                voies_groups = {}
                processed = set()
                
                for i, voie1 in enumerate(voies_norm):
                    if voie1 in processed:
                        continue
                        
                    similar_group = [voies[i]]
                    for j, voie2 in enumerate(voies_norm):
                        if i != j and voie2 not in processed:
                            # Calcul de la distance de Levenshtein
                            distance = levenshtein_distance(voie1, voie2)
                            # Si la distance est faible par rapport à la longueur des chaînes
                            max_length = max(len(voie1), len(voie2))
                            if distance <= max_length * 0.2:  # Seuil de 20% de différence
                                similar_group.append(voies[j])
                                processed.add(voie2)
                        
                        # Choisir le nom le plus court comme représentant du groupe
                        canonical_name = min(similar_group, key=len)
                        for voie in similar_group:
                            voies_groups[voie] = canonical_name
                        processed.add(voie1)
                
                return voies_groups

            df_filtered['adresse_clean'] = df_filtered['Adresse'].apply(clean_address)
            df_filtered['voie'] = df_filtered['adresse_clean'].apply(extract_voie)
            
            # Création du mapping des voies similaires
            voies_mapping = find_similar_voies(df_filtered['voie'].unique())
            
            # Application du mapping pour normaliser les noms de voies
            df_filtered['voie_normalisee'] = df_filtered['voie'].map(lambda x: voies_mapping.get(x, x))

            # Analyse des points noirs
            points_noirs = df_filtered.groupby('adresse_clean').agg({
                'id_accident': 'count',
                'gravite_combinee': lambda x: {
                    'Tués': (x == 'Tué').sum(),
                    'Blessés hospitalisés': (x == 'Blessé hospitalisé').sum(),
                    'Blessés légers': (x == 'Blessé léger').sum()
                },
                'type_usager': lambda x: list(x.unique()),
                'latitude': 'first',
                'longitude': 'first'
            }).reset_index()
            
            # Extraction des statistiques de gravité
            points_noirs['Tués'] = points_noirs['gravite_combinee'].apply(lambda x: x['Tués'])
            points_noirs['Blessés hospitalisés'] = points_noirs['gravite_combinee'].apply(lambda x: x['Blessés hospitalisés'])
            points_noirs['Blessés légers'] = points_noirs['gravite_combinee'].apply(lambda x: x['Blessés légers'])
            points_noirs['Types usagers'] = points_noirs['type_usager'].apply(lambda x: ', '.join(x))
            
            # Calcul d'un score de gravité
            points_noirs['Score gravité'] = (
                points_noirs['Tués'] * 10 +
                points_noirs['Blessés hospitalisés'] * 5 +
                points_noirs['Blessés légers'] * 1
            )
            
            # Tri par nombre d'accidents et score de gravité
            points_noirs = points_noirs.sort_values(
                ['id_accident', 'Score gravité'],
                ascending=[False, False]
            )
            
            if not points_noirs.empty:
                # Métriques principales pour l'arrondissement
                st.subheader(f"Points noirs - Arrondissement {arr_analysis}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Point noir principal",
                        points_noirs['adresse_clean'].iloc[0],
                        f"{points_noirs['id_accident'].iloc[0]} accidents"
                    )
                with col2:
                    st.metric(
                        "Accidents mortels",
                        f"{points_noirs['Tués'].iloc[0]}"
                    )

                # Création des sous-onglets pour les différentes vues de la carte
                tab_points, tab_heatmap = st.tabs(["Carte détaillée", "Carte de chaleur"])
                
                with tab_points:
                    st.subheader(f"Carte détaillée des accidents - Arrondissement {arr_analysis}")
                    m_points = folium.Map(
                        location=[df_filtered['latitude'].mean(), df_filtered['longitude'].mean()],
                        zoom_start=15,
                        tiles='cartodbpositron'
                    )
                    
                    # Création d'un cluster de marqueurs
                    marker_cluster = MarkerCluster(
                        options={
                            'maxClusterRadius': 50,
                            'disableClusteringAtZoom': 15
                        }
                    )
                    
                    # Ajout des marqueurs pour chaque accident
                    for _, accident in df_filtered.iterrows():
                        # Couleur selon la gravité
                        color = {
                            'Tué': 'red',
                            'Blessé hospitalisé': 'orange',
                            'Blessé léger': 'yellow'
                        }[accident['gravite_combinee']]
                        
                        # Formatage de la date
                        date_formattee = pd.to_datetime(accident['date']).strftime('%d/%m/%Y à %H:%M')
                        
                        # Création du résumé de l'accident
                        resume_html = f"""
                        <div style='font-family: Arial; padding: 10px; min-width: 200px;'>
                            <h4 style='margin-top: 0; color: #2c3e50;'>Résumé de l'accident</h4>
                            <hr style='margin: 5px 0;'>
                            <p><b>📍 Lieu :</b> {accident['Adresse']}</p>
                            <p><b>📅 Date :</b> {date_formattee}</p>
                            <p><b>👤 Type d'usager :</b> {accident['type_usager']}</p>
                            <p><b>⚠️ Gravité :</b> <span style='color: {color};'>{accident['gravite_combinee']}</span></p>
                            <p><b>🏢 Arrondissement :</b> {accident['arrondissement']}</p>
                        </div>
                        """
                        
                        # Création du marqueur
                        marker = folium.CircleMarker(
                            location=[accident['latitude'], accident['longitude']],
                            radius=8,
                            color=color,
                            fill=True,
                            fillOpacity=0.7,
                            popup=folium.Popup(resume_html, max_width=300)
                        )
                        marker.add_to(marker_cluster)
                    
                    # Ajout du cluster à la carte
                    marker_cluster.add_to(m_points)
                    
                    # Ajout du contrôle des couches
                    folium.LayerControl().add_to(m_points)
                    
                    # Affichage de la carte
                    st.components.v1.html(m_points._repr_html_(), height=600)
                
                with tab_heatmap:
                    st.subheader(f"Carte de chaleur des zones à risque - Arrondissement {arr_analysis}")
                    m_heat = folium.Map(
                        location=[df_filtered['latitude'].mean(), df_filtered['longitude'].mean()],
                        zoom_start=15,
                        tiles='cartodbpositron'
                    )
                    
                    # Création des données pour la heatmap avec pondération par gravité
                    heat_data = []
                    for _, accident in df_filtered.iterrows():
                        weight = {
                            'Tué': 10,
                            'Blessé hospitalisé': 5,
                            'Blessé léger': 1
                        }[accident['gravite_combinee']]
                        heat_data.append([accident['latitude'], accident['longitude'], weight])
                    
                    # Ajout de la heatmap
                    HeatMap(
                        heat_data,
                        name="Zones à risque",
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
                    ).add_to(m_heat)
                    
                    st.components.v1.html(m_heat._repr_html_(), height=400)

                # Tableau détaillé des points noirs de l'arrondissement
                st.subheader(f"Liste des points noirs - Arrondissement {arr_analysis}")
                
                # Préparation du tableau pour l'affichage
                display_columns = [
                    'adresse_clean', 'id_accident', 'Tués',
                    'Blessés hospitalisés', 'Blessés légers',
                    'Score gravité', 'Types usagers'
                ]
                display_df = points_noirs[display_columns].copy()
                display_df.columns = [
                    'Adresse', 'Nombre accidents', 'Tués',
                    'Blessés hospitalisés', 'Blessés légers',
                    'Score gravité', "Types d'usagers"
                ]
                
                # Affichage du tableau avec pagination
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Graphique des points noirs de l'arrondissement
                st.subheader(f"Points noirs - Arrondissement {arr_analysis}")
                
                # Préparation des données pour le graphique empilé
                top_10_locations = points_noirs.head(10)
                df_stacked = pd.DataFrame({
                    'Adresse': top_10_locations['adresse_clean'],
                    'Tués': top_10_locations['Tués'],
                    'Blessés hospitalisés': top_10_locations['Blessés hospitalisés'],
                    'Blessés légers': top_10_locations['Blessés légers']
                })
                
                # Création du graphique empilé
                fig = go.Figure()
                
                # Ajout des barres empilées pour chaque niveau de gravité
                fig.add_trace(go.Bar(
                    name='Tués',
                    x=df_stacked['Adresse'],
                    y=df_stacked['Tués'],
                    marker_color='red'
                ))
                
                fig.add_trace(go.Bar(
                    name='Blessés hospitalisés',
                    x=df_stacked['Adresse'],
                    y=df_stacked['Blessés hospitalisés'],
                    marker_color='orange'
                ))
                
                fig.add_trace(go.Bar(
                    name='Blessés légers',
                    x=df_stacked['Adresse'],
                    y=df_stacked['Blessés légers'],
                    marker_color='yellow'
                ))
                
                # Mise à jour de la mise en page
                fig.update_layout(
                    title=f"Répartition des accidents par gravité - Top 10 des rues",
                    xaxis_title="",
                    yaxis_title="Nombre d'accidents",
                    barmode='stack',
                    height=500,
                    xaxis_tickangle=-45,
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Ajout du graphique d'évolution temporelle
                st.subheader(f"Évolution temporelle des accidents - Arrondissement {arr_analysis}")
                
                # Préparation des données pour l'évolution temporelle
                df_evolution = df_filtered.copy()
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
                    title=f"Évolution mensuelle des accidents par gravité",
                    xaxis_title="Mois",
                    yaxis_title="Nombre d'accidents",
                    height=400,
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
                
                # Ajout de statistiques d'évolution
                with st.expander("Voir les statistiques détaillées d'évolution"):
                    # Calcul des statistiques par année
                    df_evolution['annee'] = df_evolution['date'].dt.year
                    stats_annuelles = df_evolution.groupby(['annee', 'gravite_combinee']).size().unstack(fill_value=0)
                    
                    # Calcul des variations
                    variations = stats_annuelles.pct_change() * 100
                    
                    # Affichage des statistiques
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Nombre d'accidents par année :")
                        st.dataframe(stats_annuelles.style.format("{:.0f}"))
                    
                    with col2:
                        st.write("Variation annuelle (%) :")
                        st.dataframe(variations.style.format("{:.1f}%"))
            
            else:
                st.warning(f"Aucun point noir trouvé dans l'arrondissement {arr_analysis} avec les filtres sélectionnés.")
            
            # Après tous les graphiques d'arrondissement, ajout de l'analyse par voie
            st.markdown("---")
            st.header("Analyse par voie")
            
            # Sélection de la voie avec les noms normalisés
            voies_uniques = sorted(df_filtered['voie_normalisee'].unique())
            selected_voie = st.selectbox(
                "Sélectionner une voie pour suivre son accidentologie",
                options=voies_uniques,
                format_func=lambda x: x if pd.notna(x) else "Non spécifié"
            )
            
            # Filtrage des données pour la voie sélectionnée (utilisant le nom normalisé)
            df_voie = df_filtered[df_filtered['voie_normalisee'] == selected_voie].copy()
            
            if not df_voie.empty:
                st.subheader(f"Analyse de l'accidentologie - {selected_voie}")
                
                # Statistiques générales de la voie
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total accidents",
                        len(df_voie)
                    )
                with col2:
                    accidents_mortels = len(df_voie[df_voie['gravite_combinee'] == 'Tué'])
                    st.metric(
                        "Accidents mortels",
                        accidents_mortels
                    )
                with col3:
                    blesses_graves = len(df_voie[df_voie['gravite_combinee'] == 'Blessé hospitalisé'])
                    st.metric(
                        "Blessés hospitalisés",
                        blesses_graves
                    )
                
                # Préparation des données pour l'évolution temporelle de la voie
                df_voie['mois_annee'] = df_voie['date'].dt.strftime('%Y-%m')
                evolution_voie = df_voie.groupby(['mois_annee', 'gravite_combinee']).size().reset_index(name='count')
                evolution_voie = evolution_voie.sort_values('mois_annee')
                
                # Création du graphique d'évolution pour la voie
                fig_voie = go.Figure()
                
                # Ajout des lignes pour chaque niveau de gravité
                for gravite, color in [('Tué', 'red'), ('Blessé hospitalisé', 'orange'), ('Blessé léger', 'yellow')]:
                    data_gravite = evolution_voie[evolution_voie['gravite_combinee'] == gravite]
                    
                    fig_voie.add_trace(go.Scatter(
                        x=data_gravite['mois_annee'],
                        y=data_gravite['count'],
                        name=gravite,
                        mode='lines+markers',
                        line=dict(width=3, color=color),
                        marker=dict(size=8, color=color)
                    ))
                
                # Mise à jour de la mise en page
                fig_voie.update_layout(
                    title=f"Évolution mensuelle des accidents - {selected_voie}",
                    xaxis_title="Mois",
                    yaxis_title="Nombre d'accidents",
                    height=400,
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
                fig_voie.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                )
                fig_voie.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                )
                
                st.plotly_chart(fig_voie, use_container_width=True)
                
                # Statistiques détaillées de la voie
                with st.expander("Voir les statistiques détaillées de la voie"):
                    # Statistiques par année
                    df_voie['annee'] = df_voie['date'].dt.year
                    stats_voie = df_voie.groupby(['annee', 'gravite_combinee']).size().unstack(fill_value=0)
                    
                    # Types d'usagers impliqués
                    usagers_voie = df_voie['type_usager'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Accidents par année et gravité :")
                        st.dataframe(stats_voie.style.format("{:.0f}"))
                    
                    with col2:
                        st.write("Types d'usagers impliqués :")
                        st.dataframe(
                            pd.DataFrame({
                                "Type d'usager": usagers_voie.index,
                                "Nombre d'accidents": usagers_voie.values
                            })
                        )
            
            else:
                st.warning(f"Aucun point noir trouvé dans l'arrondissement {arr_analysis} avec les filtres sélectionnés.")

    elif analysis_type == "Analyses et Prédictions":
        st.header("Analyses et Prédictions des Accidents")
        
        description = "Le modèle XGBoost utilise les données historiques d'accidents (2017-2022), combinées avec les données météorologiques "
        description += "(température, précipitations, neige, vent) et de trafic (débit et concentration) pour prédire le nombre quotidien "
        description += "d'accidents en 2023. Le modèle apprend les patterns saisonniers et les corrélations entre ces différentes variables "
        description += "pour générer des prédictions précises."
        st.write(description)
        
        # Sélection du modèle
        model_type = st.selectbox(
            "Sélectionner le type de modèle",
            ["XGBoost (Machine Learning)"]
        )
        
        try:
            with st.spinner("Calcul des prédictions en cours..."):
                # Chargement des données météo
                weather_df = load_weather_data()
                if weather_df is None:
                    st.error("Impossible de charger les données météo")
                else:
                    # Préparation des données avec le même type datetime64[ns]
                    weather_df['date'] = pd.to_datetime(weather_df['date'])
                    df['date'] = pd.to_datetime(df['date_heure']).dt.normalize()
                    
                    # Chargement des données de trafic
                    try:
                        traffic_df = pd.read_csv('trafic_routier_paris.csv', sep=';')
                        traffic_df['date'] = pd.to_datetime(traffic_df['date'])
                        # Garder uniquement les colonnes nécessaires
                        traffic_df = traffic_df[['date', 'q', 'k']]  # q: débit, k: concentration
                    except Exception as e:
                        st.error(f"Erreur lors du chargement des données de trafic : {str(e)}")
                        traffic_df = None
                    
                    # Enrichissement et agrégation des données
                    df_enriched = pd.merge(df, weather_df, on='date', how='left')
                    if traffic_df is not None:
                        df_enriched = pd.merge(df_enriched, traffic_df, on='date', how='left')
                    
                    daily_data = df_enriched.groupby('date').agg({
                        'id_accident': 'count',
                        'tavg': 'first',
                        'prcp': 'first',
                        'snow': 'first',
                        'wspd': 'first',
                        'q': 'first',  # débit moyen du trafic
                        'k': 'first'   # concentration moyenne du trafic
                    }).reset_index()
                    daily_data.columns = ['date', 'accidents', 'tavg', 'prcp', 'snow', 'wspd', 'trafic_debit', 'trafic_concentration']
                    
                    # Séparation des données d'entraînement (jusqu'à 2021) et de test (2022-2023)
                    train_data = daily_data[daily_data['date'].dt.year <= 2021].copy()
                    test_data_2022 = daily_data[daily_data['date'].dt.year == 2022].copy()
                    real_data_2023 = daily_data[daily_data['date'].dt.year == 2023].copy()
                    
                    # Création des données de test pour 2023
                    test_dates_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
                    test_data_2023 = pd.DataFrame({'date': test_dates_2023})
                    
                    # Calcul des moyennes mensuelles pour chaque variable
                    monthly_means = {}
                    for col in ['tavg', 'prcp', 'snow', 'wspd', 'trafic_debit', 'trafic_concentration']:
                        if col in train_data.columns:
                            monthly_means[col] = train_data.groupby(train_data['date'].dt.month)[col].mean()
                            test_data_2023[col] = 0.0
                    
                    # Ajout des variables météo et trafic aux données de test 2023
                    for col in ['tavg', 'prcp', 'snow', 'wspd', 'trafic_debit', 'trafic_concentration']:
                        if col in monthly_means:
                            test_data_2023[col] = test_data_2023['date'].dt.month.map(monthly_means[col])
                            if col == 'snow':
                                test_data_2023.loc[test_data_2023['date'].dt.month.isin([6, 7, 8, 9]), 'snow'] = 0
                    
                    # Préparation des données temporelles
                    train_data = prepare_temporal_features(train_data)
                    test_data_2022 = prepare_temporal_features(test_data_2022)
                    test_data_2023 = prepare_temporal_features(test_data_2023)
                    if not real_data_2023.empty:
                        real_data_2023 = prepare_temporal_features(real_data_2023)
                    
                    # Entraînement et prédiction avec XGBoost
                    xgb_model, predictions_2022, predictions_2023, feature_importance = train_xgboost_model(
                        train_data, test_data_2022, test_data_2023
                    )
                    
                    # Calcul des métriques pour 2022
                    mae_2022 = mean_absolute_error(test_data_2022['accidents'], predictions_2022)
                    mse_2022 = mean_squared_error(test_data_2022['accidents'], predictions_2022)
                    rmse_2022 = np.sqrt(mse_2022)
                    r2_2022 = r2_score(test_data_2022['accidents'], predictions_2022)
                    
                    # Affichage des métriques pour 2022
                    st.subheader("Performance du modèle sur l'année 2022")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{mae_2022:.1f}")
                    with col2:
                        st.metric("MSE", f"{mse_2022:.1f}")
                    with col3:
                        st.metric("RMSE", f"{rmse_2022:.1f}")
                    with col4:
                        st.metric("R²", f"{r2_2022:.3f}")
                    
                    # Affichage des features et leur importance
                    st.subheader("Variables utilisées pour la prédiction")
                    
                    # Description des features
                    feature_descriptions = {
                        'tavg': 'Température moyenne (°C)',
                        'prcp': 'Précipitations (mm)',
                        'snow': 'Neige (mm)',
                        'wspd': 'Vitesse du vent (km/h)',
                        'trafic_debit': 'Débit du trafic (véhicules/h)',
                        'trafic_concentration': 'Concentration du trafic (véhicules/km)',
                        'month_sin': 'Saisonnalité mensuelle (sinus)',
                        'month_cos': 'Saisonnalité mensuelle (cosinus)',
                        'day_sin': 'Saisonnalité journalière (sinus)',
                        'day_cos': 'Saisonnalité journalière (cosinus)',
                        'dayofweek_sin': 'Jour de la semaine (sinus)',
                        'dayofweek_cos': 'Jour de la semaine (cosinus)'
                    }
                    
                    # Création d'un DataFrame avec les descriptions
                    feature_importance_df = feature_importance.copy()
                    feature_importance_df['description'] = feature_importance_df['feature'].map(feature_descriptions)
                    feature_importance_df['importance_percent'] = feature_importance_df['importance'] * 100
                    
                    # Affichage dans un tableau
                    st.write("Liste des variables d'entrée et leur importance relative :")
                    
                    # Formatage du tableau
                    formatted_df = pd.DataFrame({
                        'Variable': feature_importance_df['description'],
                        'Importance (%)': feature_importance_df['importance_percent'].round(2)
                    }).reset_index(drop=True)
                    
                    st.dataframe(formatted_df)
                    
                    # Création d'un graphique à barres pour l'importance des features
                    fig_importance = go.Figure()
                    
                    fig_importance.add_trace(go.Bar(
                        x=feature_importance_df['importance_percent'],
                        y=feature_importance_df['description'],
                        orientation='h'
                    ))
                    
                    fig_importance.update_layout(
                        title="Importance relative des variables dans le modèle",
                        xaxis_title="Importance (%)",
                        yaxis_title="Variable",
                        height=400,
                        margin=dict(t=30, r=10, b=0, l=10)
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Création du graphique des prédictions
                    fig = go.Figure()
                    
                    # Données historiques d'entraînement (2017-2021)
                    fig.add_trace(go.Scatter(
                        x=train_data['date'],
                        y=train_data['accidents'],
                        name='Données historiques (2017-2021)',
                        line=dict(color='blue')
                    ))
                    
                    # Données réelles et prédictions 2022
                    fig.add_trace(go.Scatter(
                        x=test_data_2022['date'],
                        y=test_data_2022['accidents'],
                        name='Données réelles 2022',
                        line=dict(color='green')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=test_data_2022['date'],
                        y=predictions_2022,
                        name='Prédictions 2022',
                        line=dict(color='orange', dash='dash')
                    ))
                    
                    # Données réelles 2023
                    if not real_data_2023.empty:
                        fig.add_trace(go.Scatter(
                            x=real_data_2023['date'],
                            y=real_data_2023['accidents'],
                            name='Données réelles 2023',
                            line=dict(color='green')
                        ))
                    
                    # Prédictions XGBoost pour 2023
                    fig.add_trace(go.Scatter(
                        x=test_data_2023['date'],
                        y=predictions_2023,
                        name='Prédictions 2023',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Mise en forme du graphique
                    fig.update_layout(
                        title="Évolution du nombre d'accidents quotidiens et prédictions (XGBoost)",
                        xaxis_title="Date",
                        yaxis_title="Nombre d'accidents",
                        height=600,
                        showlegend=True,
                        shapes=[
                            # Ligne verticale pour marquer le début de 2022
                            dict(
                                type='line',
                                x0='2022-01-01',
                                x1='2022-01-01',
                                y0=0,
                                y1=1,
                                yref='paper',
                                line=dict(
                                    color='orange',
                                    dash='dash'
                                )
                            ),
                            # Ligne verticale pour marquer le début de 2023
                            dict(
                                type='line',
                                x0='2023-01-01',
                                x1='2023-01-01',
                                y0=0,
                                y1=1,
                                yref='paper',
                                line=dict(
                                    color='red',
                                    dash='dash'
                                )
                            )
                        ],
                        annotations=[
                            # Annotation pour la ligne de 2022
                            dict(
                                x='2022-01-01',
                                y=1.02,
                                yref='paper',
                                showarrow=False,
                                text='Début 2022',
                                font=dict(color='orange')
                            ),
                            # Annotation pour la ligne de 2023
                            dict(
                                x='2023-01-01',
                                y=1.02,
                                yref='paper',
                                showarrow=False,
                                text='Début 2023',
                                font=dict(color='red')
                            )
                        ],
                        margin=dict(t=50, r=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")
            st.text("Traceback complet :")
            st.code(traceback.format_exc())
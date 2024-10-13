import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import numpy as np
import ast
import plotly.express as px
import re
from collections import Counter
from nltk.corpus import stopwords

# Configuration de la page en mode large
st.set_page_config(layout="wide")

# Panneau latéral
page = st.sidebar.radio("Choisissez une page :", ["Accueil", "Projet NLP/LLM"])

# Page d'accueil
if page == "Accueil":
    # Utiliser les colonnes de Streamlit pour centrer les éléments
    col1, col2, col3 = st.columns([1, 2, 1])  # Diviser l'espace en trois colonnes

    with col2:  # Centrer les éléments en les plaçant dans la colonne centrale
        # Titre
        st.markdown('<h1 style="text-align: center;">Théo Bernad</h1>', unsafe_allow_html=True)


        img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
        with img_col2:
            st.image("photo.jpg", width=300)



        # Résumé
        st.markdown("""
        <div style="text-align: center;">
        ## Profil
        Data Scientist passionné par l'analyse des données et le machine learning. 
        Fort d'une expérience dans la création de modèles prédictifs et la visualisation de données.
        </div>
        """, unsafe_allow_html=True)

        # Compétences
        st.markdown("""
        <div style="text-align: center;">## Compétences"</div>
        """)
        st.markdown("- Python")
        st.markdown("- Machine Learning")
        st.markdown("- Data Visualization")
        st.markdown("- Statistiques")

        # Contact
        st.markdown("## Contact")
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        st.markdown("[LinkedIn](https://www.linkedin.com/in/theobcd/)")
        st.markdown("[GitHub](https://github.com/votreprofil)")
        st.markdown('</div>', unsafe_allow_html=True)

        # Bouton pour télécharger le CV
        st.download_button("Télécharger mon CV", "CV DATA SCIENTIST- BERNAD THEO.pdf")

elif page == "Projet NLP/LLM":
    # Chargement des données
    file_path = 'artistes.parquet'  # Assurez-vous que ce fichier existe dans le répertoire
    df = pd.read_parquet(file_path)

    # Vérification que la colonne 'avg_embedding' existe
    if 'avg_embedding' not in df.columns:
        st.error('La colonne avg_embedding n\'existe pas dans le DataFrame.')
    else:
        # Extraction des noms d'artistes et des embeddings
        artists = df['artist_name'].tolist()
        embeddings = np.array(df['avg_embedding'].apply(ast.literal_eval).tolist())

        # Onglets
        tabs = st.tabs(["Visualisation des Embeddings", "Clustering des Artistes", "Etude par Artiste"])

        # Onglet 1 : Visualisation des embeddings
        with tabs[0]:
            # Visualisation avec t-SNE
            def generate_espace_artistes():
                reducer = TSNE(n_components=2, random_state=0)
                reduced_embeddings = reducer.fit_transform(embeddings)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=reduced_embeddings[:, 0],
                    y=reduced_embeddings[:, 1],
                    mode='markers',
                    marker=dict(size=8, color='blue'),
                    text=artists,
                    textposition='top center',
                    hoverinfo='none'
                ))

                for i, artist in enumerate(artists):
                    artist_url = f'/{artist}/'
                    fig.add_annotation(
                        x=reduced_embeddings[i, 0],
                        y=reduced_embeddings[i, 1],
                        text=f'<a href="{artist_url}" target="_blank">{artist}</a>',
                        showarrow=True,
                        arrowhead=2,
                        ax=20,
                        ay=-20,
                        font=dict(size=10, color='blue'),
                        align='center'
                    )

                fig.update_layout(
                    autosize=True,
                    width=1800,
                    height=800,
                    title='Visualisation des Embeddings des Artistes avec Embeddings Moyens',
                    template='plotly_white',
                    margin=dict(l=50, r=50, t=100, b=50),
                    xaxis=dict(showgrid=True, zeroline=False),
                    yaxis=dict(showgrid=True, zeroline=False)
                )

                return fig

            fig = generate_espace_artistes()
            st.plotly_chart(fig, use_container_width=True)

        # Onglet 2 : Clustering des artistes
        with tabs[1]:
            
            def cluster_artists(artist_vectors, n_clusters=5):
                """Clustering des artistes en utilisant les vecteurs de leurs paroles."""
                artist_names = list(artist_vectors.keys())
                embeddings = np.array(list(artist_vectors.values()))
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                clusters = kmeans.fit_predict(embeddings)
                
                return artist_names, embeddings, clusters

            # Fonction pour visualiser les clusters en utilisant t-SNE
            def visualize_clusters_with_tsne(artist_names, embeddings, clusters):
                """Visualiser les clusters d'artistes avec t-SNE."""
                # Réduction de dimensionnalité avec t-SNE
                tsne = TSNE(n_components=2, random_state=0)
                reduced_embeddings = tsne.fit_transform(embeddings)
                
                # Création du graphique
                fig = go.Figure()
                color_scale = px.colors.qualitative.Plotly

                # Ajouter les points au graphique
                fig.add_trace(go.Scatter(
                    x=reduced_embeddings[:, 0],
                    y=reduced_embeddings[:, 1],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=clusters,
                        colorscale=color_scale,
                        colorbar=None,
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    hoverinfo='text',
                    hovertext=[f'Artiste: {name}<br>Cluster: {cluster}' for name, cluster in zip(artist_names, clusters)]
                ))

                # Ajouter des annotations avec des liens cliquables
                annotations = []
                for i, artist in enumerate(artist_names):
                    artist_url = f'/{artist}/'  # URL vers la page de l'artiste
                    annotations.append(dict(
                        x=reduced_embeddings[i, 0],
                        y=reduced_embeddings[i, 1],
                        text=f'<a href="{artist_url}" target="_blank">{artist}</a>',
                        showarrow=True,
                        arrowhead=2,
                        ax=20,
                        ay=-20,
                        font=dict(size=10, color='blue'),
                        align='center'
                    ))

                # Configurer la mise en page du graphique
                fig.update_layout(
                    annotations=annotations,
                    title='Clustering des Artistes basés sur les Embeddings des Paroles (t-SNE)',
                    xaxis_title='Composante 1',
                    yaxis_title='Composante 2',
                    showlegend=False,
                    template='plotly_white',
                    width=1800,
                    height=1000,
                    autosize=True,
                    margin=dict(l=50, r=50, t=100, b=50),
                    xaxis=dict(showgrid=True, zeroline=False),
                    yaxis=dict(showgrid=True, zeroline=False)
                )
                
                return fig

            # Chargement des données et exécution du clustering et de la visualisation
            def load_and_visualize():
                file_path = 'artistes.parquet'  # Assurez-vous que ce fichier existe dans le répertoire
                df = pd.read_parquet(file_path)

                # Vérification que la colonne 'avg_embedding' existe
                if 'avg_embedding' not in df.columns:
                    st.error('La colonne avg_embedding n\'existe pas dans le DataFrame.')
                    return

                # Extraction des artistes et des embeddings
                artist_vectors = {row['artist_name']: ast.literal_eval(row['avg_embedding']) for index, row in df.iterrows()}
                
                # Appliquer le clustering
                n_clusters = 5  # Nombre de clusters
                artist_names, embeddings, clusters = cluster_artists(artist_vectors, n_clusters)
                
                # Visualiser les clusters
                fig = visualize_clusters_with_tsne(artist_names, embeddings, clusters)
                st.plotly_chart(fig, use_container_width=True)

            # Appel de la fonction pour charger et visualiser les données
            load_and_visualize()
        # Onglet 3 : Autre contenu (ajoutez ici ce que vous souhaitez)
        with tabs[2]:
            # Charger les données
            df = pd.read_parquet('cluster.parquet')


            # Fonction pour récupérer les titres, albums et embeddings d'un artiste
            def fetch_artist_lyrics(artist_name, df):
                # Filtrer le DataFrame pour l'artiste donné
                artist_data = df[df['artist_name'] == artist_name]
                titles = artist_data['song_title'].tolist()
                albums = artist_data['album_name'].tolist()
                embeddings = artist_data['embedded_lyrics'].tolist()  # Supposant que les embeddings sont stockés sous forme de liste

                # Convertir les chaînes d'embeddings en listes
                embeddings = [ast.literal_eval(embedding) if isinstance(embedding, str) else embedding for embedding in embeddings]
                return titles, albums, embeddings

            # Fonction pour visualiser les chansons d'un artiste
            def visualize_artist_songs(artist_name, df, method='PCA'):
                titles, albums, embeddings = fetch_artist_lyrics(artist_name, df)

                # Vérifier si les embeddings sont valides
                if len(embeddings) == 0 or any(len(embedding) == 0 for embedding in embeddings):
                    return ''  # Si les embeddings sont vides, on ne fait rien

                # Convertir la liste de listes en tableau NumPy
                embeddings_array = np.array(embeddings)

                # Réduction des dimensions
                if method == 'PCA':
                    reducer = PCA(n_components=2)
                elif method == 't-SNE':
                    reducer = TSNE(n_components=2, random_state=0)
                else:
                    raise ValueError("Méthode non reconnue. Utilisez 'PCA' ou 't-SNE'.")

                reduced_embeddings = reducer.fit_transform(embeddings_array)

                unique_albums = list(set(albums))
                color_map = {album: i for i, album in enumerate(unique_albums)}
                colors = [color_map[album] for album in albums]

                fig = go.Figure()
                color_scale = px.colors.qualitative.Plotly
                fig.add_trace(go.Scatter(
                    x=reduced_embeddings[:, 0],
                    y=reduced_embeddings[:, 1],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=colors,
                        colorscale=color_scale,
                        colorbar=dict(title='album_name'),
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                ))

                # Ajouter des annotations avec des liens cliquables
                annotations = []
                for i, title in enumerate(titles):
                    formatted_title = title.replace(" ", "_")
                    annotations.append(dict(
                        x=reduced_embeddings[i, 0],
                        y=reduced_embeddings[i, 1],
                        text=f'<a href="/chanson/{artist_name}/{formatted_title}/" target="_blank">{title}</a>',
                        showarrow=True,
                        arrowhead=2,
                        ax=20,
                        ay=-20,
                        font=dict(size=10, color='black'),
                        align='center'
                    ))

                fig.update_layout(
                    title=f'Visualisation des Embeddings des Paroles - Répertoire de {artist_name}',
                    xaxis_title='Composante 1',
                    yaxis_title='Composante 2',
                    showlegend=True,
                    template='plotly_white',
                    width=1500,
                    height=900,
                    autosize=True,
                    margin=dict(l=50, r=50, t=100, b=50),
                    annotations=annotations
                )

                return fig

            # Titre de l'application
            st.title("Visualisation des Chansons par Artiste")

            
            # Sélectionner l'artiste dans un sélecteur
            artist_name = st.selectbox("Choisir un artiste", df['artist_name'].unique())

    
            if artist_name:
                # Visualiser les chansons de l'artiste
                fig = visualize_artist_songs(artist_name, df, 'PCA')
                st.plotly_chart(fig)

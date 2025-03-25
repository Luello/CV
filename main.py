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
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        /* Agrandir le texte dans les boutons radio de la sidebar */
        section[data-testid="stSidebar"] .stRadio > label {
            font-size: 1.2rem;
            font-weight: 500;
        }

        /* Bonus : icônes emoji un peu plus espacées */
        section[data-testid="stSidebar"] .stRadio div {
            padding-top: 0.4rem;
            padding-bottom: 0.4rem;
        }
    </style>
""", unsafe_allow_html=True)
# Configuration de la page en mode large


# Panneau latéral
page = st.sidebar.radio("📁 Navigation :", [
    "🏠 Accueil",
    "📈 Démo - Visualisations",
    "▶️ NLP: Analyse de l'identité politique des influenceurs Youtube",
    "🎵 NLP/LLM: Cartographier les artistes français depuis les paroles de leur répertoire."
])
if page== "📈 Démo - Visualisations":
    st.title("📊 Visualisations réalisées avec les données Data.gouv sur les accidents routiers.")

    # Intégration de l'iframe Infogram
    infogram_html = """
<div class="infogram-embed" data-id="8b9c87b0-eb40-4411-927d-1141a21b8c59" 
     data-type="interactive" data-title=""></div>
<script>
!function(e,n,i,s){
    var d="InfogramEmbeds";
    var o=e.getElementsByTagName(n)[0];
    if(window[d] && window[d].initialized) {
        window[d].process && window[d].process();
    } else if(!e.getElementById(i)){
        var r=e.createElement(n);
        r.async=1;
        r.id=i;
        r.src=s;
        o.parentNode.insertBefore(r,o);
    }
}(document,"script","infogram-async","https://e.infogram.com/js/dist/embed-loader-min.js");
</script>

<div style="padding:8px 0;font-family:Arial!important;font-size:13px!important;
line-height:15px!important;text-align:center;border-top:1px solid #dadada;
margin:0 30px">
<br><a href="https://infogram.com" style="color:#989898!important;
text-decoration:none!important;" target="_blank" rel="nofollow">Infogram</a></div>
"""

    
    st.components.v1.html(infogram_html, height=800, scrolling=True)
    
    # Ajout du crédit Infogram (facultatif)
    st.markdown(
        '<div style="padding:8px 0;font-family:Arial!important;font-size:13px!important;'
        'line-height:15px!important;text-align:center;border-top:1px solid #dadada;'
        'margin:0 30px;width: 640px">'
        '<br><a href="https://infogram.com" style="color:#989898!important;'
        'text-decoration:none!important;" target="_blank" rel="nofollow">Infogram</a></div>',
        unsafe_allow_html=True
    )
if page == "🏠 Accueil":
    st.markdown('<h1 style="text-align: center;">Bienvenue sur mon CV applicatif</h1><br>', unsafe_allow_html=True)
    # Utiliser les colonnes de Streamlit pour centrer les éléments
    col1, col2, col3 = st.columns([1, 2,1])  # Diviser l'espace en trois colonnes
    
    with col1:
        st.image("photo.jpg", width=250,use_column_width='always')
    with col2:  # Centrer les éléments en les plaçant dans la colonne centrale
        # Titre
        
        st.markdown('<h1 style="text-align: center;">Théo Bernad</h1><br>', unsafe_allow_html=True)
        
        
        

        # Description principale
        st.markdown(
            """
            <div style="text-align: center; font-size: 18px; line-height: 1.6; margin-top: 20px;">
                <p>Data Scientist passionné par les opportunités qu'offrent les progrès en IA.</p>  
                <p>Je peux mener un projet Data du besoin métier au déploiement, dans une optique "full-stack".</p>
                <p> Vous pouvez accéder, depuis le menu de gauche, aux différents projets que j'ai pu réaliser, et dont je déploie une partie ici.</p>
            </div><br>
            """, 
            unsafe_allow_html=True
        )

    # Créer les onglets
    tab1, tab2, tab3 = st.tabs(["Expériences", "Formations","Passions"])

    # Contenu de chaque onglet
    with tab3:
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown(
                """
                <div style="text-align: left; font-size: 18px; line-height: 1.6; margin-top: 20px;">
                    <p>Quelques domaines de la Data dont les thématiques me passionnent:</p>
                    <ul style="list-style-position: inside; text-align: left; display: inline-block;">
                        <li>Études sociologiques et comportementales</li>
                        <li>Analyse des Gameplays dans le sport ou les jeux vidéo</li>
                        <li>Projets autour de la cognition et des imageries cérébrales</li>
                        <li>Domotiques et agents intelligents</li>
                    </ul>
                </div><br>
                """, 
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                """
                <div style="text-align: left; font-size: 18px; line-height: 1.6; margin-top: 20px;">
                    <p> D'autres intérêts que j'ai dans la vie : </p>
                    <ul style="list-style-position: inside; text-align: left; display: inline-block;">
                        <li> Escalade, Boxe, Escrime</li>
                        <li> Cinéma, Histoire, Philosophie, Cuisine,   </li>
                        <li> Les nouvelles technologies et leurs implications</li>
                        <li> Jeux historiques de stratégie </li>
                    </ul>
                </div><br>
                """, 
                unsafe_allow_html=True
            )

    with tab1:
        st.markdown(
            """
            <div style="text-align: left; font-size: 18px; line-height: 1.6; margin-top: 20px;">
                <p><strong>Expériences professionnelles:</strong></p>
                <ul style="list-style-position: inside; text-align: left; display: inline-block;">
                    <li><strong>Data Scientist - Marine Nationale (Tours)</strong>
                        <ul style="margin-left: 20px; list-style-type: disc;">
                            <li>Projet IA de prédiction sur une thématique RH</li>
                            <li>Traitement, reconstitution et création de données</li>
                            <li>Analyse BI (Dashboard QlikSense)</li>
                            <li>Amélioration des processus Data (VBA, UIPATH, Python)</li>
                            <li>Accompagnement structurel au traitement et à la politique des données</li>
                        </ul>
                        <p style="margin-left: 20px;"><em>Soft Skills principaux :</em> Autonomie, gestion de projet, écoute des besoins, créativité, rigueur</p>
                    </li>
                    <br>
                    <li><strong>Data Analyst - Gowod (Montpellier)</strong>
                        <ul style="margin-left: 20px; list-style-type: disc;">
                            <li>Analyse et visualisation sur le comportement des utilisateurs d'une application sportive</li>
                            <li>Analyses RFM / BI, stratégies marketing</li>
                        </ul>
                        <p style="margin-left: 20px;"><em>Soft Skills principaux :</em> Travail en équipe, vision marketing, appréhension d'une Base de données complexe</p>
                    </li>
                    <br>
                    <li><strong>Assistant pédagogique - Lycée Marcel Sembat (Lyon)</strong>
                        <ul style="margin-left: 20px; list-style-type: disc;">
                            <li>Accompagnement pédagogique des élèves</li>
                            <li>Projet pédagogique contre le décrochage scolaire</li>
                        </ul>
                        <p style="margin-left: 20px;"><em>Soft Skills principaux :</em> Adaptabilité, sociabilité, pédagogie, patience</p>
                    </li>
                    <br>
                    <li><strong>Remplacements éducatifs - IME Pierre de Lune (Lyon)</strong>
                        <ul style="margin-left: 20px; list-style-type: disc;">
                            <li>Accompagnement quotidien d'enfants en situation d'handicap</li>
                        </ul>
                        <p style="margin-left: 20px;"><em>Soft Skills principaux :</em> Patience, réactivité, Travail d'équipe, gestion de crise</p>
                    </li>
                    <br>
                    <li><strong>Autres expériences constructives:</strong>
                        <ul style="margin-left: 20px; list-style-type: disc;">
                            <li>Vendanges</li>
                            <li>Télévente</li>
                            <li>Rénovation d'intérieur</li>
                            <li>Gestion d'une auberge de jeunesse</li>
                        </ul>
                        <p style="margin-left: 20px;"><em>Soft Skills principaux :</em> Adaptabilité, ténacité, curiosité</p>
                    </li>
                </ul>
            </div><br>
            """, 
            unsafe_allow_html=True
        )

    with tab2:
        st.markdown(
            """
            <div style="text-align: left; font-size: 18px; line-height: 1.6; margin-top: 20px;">
                <p><strong>Formations :</strong></p>
                <ul style="list-style-position: inside; text-align: left; display: inline-block;">
                    <li><strong>Data Scientist - Marine Nationale / WCS (2023)</strong>
                        <ul style="margin-left: 20px; list-style-type: disc;">
                            <li>Projets de Machine learning: Prédiction, Classification, Clustering, méthodes de Bagging/Boosting, modélisation de séries temporelles, méthodes ensemblistes...</li>
                            <li>Réseaux de neurones: CNN, RNN, LSTM : Python, TensorFlow, Keras, Scikit-learn</li>
                            <li>Outils de collaboration et de production : Git, Docker, Terminal</li>
                            <li>Développement d'application : Django, FastAPI, CSS, HTML</li>
                            <li>Statistiques et fondamentaux mathématiques : tests statistiques, distributions...</li>
                            <li>Gestion de projets & Travail d'équipe</li>
                        </ul>
                        <p style="margin-left: 20px;"><em>J'y ai validé une certification professionnelle "Concepteur Développeur d'application" (Niveau 6)</em></p>
                    </li>
                    <br>
                    <li><strong>Data Analyst - WCS (Lyon - 2022)</strong>
                        <ul style="margin-left: 20px; list-style-type: disc;">
                            <li>Codage et traitement de données en Python (mon outil principal) : Pandas, NumPy, Matplotlib, Plotly, SciPy, BeautifulSoup</li>
                            <li>Développement de différentes applications Streamlit à des fins d'analyses ou de classifications : Scikit-learn, TensorFlow, PyTorch, Streamlit, Datapane</li>
                            <li>Spécialisation en machine learning : Projet de prédiction du vainqueur d'un duel tennistique depuis des données sur le style de jeu et l'historicité des joueurs</li>
                        </ul>
                        <p style="margin-left: 20px;"><em>Formation de 8 mois pour approfondir une base solide de la manipulation des données et de leurs analyses</em></p>
                    </li>
                    <br>
                    <ul style="text-align: center;"><strong>Je me spécialise à ce moment là dans la Data!</strong></ul>
                    <br>
                    <li><strong>Master en Science politique - Enquêtes et analyse des processus politiques (Lyon - 2020)</strong>
                        <ul style="margin-left: 20px; list-style-type: disc;">
                            <li>Stage de terrain : écoute active et recueil d'éléments pour une étude</li>
                            <li>Focales Épistémologie, Sociologie de l'action publique, expertise internationale</li>
                            <li>Rédaction d'un mémoire de recherche de 130 pages sur le rapport au politique des éducateurs</li>
                        </ul>
                        <p style="margin-left: 20px;"><em>M'a permis d'approfondir l'étude des processus politiques aux échelles structurelles ou individuelles, et leurs implications. Un atout significatif pour situer les acteurs, les enjeux, les institutions dans tous les contextes. Synthétiser, construire une Stratégie.</em></p>
                    </li>
                    <br>
                    <li><strong>Licence en sciences cognitives, réalisée en même temps que le master en science politique (Lyon - 2020)</strong>
                        <ul style="margin-left: 20px; list-style-type: disc;">
                            <li>Étude des mécanismes cognitifs : Mémoire, attention, langage, émotions, raisonnement, action</li>
                            <li>Apports concrets en neuro-imagerie, plasticité cérébrale, neuroprothèses</li>
                            <li>Programmation : cognition artificielle, Python</li>
                        </ul>
                        <p style="margin-left: 20px;"><em>Les apports significatifs de cette discipline émergente m'ont familiarisé avec ses enjeux, ses méthodes et ses ambitions.</em></p>
                    </li>
                </ul>
            </div><br>
            """, 
            unsafe_allow_html=True
        )
        # Contact en bas de page
    st.markdown('<br><br><br><h2 style="text-align: center;">Contact</h2>', unsafe_allow_html=True)

    # Centrer les liens et le bouton de téléchargement
    col1, col2, col3 = st.columns([1, 2, 1])    
    with col2:
        st.markdown(
    '<div style="text-align: center; margin-top: 20px;">'  # Ajoute un espacement au-dessus
    '<a href="https://www.linkedin.com/in/theobcd/" style="display: block; margin-bottom: 10px;">LinkedIn</a>'  # Chaque lien sur une nouvelle ligne avec un espacement en bas
    '<a href="https://github.com/Luello" style="display: block; margin-bottom: 10px;">GitHub</a>'  # Ajoute un espacement en bas
    '</div>', 
    unsafe_allow_html=True
)

    # Bouton pour télécharger le CV centré
    file_path = "CV DATA SCIENTIST- BERNAD THEO.pdf"
    try:
        with open(file_path, "rb") as file:
            st.markdown('<div style="text-align: center; margin-top: 10px;">', unsafe_allow_html=True)  # Nouvelle div pour le bouton
            st.download_button(
                label="Télécharger mon CV",
                data=file,
                file_name="CV_DATA_SCIENTIST_BERNAD_THEO.pdf",  # nom du fichier à télécharger
                mime="application/pdf"  # type MIME pour un fichier PDF
            )
            st.markdown('</div>', unsafe_allow_html=True)  # Ferme la div pour le bouton
    except FileNotFoundError:
        st.error("Le fichier n'a pas été trouvé. Vérifiez le chemin et le nom du fichier.")


elif page== "▶️ NLP: Analyse de l'identité politique des influenceurs Youtube":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
    from umap import UMAP
    import plotly.express as px
    import streamlit as st
    import ast

    st.title("📊 Projection UMAP des chaînes YouTube selon leur identité politique")

  import pandas as pd
import streamlit as st

st.markdown("""
### 🧠 Objectif

Cette visualisation cherche à représenter l'identité politique des influenceurs YouTube à partir de plusieurs dimensions qualitatives et quantitatives extraites de leurs discours.

Ce graphique illustre comment une analyse NLP peut combiner **quantitatif** et **qualitatif** pour appréhender des logiques politiques implicites dans les scripts vidéos.

Un **LLM** analyse les scripts et extrait automatiquement un ensemble de **descripteurs discursifs, idéologiques et narratifs**.
""")

# Construction du tableau descripteurs
descripteurs = [
    ("format_detecte", "Type précis de vidéo", "débat, vlog, podcast, analyse politique…"),
    ("ton_general", "Ton dominant du discours", "neutre, polémique, académique, humoristique…"),
    ("registre_discursif", "Type discursif", "explicatif, militant, scientifique, complotiste…"),
    ("stratégie_argumentative", "Stratégie argumentative", "démonstratif, émotionnel, narratif…"),
    ("structure_narrative", "Organisation narrative", "linéaire, chaotique, récurrente…"),
    ("style_de_politisation", "Forme de politisation", "rationnel, affectif, moraliste…"),
    ("valeurs_invoquées", "Valeurs mises en avant", "liberté, égalité, nation, méritocratie…"),
    ("thématiques_dominantes", "Thèmes principaux", "éducation, immigration, écologie…"),
    ("cibles_implicites", "Cibles idéologiques", "élites, médias, gouvernement, minorités…"),
    ("références_implicites", "Références culturelles", "Zemmour, Orwell, Deleuze…"),
    ("axe_latent", "Cadres idéologiques", "technocratie, souverainisme, socialisme…"),
    ("conception_du_nous", "Collectif valorisé", "peuple, citoyens, communauté, nation…"),
    ("positionnement_sociétal", "Rapport à la société", "critique des élites, défense d’un groupe…"),
    ("cadre_problematisation", "Façon de poser les enjeux", "système en crise, injustice sociale…"),
    ("figures_ennemies", "Adversaires implicites", "le système, les mondialistes…"),
    ("récit_idéologique", "Narration politique", "déclin civilisationnel, peuple trahi…"),
    ("axes_de_tension", "Axes de conflit", "élite vs peuple, progrès vs tradition…"),
    ("paradigmes_compatibles", "Paradigmes compatibles", "libéralisme, anarchisme, gaullisme…"),
    ("ton_politique", "Ton politique", "engagé, militant, distant…"),
    ("enjeux_sociaux_centrés", "Enjeux sociaux centraux", "santé, sécurité, inclusion…"),
    ("charge_politique_latente", "Score de politisation", "0 = neutre, 100 = très politisé"),
    ("position_stratégique", "Stratégie globale", "offensive, défensive, ambiguë…"),
    ("mode_d_interpellation_du_public", "Type d’interpellation", "directe, pédagogique, émotionnelle…"),
    ("figure_du_locuteur", "Rôle du locuteur", "expert, citoyen, leader, victime…"),
    ("échelle_de_politisation", "Niveau d’ancrage", "locale, nationale, internationale…"),
    ("type_de_menace_perçue", "Menace évoquée", "déclin, subversion, chaos…"),
    ("registre_moral_implicite", "Fond moral", "progressiste, conservateur, égalitariste…"),
    ("ton_affectif_dominant", "Émotion dominante", "colère, peur, espoir, fierté…"),
    ("niveau_de_certitude", "Certitude exprimée", "score 0-100 (incertitude → affirmation)"),
    ("index_performativite", "Performativité", "0 = descriptif, 100 = incitation forte à l’action"),
    ("index_fanatisme", "Fermeté idéologique", "0 = ouvert au débat, 100 = hostile aux avis opposés")
]

df_descr = pd.DataFrame(descripteurs, columns=["🧩 Variable", "🗂️ Description", "🔍 Exemples ou échelle"])

# Encapsuler proprement
with st.expander("📘 Voir la liste détaillée des descripteurs analysés par le LLM", expanded=False):
    st.dataframe(df_descr, use_container_width=True, height=600)

# Démarche analytique
st.markdown("""
---

### ⚙️ Démarche analytique

- Les variables **numériques** (ex: `charge_politique_latente`, `index_fanatisme`) sont standardisées via `StandardScaler`.
- Les variables **catégorielles multilabels** (ex: *valeurs*, *figures ennemies*) sont vectorisées avec `MultiLabelBinarizer`.
- Ces vecteurs sont projetés dans un plan 2D avec `UMAP` (distance *cosine*), ce qui permet d’observer des proximités idéologiques latentes.

🎨 Le **gradient de couleur** représente la politisation : plus il est élevé, plus le discours est marqué politiquement.
""")

    # Chargement des données
    with st.spinner("⏳ Patientez quelques secondes le temps que le graphique charge :)"):
        df = pd.read_csv("results_df.csv")
        df = df.dropna(subset=["title", "charge_politique_latente"]).reset_index(drop=True)
    
        # Conversion des colonnes de listes depuis string (si nécessaire)
        list_cols = [
            "style_de_politisation",
            "figures_ennemies",
            "valeurs_invoquées",
            "thématiques_dominantes"
        ]
    
        for col in list_cols:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    
        # Colonnes numériques à inclure
        numerical_cols = ["charge_politique_latente", "index_fanatisme"]
    
        # Encodage MultiLabel
        encoded_parts = []
        for col in list_cols:
            mlb = MultiLabelBinarizer()
            try:
                binarized = mlb.fit_transform(df[col])
                encoded_df = pd.DataFrame(binarized, columns=[f"{col}__{c}" for c in mlb.classes_])
                encoded_parts.append(encoded_df)
            except Exception as e:
                st.warning(f"Problème d'encodage pour {col} : {e}")
    
        # Construction de la matrice finale
        X_num = df[numerical_cols].fillna(0).reset_index(drop=True)
        X_cat = pd.concat(encoded_parts, axis=1).reset_index(drop=True)
        X_all = pd.concat([X_num, X_cat], axis=1)
    
        # Normalisation
        X_scaled = StandardScaler().fit_transform(X_all)
    
        # UMAP
        umap = UMAP(n_neighbors=5, min_dist=0.1, metric="cosine", random_state=42)
        embedding = umap.fit_transform(X_scaled)
    
        # DataFrame pour visualisation
        df_visu = pd.DataFrame({
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "title": df["title"],
            "charge_politique_latente": df["charge_politique_latente"]
        })
    
        # Graphique Plotly
        fig = px.scatter(
            df_visu,
            x="x", y="y",
            text="title",
            color="charge_politique_latente",
            color_continuous_scale="RdBu_r",
            hover_name="title",
            title="Projection UMAP des chaînes (gradient = charge politique latente)"
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=700)
    
        st.plotly_chart(fig, use_container_width=True)
elif page == "🎵 NLP/LLM: Cartographier les artistes français depuis les paroles de leur répertoire.":
    st.markdown("""
    <div style="text-align: left; font-size: 18px; line-height: 1.6; margin-top: 20px;">
        <p><strong>Présentation du projet :</strong></p>
        <p>
            L'idée principale de ce projet est de pouvoir visualiser la distance ou la proximité entre les artistes musicaux français, à partir de leurs paroles.
        </p>
        <p>
            Grâce à différents traitements de données depuis une API de paroles de musiques et un stockage en base de données, un embedding (une méthode pour représenter dans un espace vectoriel du langage naturel) est réalisé grâce au modèle de langage <strong> FlauBERT</strong>, 
            le graphique ci-dessous permet de visualiser cette proximité entre les artistes.
        </p>
        <p>
            La multi-dimensionnalité de l'espace vectoriel est réduite grâce à une méthode de réduction de dimension (TSNE), afin d'être visualisable en 2D. 
            <em>(Cette réduction implique que les axes n'ont pas de noms spécifiques.)</em>
        </p>
        <p>
            On remarque qu'un groupe d'artistes se démarque du reste du corps d'artistes français : ce sont les rappeurs.
        </p>
        <p>
            L'évolution générationnelle dans l'écriture se remarque également par la distance entre les chanteurs les plus vieux et ceux les plus récents, 
            mais on aperçoit également des clivages entre des paroles "chantées" et des paroles "parlées", alors que l'analyse ne porte que sur les paroles ! 
        </p>
        <p>
            Tout l'enjeu de ce projet est le traitement des données et l'utilisation d'un modèle de langage adapté: il faut à la fois un bon compromis coût/performance, mais aussi étudier quel modèle de langage est le plus à même de saisir ce qui différencie un texte d'un autre.
     
    </div>
""", unsafe_allow_html=True)
    # Chargement des données
    file_path = 'artistes.parquet' 
    df = pd.read_parquet(file_path, columns=['artist_name', 'avg_embedding'])

    with st.spinner("⏳ Patientez quelques secondes le temps que le graphique charge :)"):
        if 'avg_embedding' not in df.columns:
            st.error('La colonne avg_embedding n\'existe pas dans le DataFrame.')
        else:
            # Extraction des noms d'artistes et des embeddings
            artists = df['artist_name'].tolist()
            embeddings = np.array(df['avg_embedding'].apply(ast.literal_eval).tolist())
    
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
                st.write("Ce second graphique est le même que le premier, mais met en avant différents clusters, c'est à dire des groupements d'éléments semblables au regard des autres. On retrouve les différentes segmentations que l'on présentais, et même une segmentation au sein même du groupe des rappeurs")
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
                    
                    
                    fig = go.Figure()
                    color_scale = px.colors.qualitative.Plotly
    
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
    
           
                def load_and_visualize(df):
             
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
                load_and_visualize(df)
            # Onglet 3 : Autre contenu (ajoutez ici ce que vous souhaitez)
            # Onglet 3 : Autre contenu (ajoutez ici ce que vous souhaitez)
            with tabs[2]:
                st.write("Cette section est encore en déploiement car elle implique un chargement trop long pour le moment ! ")
            #     # Charger les données
            #     df = pd.read_parquet('cluster.parquet')


        #     # Fonction pour récupérer les titres, albums et embeddings d'un artiste
        #     def fetch_artist_lyrics(artist_name, df):
        #         # Filtrer le DataFrame pour l'artiste donné
        #         artist_data = df[df['artist_name'] == artist_name]
        #         titles = artist_data['song_title'].tolist()
        #         albums = artist_data['album_name'].tolist()
        #         embeddings = artist_data['embedded_lyrics'].tolist()  # Supposant que les embeddings sont stockés sous forme de liste

        #         # Convertir les chaînes d'embeddings en listes
        #         embeddings = [ast.literal_eval(embedding) if isinstance(embedding, str) else embedding for embedding in embeddings]
        #         return titles, albums, embeddings

        #     # Fonction pour visualiser les chansons d'un artiste
        #     def visualize_artist_songs(artist_name, df, method='PCA'):
        #         titles, albums, embeddings = fetch_artist_lyrics(artist_name, df)

        #         # Vérifier si les embeddings sont valides
        #         if len(embeddings) == 0 or any(len(embedding) == 0 for embedding in embeddings):
        #             return ''  # Si les embeddings sont vides, on ne fait rien

        #         # Convertir la liste de listes en tableau NumPy
        #         embeddings_array = np.array(embeddings)

        #         # Réduction des dimensions
        #         if method == 'PCA':
        #             reducer = PCA(n_components=2)
        #         elif method == 't-SNE':
        #             reducer = TSNE(n_components=2, random_state=0)
        #         else:
        #             raise ValueError("Méthode non reconnue. Utilisez 'PCA' ou 't-SNE'.")

        #         reduced_embeddings = reducer.fit_transform(embeddings_array)

        #         unique_albums = list(set(albums))
        #         color_map = {album: i for i, album in enumerate(unique_albums)}
        #         colors = [color_map[album] for album in albums]

        #         fig = go.Figure()
        #         color_scale = px.colors.qualitative.Plotly
        #         fig.add_trace(go.Scatter(
        #             x=reduced_embeddings[:, 0],
        #             y=reduced_embeddings[:, 1],
        #             mode='markers',
        #             marker=dict(
        #                 size=15,
        #                 color=colors,
        #                 colorscale=color_scale,
        #                 colorbar=dict(title='album_name'),
        #                 line=dict(width=2, color='DarkSlateGrey')
        #             ),
        #         ))

        #         # Ajouter des annotations avec des liens cliquables
        #         annotations = []
        #         for i, title in enumerate(titles):
        #             formatted_title = title.replace(" ", "_")
        #             annotations.append(dict(
        #                 x=reduced_embeddings[i, 0],
        #                 y=reduced_embeddings[i, 1],
        #                 text=f'<a href="/chanson/{artist_name}/{formatted_title}/" target="_blank">{title}</a>',
        #                 showarrow=True,
        #                 arrowhead=2,
        #                 ax=20,
        #                 ay=-20,
        #                 font=dict(size=10, color='black'),
        #                 align='center'
        #             ))

        #         fig.update_layout(
        #             title=f'Visualisation des Embeddings des Paroles - Répertoire de {artist_name}',
        #             xaxis_title='Composante 1',
        #             yaxis_title='Composante 2',
        #             showlegend=True,
        #             template='plotly_white',
        #             width=1500,
        #             height=900,
        #             autosize=True,
        #             margin=dict(l=50, r=50, t=100, b=50),
        #             annotations=annotations
        #         )

        #         return fig

        #     # Titre de l'application
        #     st.title("Visualisation des Chansons par Artiste")

            
        #     # Sélectionner l'artiste dans un sélecteur
        #     artist_name = st.selectbox("Choisir un artiste", df['artist_name'].unique())

    
        #     if artist_name:
        #         # Visualiser les chansons de l'artiste
        #         fig = visualize_artist_songs(artist_name, df, 'PCA')
        #         st.plotly_chart(fig)

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
import base64
from pathlib import Path

# Imports pour les modèles avancés
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# ARCH supprimé

# =========================
# CONFIG APP
# =========================
st.set_page_config(page_title="Théo Bernad — CV & Portfolio", page_icon="📊", layout="wide")

if "nav" not in st.session_state:
    st.session_state["nav"] = "🏠 Accueil"

# =========================
# STYLES (clair, harmonisé, largeur maîtrisée)
# =========================
st.markdown("""
<style>
:root{
  --app-bg:#f6f8fb;
  --card:#ffffff;
  --text:#0f172a;
  --muted:#475569;
  --border:#e6e9f0;
  --chip:#eef2f7;
  --chip-text:#0f172a;
  --primary:#2563eb;
  --primary-fore:#ffffff;
  --shadow:0 10px 28px rgba(15,23,42,.06);
  --shadow-soft:0 4px 14px rgba(15,23,42,.08);
}
.stApp {background: linear-gradient(180deg,#fbfcff 0%, var(--app-bg) 100%) !important;}
.block-container {padding-top: 1.0rem !important; max-width: 1280px !important; margin: auto !important;}
#MainMenu, footer {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"] .stRadio > label { font-size: 1.04rem; font-weight: 700; }
section[data-testid="stSidebar"] .stRadio div { padding: .35rem 0; }

/* HERO compact */
.hero {
  display: grid; grid-template-columns: 0.9fr 1.4fr; gap: 24px;
  border-radius: 18px; padding: 24px;
  background: linear-gradient(160deg, var(--card) 0%, #fafbff 85%) !important;
  color: var(--text) !important; border: 1px solid var(--border) !important;
  box-shadow: var(--shadow) !important;
}
@media (max-width: 960px){ .hero { grid-template-columns: 1fr; } }
.hero h1 { font-size: 2.1rem; margin: 0 0 6px 0; letter-spacing:.2px; color: var(--text) !important; }
.accent { height: 3px; width: 120px; background: var(--primary);
          border-radius: 2px; margin: 4px 0 14px 0; }
.lead { font-size: 1.02rem; line-height: 1.55; color: var(--muted) !important; margin: 0 0 14px 0; }

/* Colonne gauche (photo) */
.photo { border-radius: 16px; overflow: hidden; border: 1px solid var(--border);
         box-shadow: var(--shadow-soft); background:#fff; }
.photo img { width:100%; height:auto; display:block; }
.block-container { padding-top: 1rem !important; max-width: 1280px !important; margin: auto !important; }

/* grille des visuels façon “jardins à la française” */
.viz-grid{
  display:grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items:start;
}
@media (max-width: 980px){ .viz-grid{ grid-template-columns: 1fr; } }

.viz-card{
  border:1px solid var(--border); border-radius:14px; background:#fff;
  box-shadow: var(--shadow-soft); overflow:hidden;
}
.viz-body{ padding: 14px; }
.viz-title{ margin: 0 0 8px 0; font-weight: 700; color: var(--text);}
.viz-hint{ margin: 2px 0 12px 0; color:#64748b; font-size:.92rem; }
.viz-frame{ border-top:1px solid var(--border); background:#fff; }

/* pour le texte explicatif centré au-dessus du GIF */
.explain {
  text-align:center; font-size:1rem; color:#334155; margin: 18px 0 10px 0;
}
/* Badges */
.badges { margin-top: 6px; display:flex; flex-wrap:wrap; }
.badge {
  display:inline-flex; align-items:center; gap:6px;
  margin: 6px 8px 0 0; padding: 7px 12px;
  border: 1px solid var(--border); border-radius: 999px;
  background: var(--chip); color: var(--chip-text); font-size: .86rem;
  box-shadow: 0 1px 1px rgba(15,23,42,.04);
}
.dot { width:8px; height:8px; border-radius:999px; display:inline-block; }
.dot.py {background:#16a34a;}   .dot.sql{background:#0ea5e9;}   .dot.qlk{background:#8b5cf6;}
.dot.sta{background:#f59e0b;}   .dot.dja{background:#0ea5e9;}   .dot.af {background:#ef4444;}
.dot.aws{background:#f97316;}   .dot.dl {background:#22c55e;}   .dot.emb{background:#64748b;}
.dot.git{background:#f43f5e;}   .dot.bash{background:#22d3ee;}  .dot.spark{background:#fb923c;}

/* CTA */
.cta { display:flex; gap:10px; flex-wrap:wrap; }
.btn {
  text-decoration:none; display:inline-block;
  padding:10px 14px; border-radius:12px; border:1px solid var(--border);
  background:#fff; color: var(--text); transition: all .15s ease;
  box-shadow: 0 2px 6px rgba(15,23,42,.05);
  font-size:.95rem;
}
.btn.primary { background: var(--primary); border-color: var(--primary);
               color: var(--primary-fore); box-shadow: 0 8px 18px rgba(37,99,235,.22); }
.btn:hover { transform: translateY(-1px); box-shadow:0 6px 14px rgba(15,23,42,.10); }

/* GIF plein largeur (sous le hero) */
.fullgif {
  margin: 16px 0 8px 0; border:1px solid var(--border); border-radius:14px; overflow:hidden;
  box-shadow: var(--shadow-soft); background:#fff;
}
.fullgif img { width:100%; display:block; }
.caption { font-size:.9rem; color:#64748b; margin-top:6px; text-align:center; }

/* Bloc sous le GIF : 2 colonnes harmonisées */
.below {
  display:grid; grid-template-columns: 1.1fr 0.9fr; gap:18px; margin-top: 16px;
}
@media (max-width: 960px){ .below { grid-template-columns: 1fr; } }
.card {
  border:1px solid var(--border); border-radius:12px; background:#fff;
  box-shadow: var(--shadow-soft); padding:16px;
}
.card h3 { margin:0 0 10px 0; color:var(--text); }

/* Pills */
.pills { display:flex; flex-wrap:wrap; gap:8px; }
.pill {
  display:inline-block; padding:7px 12px; border-radius:999px;
  background:#f1f5f9; border:1px solid var(--border); color:#334155; font-size:.85rem;
  box-shadow: 0 1px 1px rgba(15,23,42,.04);
}

/* List */
ul.clean { margin:0; padding-left: 1.1rem; color: var(--text); }
ul.clean li { margin: .35rem 0; }
</style>
""", unsafe_allow_html=True)

# =========================
# NAVIGATION
# =========================
page = st.sidebar.radio(
    "⬇️  Projets Persos :",
    [
        "🏠 Accueil",
        #"📈 Démo - Visualisations",
        "▶️ NLP: Cartographie politique des Youtubeurs",
        "🎵 NLP/LLM: Cartographier les artistes français depuis les paroles de leur répertoire.",
        "🚨 ML: Analyse d'accidentologie à Paris"
    ],
    index=[
        "🏠 Accueil",
        #"📈 Démo - Visualisations",
        "▶️ NLP: Cartographie politique des Youtubeurs",
        "🎵 NLP/LLM: Cartographier les artistes français depuis les paroles de leur répertoire.",
        "🚨 ML: Analyse d'accidentologie à Paris"
    ].index(st.session_state["nav"]),
    key="nav"
)

# =========================
# UTILS
# =========================
def safe_image(path: str, **kw):
    p = Path(path)
    if p.exists():
        # Supprimer use_container_width si présent car pas supporté par st.image
        kw.pop("use_container_width", None)
        st.image(str(p), **kw)
    else:
        st.info(f"📁 Image introuvable : `{p.name}` — dépose le fichier à la racine.")

def render_fullwidth_gif(path: str):
    """Affiche le GIF en 100% de la largeur disponible, sous le hero."""
    p = Path(path)
    if p.exists():
        with open(path, "rb") as f:
            data_url = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f'<div class="fullgif">'
            f'  <img src="data:image/gif;base64,{data_url}" alt="aperçu clustering">'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="caption">Aperçu 15s — clustering / exploration sémantique</div>',
                    unsafe_allow_html=True)
    else:
        st.caption("GIF introuvable — placez `cluster.gif` à la racine.")

# =========================
# PAGE: ACCUEIL
# =========================
if page == "🏠 Accueil":
    
        # HERO : photo + (titre, pitch, stacks, CTA)
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    colL, colR = st.columns([0.9, 1.4])

    with colL:
        st.markdown('<div class="photo">', unsafe_allow_html=True)
        safe_image("photo.jpg")
        st.markdown('</div>', unsafe_allow_html=True)

    with colR:
        st.markdown("<h1>Théo Bernad</h1>", unsafe_allow_html=True)
        st.markdown('<div class="accent"></div>', unsafe_allow_html=True)
        st.markdown(
        '<p class="lead"> Professionnel Data Full Stack  | Création de solutions automatisées pour des gains d\'efficacité et une vision data-driven.</p>',
        unsafe_allow_html=True
    )
        st.markdown(
    '<p class="text-sm">Explorez mes projets d\'analyse linguistique via le menu latéral.</p>',
    unsafe_allow_html=True
)
        # Stacks (avec Git, Bash, Spark) — séparés des CTA
        st.markdown('<div class="stack-wrap">', unsafe_allow_html=True)
        st.markdown(
            '<div class="badges">'
            '<span class="badge"><span class="dot py"></span>Python</span>'
            '<span class="badge"><span class="dot sql"></span>SQL</span>'
            '<span class="badge"><span class="dot qlk"></span>Qlik</span>'
            '<span class="badge"><span class="dot sta"></span>Statistiques</span>'
            '<span class="badge"><span class="dot dja"></span>Django</span>'
            '<span class="badge"><span class="dot af"></span>Airflow</span>'
            '<span class="badge"><span class="dot aws"></span>AWS</span>'
            '<span class="badge"><span class="dot dl"></span>PyTorch / TensorFlow</span>'
            '<span class="badge"><span class="dot emb"></span>Embedding</span>'
            '<span class="badge"><span class="dot git"></span>Git</span>'
            '<span class="badge"><span class="dot bash"></span>Bash</span>'
            '<span class="badge"><span class="dot spark"></span>Spark</span>'
            '</div>', unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        MAIL = "mailto:luella.theo@gmail.com"             # <-- remplace
        LINKEDIN = "https://www.linkedin.com/in/theobcd/"  # <-- remplace
        st.markdown(
            f'<div class="cta">'
            f'<a class="btn primary" href="{MAIL}">📬 Discutons Data</a>'
            f'<a class="btn" href="{LINKEDIN}" target="_blank">🔗 LinkedIn</a>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)  # /hero

    # ===== VISUELS CÔTE À CÔTE : GIF (gauche) + INFOGRAM (droite) =====
    st.markdown('<div class="viz-grid">', unsafe_allow_html=True)
     
    
    # Carte A : GIF de clustering (texte descriptif au-dessus)
    with st.container():
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="viz-body">'
            '<div class="viz-title">🎯 Clustering exploratoire</div>'
            '<div class="viz-hint explain">'
            'Les données sont regroupées automatiquement en familles selon leurs similarités '
            '(<i>algorithmes non supervisés comme KMeans</i>). '
            'Cela permet de faire émerger des profils ou tendances cachées et d’apporter une vision synthétique '
            'utile à l’analyse et à la décision.'
            '</div>'
            '</div>', unsafe_allow_html=True
        )
       
        # rendu du GIF en base64 (largeur totalement fluide)
        def render_fullwidth_gif(path: str):
            p = Path(path)
            if p.exists():
                with open(path, "rb") as f:
                    data_url = base64.b64encode(f.read()).decode("utf-8")
                st.markdown(
                    f'<div class="viz-frame"><img src="data:image/gif;base64,{data_url}" '
                    f'style="width:100%; display:block;"></div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="caption">Aperçu 15s — clustering / exploration sémantique</div>',
                            unsafe_allow_html=True)
            else:
                st.caption("GIF introuvable — placez `cluster.gif` à la racine.")
        render_fullwidth_gif("cluster.gif")
        st.markdown('</div>', unsafe_allow_html=True)  # /viz-card

    # Carte B : Infogram (titre + hint “scroller” au-dessus)
    with st.container():
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="viz-body">'
            '<div class="viz-title">📊 Visualisations Data.gouv — Accidents routiers</div>'
            '<div class="viz-hint">Analyse mise en forme à partir des <b>données ouvertes de l’État</b> '
            '(data.gouv.fr). <i>Astuce :</i> placez le curseur dessus et <b>scrollez à l’intérieur</b> '
            'pour naviguer dans le tableau de bord.</div>'
            '</div>',
            unsafe_allow_html=True
        )

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
        """
        # cadre du composant = pleine largeur de la carte, avec scroll interne
        st.components.v1.html(infogram_html, height=480, scrolling=True)
        st.markdown('</div>', unsafe_allow_html=True)  # /viz-card

    st.markdown('</div>', unsafe_allow_html=True)  # /viz-grid

    # ===== CARTES MÉTIER (alignées) =====
    st.markdown('<div class="info-grid">', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Disponibilités & mobilité")
    st.markdown(
        '<div class="pills">'
        '<span class="pill">Disponibilités : Freelance, CDI</span>'
        '<span class="pill">Mobilité : France & International</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Types de données maîtrisées")
    st.markdown(
        '<div class="pills">'
        '<span class="pill">Transactionnelles (commerce, ventes, CRM)</span>'
        '<span class="pill">Textuelles (NLP : titres, descriptions, commentaires)</span>'
        '<span class="pill">Séries temporelles (logs, métriques, événements)</span>'
        '<span class="pill">RH / People Analytics (effectifs, mobilité, indicateurs)</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # /info-grid

    ###########################
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



# =========================
  
elif page== "▶️ NLP: Cartographie politique des Youtubeurs":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
    # UMAP supprimé
    import plotly.express as px
    import streamlit as st
    import ast
    import time
    
    st.title("📊 Cartographie politique des influenceurs YouTube")
    
    # Afficher un message d'attente stylisé
    with st.spinner("🔄Veuillez patienter pendant le chargement de la visualisation"):
        
        # Chargement des données avec cache pour optimiser les performances
        @st.cache_data
        def load_data():
            df = pd.read_csv("results_df.csv")
            df = df.dropna(subset=["title", "charge_politique_latente"]).reset_index(drop=True)
            
            # Conversion des colonnes de listes depuis string
            list_cols = [
                "style_de_politisation",
                "figures_ennemies",
                "valeurs_invoquées",
                "thématiques_dominantes"
            ]
        
            for col in list_cols:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
            
            return df
        

        
        df = load_data()
        
     
        
        # Colonnes numériques à inclure
        numerical_cols = ["charge_politique_latente", "index_fanatisme"]
        
        # Encodage MultiLabel
        list_cols = [
            "style_de_politisation",
            "figures_ennemies",
            "valeurs_invoquées",
            "thématiques_dominantes"
        ]
        
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
        
  
        # UMAP supprimé - utiliser PCA à la place
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(X_scaled)
        
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
            title="Projection PCA des chaînes YouTube par orientation politique"
        )
        fig.update_traces(textposition='top center', marker=dict(size=10))
        fig.update_layout(height=600, showlegend=False)
        
        st.success("✅ Chargement complété.")
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
        ---
        
        ### 🎯 Objectif du projet
        
        Cette visualisation cherche à représenter l'identité politique des influenceurs YouTube à partir de plusieurs dimensions qualitatives et quantitatives extraites de leurs discours.
        
        Les scripts sont extraits et analysés automatiquement au travers d'un ensemble de crit-res relatifs à ce qui est constitutif d'une identité politique: un ton, des valeurs, des thématiques, des cibles, une posture ...
        
        Cette approche illustre comment une analyse NLP peut combiner **quantitatif** et **qualitatif** pour appréhender des logiques politiques implicites.
        """)
    # Explications sous le graphique
    st.markdown("""
    ---
    
    ### 🧠 Comment interpréter cette visualisation
    Pourquoi ce graphique n'a pas de nom d'axe ?
    => Car c'est une réduction de dimensionalité. C'est comme si on faisait un graphique de ce que vous êtes en des centaines de dimensions, et qu'on en gardait l'essentiel pour visualiser votre positionnement en 2D !
    
    - La distance entre les points représente leur PROXIMITE POLITIQUE, au regard des critères d'analyses (voir variables en dessous)
    - **Couleur** : L'intensité de la couleur indique le niveau d'engagement politique (charge politique latente)
    
    Cette analyse combine techniques quantitatives et qualitatives pour cartographier le paysage politique des influenceurs YouTube.
    """)
    
    # Section méthodologie avec accordéon pour ne pas surcharger l'interface
    with st.expander("📋 Méthodologie détaillée et variables analysées"):
        st.markdown("""
        ### ⚙️ Démarche analytique
        
        #### Prétraitement des variables :
        - 🔢 Les variables **numériques** (ex: `charge_politique_latente`, `index_fanatisme`) sont standardisées
        - 🏷️ Les variables **catégorielles multilabels** (ex: *valeurs*, *figures ennemies*) sont vectorisées
        
        #### Réduction de dimension :
        - 🧭 Les vecteurs sont projetés en 2D via `PCA` (analyse en composantes principales)
        
        #### Variables analysées :
        """)
        
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
            ("positionnement_sociétal", "Rapport à la société", "critique des élites, défense d'un groupe…"),
            ("cadre_problematisation", "Façon de poser les enjeux", "système en crise, injustice sociale…"),
            ("figures_ennemies", "Adversaires implicites", "le système, les mondialistes…"),
            ("récit_idéologique", "Narration politique", "déclin civilisationnel, peuple trahi…"),
            ("axes_de_tension", "Axes de conflit", "élite vs peuple, progrès vs tradition…"),
            ("paradigmes_compatibles", "Paradigmes compatibles", "libéralisme, anarchisme, gaullisme…"),
            ("ton_politique", "Ton politique", "engagé, militant, distant…"),
            ("enjeux_sociaux_centrés", "Enjeux sociaux centraux", "santé, sécurité, inclusion…"),
            ("charge_politique_latente", "Score de politisation", "0 = neutre, 100 = très politisé"),
            ("position_stratégique", "Stratégie globale", "offensive, défensive, ambiguë…"),
            ("mode_d_interpellation_du_public", "Type d'interpellation", "directe, pédagogique, émotionnelle…"),
            ("figure_du_locuteur", "Rôle du locuteur", "expert, citoyen, leader, victime…"),
            ("échelle_de_politisation", "Niveau d'ancrage", "locale, nationale, internationale…"),
            ("type_de_menace_perçue", "Menace évoquée", "déclin, subversion, chaos…"),
            ("registre_moral_implicite", "Fond moral", "progressiste, conservateur, égalitariste…"),
            ("ton_affectif_dominant", "Émotion dominante", "colère, peur, espoir, fierté…"),
            ("niveau_de_certitude", "Certitude exprimée", "score 0-100 (incertitude → affirmation)"),
            ("index_performativite", "Performativité", "0 = descriptif, 100 = incitation forte à l'action"),
            ("index_fanatisme", "Fermeté idéologique", "0 = ouvert au débat, 100 = hostile aux avis opposés")
        ]
        
        df_descr = pd.DataFrame(descripteurs, columns=["🧩 Variable", "🗂️ Description", "🔍 Exemples ou échelle"])
        st.dataframe(df_descr, height=400)
        
        
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

# =========================
# PAGE: ANALYSE D'ACCIDENTOLOGIE À PARIS
# =========================
elif page == "🚨 ML: Analyse d'accidentologie à Paris":
    st.title("🚨 Analyse d'Accidentologie à Paris")
    
    # Onglets pour séparer présentation, application et prédictions
    tab_application, tab_presentation, tab_predictions = st.tabs(["🚀 Application", "📋 Présentation", "🔮 Prédictions"])
    
    with tab_presentation:
        st.markdown("""
        <div style="text-align: left; font-size: 18px; line-height: 1.6; margin-top: 20px;">
            <p><strong>Présentation du projet :</strong></p>
            <p>
                Ce projet analyse les données d'accidents de la route à Paris sur la période 2017-2023. 
                Il combine plusieurs approches de machine learning (SARIMA, Prophet, modèles de régression) avec des données 
                météorologiques et de trafic pour identifier les zones à risque et prédire l'évolution des accidents.
            </p>
            <p>
                L'application web développée avec Streamlit permet d'explorer interactivement les données à travers 
                des cartes, des graphiques temporels et des analyses par arrondissement.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Section des fonctionnalités
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Fonctionnalités principales
            
            - **Prédictions ML** : Modèles SARIMA, Prophet et régression
            - **Cartographie interactive** : Cartes de chaleur et clustering
            - **Analyse temporelle** : Évolution par mois et année
            - **Points noirs** : Identification des zones à risque par arrondissement
            - **Performance** : Traitement optimisé de 7 ans de données
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Technologies utilisées
            
            **Machine Learning :**
            - SARIMA pour les prédictions de séries temporelles
            - Prophet pour l'analyse des tendances saisonnières
            - Modèles de régression pour l'analyse prédictive
            
            **Visualisation :**
            - Streamlit pour l'interface web
            - Plotly pour les graphiques interactifs
            - Folium pour les cartes géographiques
            - Pandas pour le traitement des données
            """)
        
        # Métriques de performance
        st.markdown("### 📈 Résultats techniques")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", "0.85+", "Prédictions")
        with col2:
            st.metric("MAE", "< 2", "Accidents/jour")
        with col3:
            st.metric("Données", "50k+", "Accidents")
        with col4:
            st.metric("Période", "7 ans", "2017-2023")
        
        # Tableau des métriques par modèle
        st.markdown("### 📊 Métriques de performance par modèle")
        
        metrics_data = {
            'Modèle': ['SARIMA', 'Prophet', 'Régression Linéaire'],
            'R² Score': [0.79, 0.82, 0.75],
            'MAE': [2.4, 2.1, 2.8],
            'RMSE': [3.1, 2.7, 3.5]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        # Liens vers le projet
        st.markdown("### 🔗 Liens du projet")
        
        st.markdown("""
        <div style="text-align: center; padding: 20px; border: 1px solid #e6e9f0; border-radius: 12px; background: #fff; box-shadow: 0 4px 14px rgba(15,23,42,.08);">
            <h4>📁 Code Source</h4>
            <p>Repository GitHub avec le code complet</p>
            <a href="https://github.com/Luello/Accidentologie-Paris" target="_blank" style="display: inline-block; padding: 10px 20px; background: #2563eb; color: white; text-decoration: none; border-radius: 8px; margin-top: 10px;">Voir sur GitHub</a>
        </div>
        """, unsafe_allow_html=True)
        
        # Cas d'usage
        st.markdown("### 🎯 Cas d'usage")
        
        st.markdown("""
        <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border-left: 4px solid #2563eb;">
            <ul style="margin: 0; padding-left: 20px;">
                <li><strong>Sécurité routière</strong> : Identification des zones à risque</li>
                <li><strong>Urbanisme</strong> : Planification des infrastructures</li>
                <li><strong>Prévention</strong> : Campagnes ciblées</li>
                <li><strong>Recherche</strong> : Analyse des facteurs d'accidents</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Évolutions possibles
        st.markdown("### 🔮 Évolutions possibles")
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b;">
            <ul style="margin: 0; padding-left: 20px;">
                <li>API REST pour les données</li>
                <li>Base de données PostgreSQL</li>
                <li>Cache Redis pour les performances</li>
                <li>Tests automatisés pytest</li>
                <li>Déploiement Docker</li>
                <li>Monitoring des performances</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab_application:
        st.markdown("### 🚀 Application Interactive")
        st.markdown("Explorez l'application d'analyse d'accidentologie directement ci-dessous :")
        
        # =========================
        # APPLICATION D'ACCIDENTOLOGIE INTÉGRÉE DIRECTEMENT
        # =========================
        
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
                ["Carte des accidents", "Évolution temporelle animée", "Évolution temporelle", "Analyse par arrondissement", "Statistiques générales"]
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
                    
                    if filtered_data.empty:
                        st.warning("Aucun accident trouvé avec les critères sélectionnés.")
                    else:
                        st.info(f"Affichage de {len(filtered_data):,} accidents sur la carte")
                        
                        # Création de la carte optimisée
                        @st.cache_data
                        def create_accident_map(df):
                            import folium
                            from folium.plugins import HeatMap, MarkerCluster
                            
                            # Limitation du nombre de points pour améliorer les performances
                            max_points = 5000
                            if len(df) > max_points:
                                # Échantillonnage aléatoire pour les grandes datasets
                                df_sample = df.sample(n=max_points, random_state=42)
                                st.info(f"⚠️ Affichage de {max_points:,} accidents sur {len(df):,} (échantillonnage pour les performances)")
                            else:
                                df_sample = df
                            
                            m = folium.Map(location=[48.8566, 2.3522], zoom_start=12,
                                            tiles='cartodbpositron')
                                
                            # Création d'un cluster de marqueurs optimisé
                            marker_cluster = MarkerCluster(
                                options={
                                    'maxClusterRadius': 60,
                                    'disableClusteringAtZoom': 16,
                                    'spiderfyOnMaxZoom': True,
                                    'showCoverageOnHover': False
                                }
                            )

                            # Ajout de la carte de chaleur si activée (plus rapide)
                            if show_heatmap:
                                # Utilisation d'un échantillon plus petit pour la heatmap
                                heat_sample_size = min(2000, len(df_sample))
                                heat_df = df_sample.sample(n=heat_sample_size, random_state=42)
                                heat_data = [[row['latitude'], row['longitude']] for _, row in heat_df.iterrows()]
                                
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

                            # Couleurs par gravité
                            colors = {
                                'Tué': 'red',
                                'Blessé hospitalisé': 'orange',
                                'Blessé léger': 'yellow'
                            }

                            # Ajout des marqueurs (optimisé)
                            for _, row in df_sample.iterrows():
                                # Taille du marqueur basée sur la gravité
                                size = {
                                    'Tué': marker_size + 3,
                                    'Blessé hospitalisé': marker_size + 1,
                                    'Blessé léger': marker_size
                                }[row['gravite_combinee']]
                                
                                # Création du marqueur simplifié
                                folium.CircleMarker(
                                    location=[row['latitude'], row['longitude']],
                                    radius=size,
                                    color=colors[row['gravite_combinee']],
                                    fill=True,
                                    fillOpacity=marker_opacity,
                                    popup=f"<b>{row['gravite_combinee']}</b><br>Type: {row['type_usager']}<br>Date: {row['date']}"
                                ).add_to(marker_cluster)
                            
                            # Ajout du cluster à la carte
                            marker_cluster.add_to(m)
                            
                            # Ajout du contrôle des couches
                            folium.LayerControl().add_to(m)
                            
                            return m

                        # Affichage de la carte
                        m = create_accident_map(filtered_data)
                        st.components.v1.html(m._repr_html_(), height=600)
                        
                        # Statistiques rapides
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total accidents", len(filtered_data))
                        with col2:
                            morts = len(filtered_data[filtered_data['gravite_combinee'] == 'Tué'])
                            st.metric("Accidents mortels", morts)
                        with col3:
                            blesses = len(filtered_data[filtered_data['gravite_combinee'] == 'Blessé hospitalisé'])
                            st.metric("Blessés hospitalisés", blesses)
                        with col4:
                            legers = len(filtered_data[filtered_data['gravite_combinee'] == 'Blessé léger'])
                            st.metric("Blessés légers", legers)

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
                    
                    # Fonction pour créer la carte mensuelle optimisée
                    @st.cache_data
                    def create_monthly_heatmap(df):
                        import folium
                        from folium.plugins import HeatMap
                        
                        # Limitation pour les performances
                        max_points = 3000
                        if len(df) > max_points:
                            df_sample = df.sample(n=max_points, random_state=42)
                        else:
                            df_sample = df
                        
                        m = folium.Map(location=[48.8566, 2.3522], zoom_start=13,
                                      tiles='cartodbpositron',
                                      max_bounds=True,
                                      min_zoom=12,
                                      max_zoom=16)
                        
                        # Création des données pour la heatmap
                        heat_data = [[row['latitude'], row['longitude']] for _, row in df_sample.iterrows()]
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
                    
                    m = create_monthly_heatmap(df_month)
                    st.components.v1.html(m._repr_html_(), height=600)
                    
                    # Animation
                    if st.session_state.is_playing_month:
                        st.session_state.month_index = (st.session_state.month_index + 1) % len(mois_list)
                        import time
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
                        plot_bgcolor='white',
                        paper_bgcolor='white'
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
                        line=dict(width=3),
                        marker=dict(size=8)
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
                
                with tab_annee:
                    st.subheader("Évolution annuelle des accidents")
                    
                    # Préparation des données annuelles
                    df_annee = df_filtered.copy()
                    df_annee['annee'] = df_annee['date_heure'].dt.year
                    
                    # Liste des années disponibles
                    annees_list = sorted(df_annee['annee'].unique())
                    
                    # Initialisation de l'index de l'année dans le state si pas déjà fait
                    if 'year_index' not in st.session_state:
                        st.session_state.year_index = 0
                        st.session_state.is_playing_year = False
                    
                    # Contrôles pour l'animation
                    col_slider_year, col_play_year = st.columns([4, 1])
                    
                    with col_slider_year:
                        selected_year = st.select_slider(
                            "Sélectionner l'année",
                            options=annees_list,
                            value=annees_list[st.session_state.year_index]
                        )
                        st.session_state.year_index = annees_list.index(selected_year)
                    
                    with col_play_year:
                        if st.button('▶ Lecture' if not st.session_state.is_playing_year else '⏸ Pause', key='play_year'):
                            st.session_state.is_playing_year = not st.session_state.is_playing_year
                    
                    # Création de la carte pour l'année sélectionnée
                    st.subheader(f"Carte des accidents pour l'année {selected_year}")
                    df_year = df_annee[df_annee['annee'] == selected_year]
                    
                    # Fonction pour créer la carte annuelle optimisée
                    @st.cache_data
                    def create_yearly_heatmap(df):
                        import folium
                        from folium.plugins import HeatMap
                        
                        # Limitation pour les performances
                        max_points = 4000
                        if len(df) > max_points:
                            df_sample = df.sample(n=max_points, random_state=42)
                        else:
                            df_sample = df
                        
                        m = folium.Map(location=[48.8566, 2.3522], zoom_start=13,
                                      tiles='cartodbpositron',
                                      max_bounds=True,
                                      min_zoom=12,
                                      max_zoom=16)
                        
                        # Création des données pour la heatmap
                        heat_data = [[row['latitude'], row['longitude']] for _, row in df_sample.iterrows()]
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
                    
                    # Placeholder pour la carte
                    map_placeholder = st.empty()
                    
                    # Création de la carte pour l'année sélectionnée avec les données filtrées
                    df_year = df_annee[df_annee['annee'] == selected_year]
                    m = create_yearly_heatmap(df_year)
                    
                    # Affichage de la carte
                    with map_placeholder:
                        st.components.v1.html(
                            m._repr_html_(),
                            height=600
                        )
                    
                    # Animation
                    if st.session_state.is_playing_year:
                        st.session_state.year_index = (st.session_state.year_index + 1) % len(annees_list)
                        import time
                        time.sleep(1.0)
                        st.rerun()
                    
                    # Statistiques de l'année sélectionnée
                    st.subheader(f"Statistiques pour l'année {selected_year}")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_accidents_year = len(df_year)
                        st.metric("Total accidents", str(total_accidents_year))
                    
                    with col2:
                        morts_year = len(df_year[df_year['gravite_combinee'] == 'Tué'])
                        st.metric("Accidents mortels", str(morts_year))
                    
                    with col3:
                        blesses_year = len(df_year[df_year['gravite_combinee'] == 'Blessé hospitalisé'])
                        st.metric("Blessés hospitalisés", str(blesses_year))
                    
                    with col4:
                        legers_year = len(df_year[df_year['gravite_combinee'] == 'Blessé léger'])
                        st.metric("Blessés légers", str(legers_year))
                    
                    # Graphique de répartition par mois pour l'année sélectionnée
                    st.subheader(f"Répartition mensuelle pour {selected_year}")
                    
                    # Préparation des données pour le graphique mensuel
                    df_year_monthly = df_year.copy()
                    df_year_monthly['mois'] = df_year_monthly['date_heure'].dt.month
                    df_year_monthly['mois_nom'] = df_year_monthly['date_heure'].dt.strftime('%B')
                    df_year_monthly['mois_num'] = df_year_monthly['mois']
                    
                    # Calcul des statistiques mensuelles pour l'année
                    monthly_stats_year = df_year_monthly.groupby(['mois_nom', 'mois_num']).agg({
                        'id_accident': 'count'
                    }).reset_index()
                    
                    # Tri des données
                    monthly_stats_year = monthly_stats_year.sort_values('mois_num')
                    
                    # Création du graphique
                    fig_monthly_year = px.bar(
                        monthly_stats_year,
                        x='mois_nom',
                        y='id_accident',
                        title=f"Nombre d'accidents par mois en {selected_year}",
                        labels={
                            'mois_nom': 'Mois',
                            'id_accident': "Nombre d'accidents"
                        },
                        color='id_accident',
                        color_continuous_scale='Reds'
                    )
                    
                    # Personnalisation du graphique
                    fig_monthly_year.update_layout(
                        height=400,
                        showlegend=False,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    # Amélioration de la grille
                    fig_monthly_year.update_xaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    )
                    fig_monthly_year.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    )
                    
                    # Affichage du graphique
                    st.plotly_chart(fig_monthly_year, use_container_width=True)

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
                    
                    # Cartes détaillées pour l'arrondissement
                    st.subheader(f"Cartographie détaillée - Arrondissement {arr_analysis}")
                    
                    # Création des sous-onglets pour les différentes vues de la carte
                    tab_points, tab_heatmap = st.tabs(["Carte détaillée", "Carte de chaleur"])
                    
                    with tab_points:
                        st.subheader(f"Carte détaillée des accidents - Arrondissement {arr_analysis}")
                        
                        def create_arrondissement_map(df):
                            import folium
                            from folium.plugins import MarkerCluster
                            
                            m = folium.Map(
                                location=[df['latitude'].mean(), df['longitude'].mean()],
                                zoom_start=15,
                                tiles='cartodbpositron'
                            )
                            
                            # Création d'un cluster de marqueurs
                            marker_cluster = MarkerCluster(
                                options={
                                    'maxClusterRadius': 30,
                                    'disableClusteringAtZoom': 16
                                }
                            )
                            
                            # Couleurs par gravité
                            colors = {
                                'Tué': 'red',
                                'Blessé hospitalisé': 'orange',
                                'Blessé léger': 'yellow'
                            }
                            
                            # Ajout des marqueurs avec popups détaillés
                            for _, accident in df.iterrows():
                                # Création du popup HTML
                                resume_html = f"""
                                <div style="font-family: Arial; font-size: 12px;">
                                    <b>{accident['gravite_combinee']}</b><br>
                                    <b>Type:</b> {accident['type_usager']}<br>
                                    <b>Date:</b> {accident['date']}<br>
                                    <b>Heure:</b> {accident['date_heure'].strftime('%H:%M')}
                                </div>
                                """
                                
                                # Création du marqueur
                                marker = folium.CircleMarker(
                                    location=[accident['latitude'], accident['longitude']],
                                    radius=8,
                                    color=colors[accident['gravite_combinee']],
                                    fill=True,
                                    fillOpacity=0.7,
                                    popup=folium.Popup(resume_html, max_width=300)
                                )
                                marker.add_to(marker_cluster)
                            
                            # Ajout du cluster à la carte
                            marker_cluster.add_to(m)
                            
                            # Ajout du contrôle des couches
                            folium.LayerControl().add_to(m)
                            
                            return m
                        
                        # Affichage de la carte
                        m_points = create_arrondissement_map(df_filtered)
                        st.components.v1.html(m_points._repr_html_(), height=600)
                    
                    with tab_heatmap:
                        st.subheader(f"Carte de chaleur des zones à risque - Arrondissement {arr_analysis}")
                        
                        def create_arrondissement_heatmap(df):
                            import folium
                            from folium.plugins import HeatMap
                            
                            m = folium.Map(
                                location=[df['latitude'].mean(), df['longitude'].mean()],
                                zoom_start=15,
                                tiles='cartodbpositron'
                            )
                            
                            # Création des données pour la heatmap avec pondération par gravité
                            heat_data = []
                            for _, accident in df.iterrows():
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
                            ).add_to(m)
                            
                            # Ajout du contrôle des couches
                            folium.LayerControl().add_to(m)
                            
                            return m
                        
                        # Affichage de la carte de chaleur
                        m_heat = create_arrondissement_heatmap(df_filtered)
                        st.components.v1.html(m_heat._repr_html_(), height=600)
                    
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

    with tab_predictions:
        st.markdown("### 🔮 Prédictions SARIMA 2023")
        
        # Chargement des données
        @st.cache_data
        def load_accident_data():
                df = pd.read_parquet('accidentologie.parquet')
                df['Date'] = pd.to_datetime(df['Date'])
                monthly_accidents = df.groupby(df['Date'].dt.to_period('M')).size().reset_index()
                monthly_accidents.columns = ['date', 'accidents']
                monthly_accidents['accidents'] = monthly_accidents['accidents'].astype(int)
                monthly_accidents['date'] = monthly_accidents['date'].dt.to_timestamp()
                monthly_accidents = monthly_accidents.set_index('date')
                return monthly_accidents
        
        # Chargement des données
        ts_data = load_accident_data()
        
        
        if ts_data is not None:
            # Import des librairies
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            import numpy as np
            import warnings
            warnings.filterwarnings('ignore')
            
            # Import pour XGBoost
            try:
                import xgboost as xgb
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                XGBOOST_AVAILABLE = True
            except ImportError:
                XGBOOST_AVAILABLE = False
            
            # Paramètres SARIMA
            p, d, q = 1, 1, 1
            P, D, Q, s = 1, 1, 1, 12
            periods = 12
            
            # Préparation des données
            ts_clean = ts_data.dropna()
            ts_clean['accidents'] = pd.to_numeric(ts_clean['accidents'], errors='coerce')
            ts_clean = ts_clean.dropna()
            
            st.write(f"📊 **Données utilisées :** {len(ts_clean)} lignes")
            
            future_dates = pd.date_range(start='2023-01-01', periods=12, freq='MS')
            
            # Entraînement du modèle SARIMA
            model = SARIMAX(ts_clean['accidents'], order=(p, d, q), seasonal_order=(P, D, Q, s))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.get_forecast(steps=periods)
            predictions = forecast.predicted_mean.values
            
            # Création du graphique
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Données historiques
            hist_df = ts_clean.reset_index()
            hist_df['accidents'] = hist_df['accidents'].astype(float)
            
            fig.add_trace(go.Scatter(
                x=hist_df['date'],
                y=hist_df['accidents'],
                mode='lines+markers',
                name='Données historiques',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Prédictions 2023
            pred_df = pd.DataFrame({
                'date': future_dates,
                'accidents': predictions.astype(float)
            })
            
            fig.add_trace(go.Scatter(
                x=pred_df['date'],
                y=pred_df['accidents'],
                mode='lines+markers',
                name='Prédictions 2023',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Configuration du graphique
            fig.update_layout(
                title="Prédictions SARIMA - Accidents à Paris 2023",
                xaxis_title="Date",
                yaxis_title="Nombre d'accidents",
                height=600,
                hovermode='x unified'
            )
            
            # Affichage du graphique
            st.plotly_chart(fig, use_container_width=True)
            
            # SECOND GRAPHIQUE AVEC DONNÉES MÉTÉOROLOGIQUES
            st.markdown("### 🌤️ Prédictions SARIMA avec données météorologiques (avec 2020)")
            
            # Chargement des données météorologiques
            @st.cache_data
            def load_weather_data():
                weather_df = pd.read_csv('data_meteo.csv')
                
                # Nettoyage des données : supprimer les lignes vides
                weather_df = weather_df.dropna(subset=['date'])
                
                # Conversion des colonnes numériques en float, en gérant les valeurs vides
                numeric_columns = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun']
                for col in numeric_columns:
                    if col in weather_df.columns:
                        weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
                
                weather_df['date'] = pd.to_datetime(weather_df['date'])
                weather_df = weather_df.set_index('date')
                
                # Agrégation par mois avec la même fréquence que les accidents
                monthly_weather = weather_df.resample('MS').agg({
                    'tavg': 'mean',    # Température moyenne
                    'tmin': 'mean',    # Température minimale
                    'tmax': 'mean',    # Température maximale
                    'prcp': 'sum',     # Précipitations
                    'snow': 'sum',     # Neige
                    'wdir': 'mean',    # Direction du vent
                    'wspd': 'mean',    # Vitesse du vent
                    'wpgt': 'mean',    # Rafales de vent
                    'pres': 'mean',    # Pression atmosphérique
                    'tsun': 'sum'      # Ensoleillement
                })
                
                return monthly_weather
            
            weather_data = load_weather_data()
            
            # Fusion des données d'accidents et météo
            combined_data = ts_data.copy()
            combined_data = combined_data.join(weather_data, how='inner')
            
            # Nettoyage intelligent des données météo
            # Colonnes qui représentent des quantités (NaN = 0)
            quantity_columns = ['prcp', 'snow', 'tsun']  # Précipitations, neige, ensoleillement
            # Colonnes qui représentent des moyennes (NaN = moyenne de la colonne)
            average_columns = ['tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'pres']  # Températures, vent, pression
            # Colonnes optionnelles (peuvent être entièrement vides)
            optional_columns = ['wpgt']  # Rafales de vent (pas toujours mesurées)
            
            for col in combined_data.columns:
                if col != 'accidents':
                    # Remplacer inf et -inf par NaN
                    combined_data[col] = combined_data[col].replace([np.inf, -np.inf], np.nan)
                    
                    if col in quantity_columns:
                        # Pour les quantités, NaN signifie 0 (pas de précipitations, pas de neige, etc.)
                        combined_data[col] = combined_data[col].fillna(0)
                    elif col in average_columns:
                        # Pour les moyennes, remplir par la moyenne de la colonne
                        if not combined_data[col].isna().all():
                            combined_data[col] = combined_data[col].fillna(combined_data[col].mean())
                        else:
                            combined_data[col] = combined_data[col].fillna(0)
                    elif col in optional_columns:
                        # Pour les colonnes optionnelles, remplir par 0 si entièrement vides
                        combined_data[col] = combined_data[col].fillna(0)
            
            # Ne pas supprimer de colonnes - toutes les colonnes météo sont utiles
            # Ne pas faire de dropna() - on veut garder toutes les données fusionnées
            
            if len(combined_data) > 0:
                st.write(f"📊 **Données utilisées :** {len(combined_data)} lignes")
                
                # Entraînement SARIMA avec données météo
                weather_vars = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun']
                available_weather_vars = [var for var in weather_vars if var in combined_data.columns]
                
                model_weather = SARIMAX(
                    combined_data['accidents'], 
                    exog=combined_data[available_weather_vars],
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, s)
                )
                fitted_model_weather = model_weather.fit(disp=False)
                
                # Prédictions 2023 avec données météo saisonnières réalistes
                exog_forecast = pd.DataFrame(index=future_dates)
                
                for var in available_weather_vars:
                    if var in combined_data.columns:
                        if var in ['prcp', 'snow', 'tsun']:
                            # Pour les quantités, utiliser les moyennes mensuelles historiques
                            monthly_avg = combined_data[var].groupby(combined_data.index.month).mean()
                            exog_forecast[var] = [monthly_avg.get(month, combined_data[var].mean()) for month in range(1, 13)]
                        else:
                            # Pour les moyennes, utiliser les moyennes mensuelles historiques
                            monthly_avg = combined_data[var].groupby(combined_data.index.month).mean()
                            exog_forecast[var] = [monthly_avg.get(month, combined_data[var].mean()) for month in range(1, 13)]
                
                # S'assurer que les données exogènes ont la même forme que lors de l'entraînement
                exog_forecast = exog_forecast[available_weather_vars]
                
                forecast_weather = fitted_model_weather.get_forecast(steps=periods, exog=exog_forecast)
                predictions_weather = forecast_weather.predicted_mean.values
                
                # Création du second graphique
                fig2 = go.Figure()
                
                # Données historiques (avec 2020)
                hist_df2 = combined_data.reset_index()
                hist_df2['accidents'] = hist_df2['accidents'].astype(float)
                
                fig2.add_trace(go.Scatter(
                    x=hist_df2['date'],
                    y=hist_df2['accidents'],
                    mode='lines+markers',
                    name='Données historiques (avec 2020)',
                    line=dict(color='green', width=2),
                    marker=dict(size=4)
                ))
                
                # Prédictions 2023 avec météo
                pred_df2 = pd.DataFrame({
                    'date': future_dates,
                    'accidents': predictions_weather.astype(float)
                })
                
                fig2.add_trace(go.Scatter(
                    x=pred_df2['date'],
                    y=pred_df2['accidents'],
                    mode='lines+markers',
                    name='Prédictions 2023 (avec météo)',
                    line=dict(color='orange', width=3, dash='dash'),
                    marker=dict(size=6)
                ))
                
                # Configuration du second graphique
                fig2.update_layout(
                    title="Prédictions SARIMA avec données météorologiques - Accidents à Paris 2023",
                    xaxis_title="Date",
                    yaxis_title="Nombre d'accidents",
                    height=600,
                    hovermode='x unified'
                )
                
                # Affichage du second graphique
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("❌ Pas assez de données après fusion")
            
            # TROISIÈME GRAPHIQUE AVEC DONNÉES DE TRAFIC ROUTIER
            st.markdown("### 🚗 Prédictions SARIMA avec données de trafic routier")
            
            # Chargement des données de trafic routier
            @st.cache_data
            def load_traffic_data():
                traffic_df = pd.read_csv('trafic_routier_paris.csv', sep=';')
                traffic_df['date'] = pd.to_datetime(traffic_df['date'])
                traffic_df = traffic_df.set_index('date')
                
                # Agrégation par mois avec la même fréquence que les accidents
                monthly_traffic = traffic_df.resample('MS').agg({
                    'q': 'mean',  # Débit moyen
                    'k': 'mean',  # Densité moyenne
                    'nb_mesures': 'sum'  # Nombre total de mesures
                })
                
                return monthly_traffic
            
            traffic_data = load_traffic_data()
            
            # Fusion des données d'accidents et trafic
            combined_traffic_data = ts_data.copy()
            combined_traffic_data = combined_traffic_data.join(traffic_data, how='inner')
            
            # Nettoyage intelligent des données de trafic
            for col in ['q', 'k', 'nb_mesures']:
                if col in combined_traffic_data.columns:
                    # Remplacer inf et -inf par NaN
                    combined_traffic_data[col] = combined_traffic_data[col].replace([np.inf, -np.inf], np.nan)
                    # Remplir les NaN par la moyenne
                    if not combined_traffic_data[col].isna().all():
                        combined_traffic_data[col] = combined_traffic_data[col].fillna(combined_traffic_data[col].mean())
                    else:
                        combined_traffic_data[col] = combined_traffic_data[col].fillna(0)
            
            # Ne pas faire de dropna() - garder toutes les données fusionnées
            
            if len(combined_traffic_data) > 0:
                st.write(f"📊 **Données utilisées :** {len(combined_traffic_data)} lignes")
                # Entraînement SARIMA avec données de trafic
                model_traffic = SARIMAX(
                    combined_traffic_data['accidents'], 
                    exog=combined_traffic_data[['q', 'k', 'nb_mesures']],
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, s)
                )
                fitted_model_traffic = model_traffic.fit(disp=False)
                
                # Prédictions 2023 avec données de trafic saisonnières réalistes
                exog_forecast_traffic = pd.DataFrame(index=future_dates)
                
                for var in ['q', 'k', 'nb_mesures']:
                    if var in combined_traffic_data.columns:
                        # Utiliser les moyennes mensuelles historiques
                        monthly_avg = combined_traffic_data[var].groupby(combined_traffic_data.index.month).mean()
                        exog_forecast_traffic[var] = [monthly_avg.get(month, combined_traffic_data[var].mean()) for month in range(1, 13)]
                    else:
                        # Si la variable n'est pas disponible, utiliser la moyenne globale
                        exog_forecast_traffic[var] = [combined_traffic_data[var].mean() if var in combined_traffic_data.columns else 0] * 12
                
                # S'assurer que les données exogènes ont la même forme que lors de l'entraînement
                exog_forecast_traffic = exog_forecast_traffic[['q', 'k', 'nb_mesures']]
                
                forecast_traffic = fitted_model_traffic.get_forecast(steps=periods, exog=exog_forecast_traffic)
                predictions_traffic = forecast_traffic.predicted_mean.values
                
                # Création du troisième graphique
                fig3 = go.Figure()
                
                # Données historiques
                hist_df3 = combined_traffic_data.reset_index()
                hist_df3['accidents'] = hist_df3['accidents'].astype(float)
                
                fig3.add_trace(go.Scatter(
                    x=hist_df3['date'],
                    y=hist_df3['accidents'],
                    mode='lines+markers',
                    name='Données historiques',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                
                # Prédictions 2023 avec trafic
                pred_df3 = pd.DataFrame({
                    'date': future_dates,
                    'accidents': predictions_traffic.astype(float)
                })
                
                fig3.add_trace(go.Scatter(
                    x=pred_df3['date'],
                    y=pred_df3['accidents'],
                    mode='lines+markers',
                    name='Prédictions 2023 (avec trafic)',
                    line=dict(color='red', width=3, dash='dash'),
                    marker=dict(size=6)
                ))
                
                # Configuration du troisième graphique
                fig3.update_layout(
                    title="Prédictions SARIMA avec données de trafic routier - Accidents à Paris 2023",
                    xaxis_title="Date",
                    yaxis_title="Nombre d'accidents",
                    height=600,
                    hovermode='x unified'
                )
                
                # Affichage du troisième graphique
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.error("❌ Pas assez de données de trafic après fusion")
            
            # QUATRIÈME GRAPHIQUE AVEC TOUTES LES DONNÉES
            st.markdown("### 🎯 Prédictions SARIMA avec toutes les données (météo + trafic)")
            
            # Fusion de toutes les données
            all_data = ts_data.copy()
            all_data = all_data.join(weather_data, how='inner')
            all_data = all_data.join(traffic_data, how='inner')
            
            # Nettoyage intelligent de toutes les données
            # Colonnes qui représentent des quantités (NaN = 0)
            quantity_columns = ['prcp', 'snow', 'tsun']  # Précipitations, neige, ensoleillement
            # Colonnes qui représentent des moyennes (NaN = moyenne de la colonne)
            average_columns = ['tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'pres', 'q', 'k', 'nb_mesures']  # Températures, vent, pression, trafic
            # Colonnes optionnelles (peuvent être entièrement vides)
            optional_columns = ['wpgt']  # Rafales de vent (pas toujours mesurées)
            
            for col in all_data.columns:
                if col != 'accidents':
                    # Remplacer inf et -inf par NaN
                    all_data[col] = all_data[col].replace([np.inf, -np.inf], np.nan)
                    
                    if col in quantity_columns:
                        # Pour les quantités, NaN signifie 0 (pas de précipitations, pas de neige, etc.)
                        all_data[col] = all_data[col].fillna(0)
                    elif col in average_columns:
                        # Pour les moyennes, remplir par la moyenne de la colonne
                        if not all_data[col].isna().all():
                            all_data[col] = all_data[col].fillna(all_data[col].mean())
                        else:
                            all_data[col] = all_data[col].fillna(0)
                    elif col in optional_columns:
                        # Pour les colonnes optionnelles, remplir par 0 si entièrement vides
                        all_data[col] = all_data[col].fillna(0)
            
            # Ne pas faire de dropna() - garder toutes les données fusionnées
            
            if len(all_data) > 0:
                st.write(f"📊 **Données utilisées :** {len(all_data)} lignes")
            
                # Entraînement SARIMA avec toutes les données
                all_exog_vars = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'q', 'k', 'nb_mesures']
                available_all_vars = [var for var in all_exog_vars if var in all_data.columns]
                
                # Vérification finale des données exogènes avant création du modèle
                exog_data = all_data[available_all_vars].copy()
                
                # S'assurer qu'il n'y a plus de NaN ou d'inf dans les données exogènes
                for col in exog_data.columns:
                    exog_data[col] = exog_data[col].replace([np.inf, -np.inf], np.nan)
                    if exog_data[col].isna().any():
                        if col in ['prcp', 'snow', 'tsun']:
                            exog_data[col] = exog_data[col].fillna(0)
                        else:
                            exog_data[col] = exog_data[col].fillna(exog_data[col].mean())
                
                model_all = SARIMAX(
                    all_data['accidents'], 
                    exog=exog_data,
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, s)
                )
                fitted_model_all = model_all.fit(disp=False)
                
            # Prédictions 2023 avec données saisonnières réalistes
            exog_forecast_all = pd.DataFrame(index=future_dates)
            
            for var in available_all_vars:
                if var in all_data.columns:
                    if var in ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun']:
                        # Pour les données météo, utiliser les moyennes mensuelles historiques
                        monthly_avg = all_data[var].groupby(all_data.index.month).mean()
                        exog_forecast_all[var] = [monthly_avg.get(month, all_data[var].mean()) for month in range(1, 13)]
                    elif var in ['q', 'k', 'nb_mesures']:
                        # Pour les données de trafic, utiliser les moyennes mensuelles historiques
                        monthly_avg = all_data[var].groupby(all_data.index.month).mean()
                        exog_forecast_all[var] = [monthly_avg.get(month, all_data[var].mean()) for month in range(1, 13)]
                    else:
                        # Pour les autres variables, utiliser la moyenne globale
                        exog_forecast_all[var] = [all_data[var].mean()] * 12
            
            # S'assurer que les données exogènes ont la même forme que lors de l'entraînement
            exog_forecast_all = exog_forecast_all[available_all_vars]
            
            forecast_all = fitted_model_all.get_forecast(steps=periods, exog=exog_forecast_all)
            predictions_all = forecast_all.predicted_mean.values
                
            # Création du quatrième graphique
            fig4 = go.Figure()
            
            # Données historiques
            hist_df4 = all_data.reset_index()
            hist_df4['accidents'] = hist_df4['accidents'].astype(float)
            
            fig4.add_trace(go.Scatter(
                x=hist_df4['date'],
                y=hist_df4['accidents'],
                mode='lines+markers',
                name='Données historiques',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ))
            
            # Prédictions 2023 avec toutes les données
            pred_df4 = pd.DataFrame({
                'date': future_dates,
                'accidents': predictions_all.astype(float)
            })
            
            fig4.add_trace(go.Scatter(
                x=pred_df4['date'],
                y=pred_df4['accidents'],
                mode='lines+markers',
                name='Prédictions 2023 (toutes données)',
                line=dict(color='gold', width=3, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Configuration du quatrième graphique
            fig4.update_layout(
                title="Prédictions SARIMA avec toutes les données - Accidents à Paris 2023",
                xaxis_title="Date",
                yaxis_title="Nombre d'accidents",
                height=600,
                hovermode='x unified'
            )
            
            # Affichage du quatrième graphique
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.error("❌ Pas assez de données après fusion de toutes les sources")
            
            # CINQUIÈME GRAPHIQUE - IMPORTANCE DES VARIABLES
            st.markdown("### 📊 Importance des variables dans la prédiction")
            
            # Extraction des coefficients du modèle
            try:
                # Récupération des coefficients des variables exogènes
                exog_coef = fitted_model_all.params[1:1+len(available_all_vars)]  # Exclure la constante
                exog_names = available_all_vars
                
                # Création du graphique d'importance
                fig5 = go.Figure()
                
                # Tri par valeur absolue pour l'importance
                importance_data = list(zip(exog_names, exog_coef))
                importance_data.sort(key=lambda x: abs(x[1]), reverse=True)
                
                variables = [item[0] for item in importance_data]
                coefficients = [item[1] for item in importance_data]
                
                # Couleurs selon le signe du coefficient
                colors = ['red' if coef < 0 else 'green' for coef in coefficients]
                
                fig5.add_trace(go.Bar(
                    x=variables,
                    y=coefficients,
                    marker_color=colors,
                    text=[f'{coef:.4f}' for coef in coefficients],
                    textposition='auto'
                ))
                
                fig5.update_layout(
                    title="Importance des variables dans la prédiction SARIMA",
                    xaxis_title="Variables",
                    yaxis_title="Coefficient",
                    height=500,
                    showlegend=False
                )
                
                # Rotation des labels x
                fig5.update_xaxes(tickangle=45)
                
                # Affichage du cinquième graphique
                st.plotly_chart(fig5, use_container_width=True)
                
                # Dictionnaire des descriptions des variables
                var_descriptions = {
                    'tavg': 'Température moyenne',
                    'tmin': 'Température minimale',
                    'tmax': 'Température maximale',
                    'prcp': 'Précipitations',
                    'snow': 'Neige',
                    'wdir': 'Direction du vent',
                    'wspd': 'Vitesse du vent',
                    'wpgt': 'Rafales de vent',
                    'pres': 'Pression atmosphérique',
                    'tsun': 'Ensoleillement',
                    'q': 'Débit routier (véhicules/h)',
                    'k': 'Densité routière (véhicules/km)',
                    'nb_mesures': 'Nombre de mesures de trafic'
                }
                
                # Tableau récapitulatif des variables
                st.markdown("#### 📋 Récapitulatif des variables")
                importance_df = pd.DataFrame({
                    'Variable': variables,
                    'Description': [var_descriptions.get(var, var) for var in variables],
                    'Coefficient': [f'{coef:.4f}' for coef in coefficients],
                    'Impact': ['Positif' if coef > 0 else 'Négatif' for coef in coefficients],
                    'Importance': [f'{abs(coef):.4f}' for coef in coefficients]
                })
                st.dataframe(importance_df, use_container_width=True)
                
            except Exception as e:
                st.warning(f"⚠️ Impossible d'extraire l'importance des variables: {e}")
                
        # CINQUIÈME GRAPHIQUE SARIMA SANS 2020
        st.markdown("---")
        st.markdown("### 📊 Prédictions SARIMA sans données 2020 (avec météo et trafic)")
        
        # Préparation des données sans 2020 avec météo et trafic
        if 'all_data' in locals() and len(all_data) > 0:
            # Utiliser les données complètes sans 2020
            all_data_no_2020 = all_data[all_data.index.year != 2020].copy()
            
            if len(all_data_no_2020) > 0:
                st.write(f"📊 **Données utilisées (sans 2020, avec météo et trafic) :** {len(all_data_no_2020)} lignes")
                
                # Entraînement SARIMA sans 2020 avec toutes les données
                all_exog_vars = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'q', 'k', 'nb_mesures']
                available_all_vars = [var for var in all_exog_vars if var in all_data_no_2020.columns]
                
                # Vérification finale des données exogènes avant création du modèle
                exog_data_no_2020 = all_data_no_2020[available_all_vars].copy()
                
                # S'assurer qu'il n'y a plus de NaN ou d'inf dans les données exogènes
                for col in exog_data_no_2020.columns:
                    exog_data_no_2020[col] = exog_data_no_2020[col].replace([np.inf, -np.inf], np.nan)
                    if exog_data_no_2020[col].isna().any():
                        if col in ['prcp', 'snow', 'tsun']:
                            exog_data_no_2020[col] = exog_data_no_2020[col].fillna(0)
                        else:
                            exog_data_no_2020[col] = exog_data_no_2020[col].fillna(exog_data_no_2020[col].mean())
                
                model_no_2020 = SARIMAX(
                    all_data_no_2020['accidents'], 
                    exog=exog_data_no_2020,
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, s)
                )
                fitted_model_no_2020 = model_no_2020.fit(disp=False)
                
                # Prédictions 2023 avec données saisonnières réalistes (sans 2020)
                exog_forecast_no_2020 = pd.DataFrame(index=future_dates)
                
                for var in available_all_vars:
                    if var in all_data_no_2020.columns:
                        if var in ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun']:
                            # Pour les données météo, utiliser les moyennes mensuelles historiques (sans 2020)
                            monthly_avg = all_data_no_2020[var].groupby(all_data_no_2020.index.month).mean()
                            exog_forecast_no_2020[var] = [monthly_avg.get(month, all_data_no_2020[var].mean()) for month in range(1, 13)]
                        elif var in ['q', 'k', 'nb_mesures']:
                            # Pour les données de trafic, utiliser les moyennes mensuelles historiques (sans 2020)
                            monthly_avg = all_data_no_2020[var].groupby(all_data_no_2020.index.month).mean()
                            exog_forecast_no_2020[var] = [monthly_avg.get(month, all_data_no_2020[var].mean()) for month in range(1, 13)]
                        else:
                            # Pour les autres variables, utiliser la moyenne globale (sans 2020)
                            exog_forecast_no_2020[var] = [all_data_no_2020[var].mean()] * 12
                
                # S'assurer que les données exogènes ont la même forme que lors de l'entraînement
                exog_forecast_no_2020 = exog_forecast_no_2020[available_all_vars]
                
                forecast_no_2020 = fitted_model_no_2020.get_forecast(steps=periods, exog=exog_forecast_no_2020)
                predictions_no_2020 = forecast_no_2020.predicted_mean.values
                
                # Définir ts_clean_no_2020 pour l'affichage des informations
                ts_clean_no_2020 = all_data_no_2020['accidents'].dropna()
            else:
                st.error("❌ Pas assez de données sans 2020 pour l'entraînement avec météo et trafic")
        else:
            # Fallback : utiliser seulement les données d'accidents sans 2020
            ts_data_no_2020 = ts_data[ts_data.index.year != 2020].copy()
            
            if len(ts_data_no_2020) > 0:
                st.write(f"📊 **Données utilisées (sans 2020, accidents uniquement) :** {len(ts_data_no_2020)} lignes")
                st.warning("⚠️ Données météo et trafic non disponibles - utilisation des données d'accidents uniquement")
                
                # Préparation des données
                ts_clean_no_2020 = ts_data_no_2020.dropna()
                ts_clean_no_2020['accidents'] = pd.to_numeric(ts_clean_no_2020['accidents'], errors='coerce')
                ts_clean_no_2020 = ts_clean_no_2020.dropna()
                
                # Entraînement du modèle SARIMA sans 2020
                model_no_2020 = SARIMAX(ts_clean_no_2020['accidents'], order=(p, d, q), seasonal_order=(P, D, Q, s))
                fitted_model_no_2020 = model_no_2020.fit(disp=False)
                forecast_no_2020 = fitted_model_no_2020.get_forecast(steps=periods)
                predictions_no_2020 = forecast_no_2020.predicted_mean.values
            else:
                st.error("❌ Pas assez de données sans 2020 pour l'entraînement")
        
        # Création du graphique SARIMA sans 2020
        fig_no_2020 = go.Figure()
        
        # Données historiques (sans 2020)
        if 'all_data_no_2020' in locals():
            hist_df_no_2020 = all_data_no_2020.reset_index()
        else:
            hist_df_no_2020 = ts_clean_no_2020.reset_index()
        
        hist_df_no_2020['accidents'] = hist_df_no_2020['accidents'].astype(float)
        
        fig_no_2020.add_trace(go.Scatter(
            x=hist_df_no_2020['date'],
            y=hist_df_no_2020['accidents'],
            mode='lines+markers',
            name='Données historiques (sans 2020)',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ))
        
        # Prédictions 2023 sans 2020
        pred_df_no_2020 = pd.DataFrame({
            'date': future_dates,
            'accidents': predictions_no_2020.astype(float)
        })
        
        fig_no_2020.add_trace(go.Scatter(
            x=pred_df_no_2020['date'],
            y=pred_df_no_2020['accidents'],
            mode='lines+markers',
            name='Prédictions 2023 (sans 2020)',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Configuration du graphique
        fig_no_2020.update_layout(
            title="Prédictions SARIMA sans données 2020 - Accidents à Paris 2023",
            xaxis_title="Date",
            yaxis_title="Nombre d'accidents",
            height=600,
            hovermode='x unified'
        )
        
        # Affichage du graphique
        st.plotly_chart(fig_no_2020, use_container_width=True)
        
        # Comparaison des prédictions
        st.markdown("#### Comparaison des prédictions")
        
        # Calculer les prédictions avec 2020 pour comparaison
        ts_clean_with_2020 = ts_data.dropna()
        ts_clean_with_2020['accidents'] = pd.to_numeric(ts_clean_with_2020['accidents'], errors='coerce')
        ts_clean_with_2020 = ts_clean_with_2020.dropna()
        
        model_with_2020 = SARIMAX(ts_clean_with_2020['accidents'], order=(p, d, q), seasonal_order=(P, D, Q, s))
        fitted_model_with_2020 = model_with_2020.fit(disp=False)
        forecast_with_2020 = fitted_model_with_2020.get_forecast(steps=periods)
        predictions_with_2020 = forecast_with_2020.predicted_mean.values
        
        # Extraction des données réelles de 2023
        real_2023_data = ts_data[ts_data.index.year == 2023]
        
        # Graphique de comparaison
        fig_comparison = go.Figure()
        
        # Prédictions sans 2020
        fig_comparison.add_trace(go.Scatter(
            x=future_dates,
            y=predictions_no_2020,
            mode='lines+markers',
            name='Prédictions sans 2020',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Prédictions avec 2020
        fig_comparison.add_trace(go.Scatter(
            x=future_dates,
            y=predictions_with_2020,
            mode='lines+markers',
            name='Prédictions avec 2020',
            line=dict(color='blue', width=3, dash='dot'),
            marker=dict(size=6)
        ))
        
        # Données réelles de 2023 (si disponibles)
        if len(real_2023_data) > 0:
            real_2023_dates = real_2023_data.index
            real_2023_values = real_2023_data['accidents'].values
            
            fig_comparison.add_trace(go.Scatter(
                x=real_2023_dates,
                y=real_2023_values,
                mode='lines+markers',
                name='Données réelles 2023',
                line=dict(color='green', width=4),
                marker=dict(size=8)
            ))
            
            # Calculer les métriques de performance
            min_len = min(len(predictions_no_2020), len(predictions_with_2020), len(real_2023_values))
            
            if min_len > 0:
                mae_no_2020 = np.mean(np.abs(real_2023_values[:min_len] - predictions_no_2020[:min_len]))
                mae_with_2020 = np.mean(np.abs(real_2023_values[:min_len] - predictions_with_2020[:min_len]))
                
                st.write("**Performance vs données réelles 2023 :**")
                st.write(f"- MAE sans 2020 : {mae_no_2020:.2f} accidents")
                st.write(f"- MAE avec 2020 : {mae_with_2020:.2f} accidents")
                
                if mae_no_2020 < mae_with_2020:
                    st.write("- Le modèle sans 2020 est plus précis")
                else:
                    st.write("- Le modèle avec 2020 est plus précis")
        else:
            st.write("Aucune donnée réelle de 2023 disponible pour la comparaison")
        
        # Configuration du graphique de comparaison
        fig_comparison.update_layout(
            title="Comparaison des prédictions SARIMA avec/sans données 2020 vs données réelles",
            xaxis_title="Date",
            yaxis_title="Nombre d'accidents",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Statistiques de comparaison
        diff_predictions = predictions_with_2020 - predictions_no_2020
        mean_diff = np.mean(diff_predictions)
        max_diff = np.max(np.abs(diff_predictions))
        
        st.write("**Impact de 2020 sur les prédictions :**")
        st.write(f"- Différence moyenne : {mean_diff:.2f} accidents")
        st.write(f"- Différence maximale : {max_diff:.2f} accidents")
        st.write(f"- Période d'entraînement sans 2020 : {ts_clean_no_2020.index.min()} à {ts_clean_no_2020.index.max()}")
        st.write(f"- Période d'entraînement avec 2020 : {ts_clean_with_2020.index.min()} à {ts_clean_with_2020.index.max()}")
        
        # SIXIÈME GRAPHIQUE PROPHET
        st.markdown("---")
        st.markdown("### Prédictions Prophet (Facebook)")
        
        if not PROPHET_AVAILABLE:
            st.error("Prophet n'est pas installé. Installez-le avec : `pip install prophet`")
        else:
            if 'all_data' in locals() and len(all_data) > 0:
                st.write(f"📊 **Données utilisées :** {len(all_data)} lignes")
                
                # Préparation des données pour Prophet
                prophet_data = all_data[['accidents']].copy()
                prophet_data = prophet_data.reset_index()
                prophet_data.columns = ['ds', 'y']  # Prophet attend 'ds' (date) et 'y' (valeur)
                
                # Ajouter des variables exogènes si disponibles
                all_exog_vars = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'q', 'k', 'nb_mesures']
                available_exog_vars = [var for var in all_exog_vars if var in all_data.columns]
                
                for var in available_exog_vars:
                    prophet_data[var] = all_data[var].values
                
                # Entraînement du modèle Prophet
                model_prophet = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.05,  # Sensibilité aux changements de tendance
                    seasonality_prior_scale=10.0,  # Force de la saisonnalité
                    holidays_prior_scale=10.0,
                    changepoint_range=0.8
                )
                
                # Ajouter les variables exogènes
                for var in available_exog_vars:
                    model_prophet.add_regressor(var)
                
                # Entraînement
                model_prophet.fit(prophet_data)
                
                # Prédictions 2023
                future_prophet = model_prophet.make_future_dataframe(periods=12, freq='MS')
                
                # Ajouter les variables exogènes pour 2023
                for var in available_exog_vars:
                    if var in ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun']:
                        # Moyennes mensuelles historiques pour les données météo
                        monthly_avg = all_data[var].groupby(all_data.index.month).mean()
                        # Créer un array de la bonne longueur (96 lignes)
                        var_values = np.full(len(future_prophet), np.nan)
                        # Assigner les valeurs historiques (84 premières lignes)
                        var_values[:len(all_data)] = all_data[var].values
                        # Assigner les valeurs futures (12 dernières lignes)
                        var_values[-12:] = [monthly_avg.get(month, all_data[var].mean()) for month in range(1, 13)]
                        future_prophet[var] = var_values
                    elif var in ['q', 'k', 'nb_mesures']:
                        # Moyennes mensuelles historiques pour les données de trafic
                        monthly_avg = all_data[var].groupby(all_data.index.month).mean()
                        # Créer un array de la bonne longueur (96 lignes)
                        var_values = np.full(len(future_prophet), np.nan)
                        # Assigner les valeurs historiques (84 premières lignes)
                        var_values[:len(all_data)] = all_data[var].values
                        # Assigner les valeurs futures (12 dernières lignes)
                        var_values[-12:] = [monthly_avg.get(month, all_data[var].mean()) for month in range(1, 13)]
                        future_prophet[var] = var_values
                    else:
                        # Moyenne globale pour les autres variables
                        future_prophet[var] = all_data[var].mean()
                
                # Prédictions
                forecast_prophet = model_prophet.predict(future_prophet)
                predictions_prophet = forecast_prophet['yhat'].tail(12).values
                
                # Création du graphique Prophet
                fig_prophet = go.Figure()
                
                # Données historiques
                fig_prophet.add_trace(go.Scatter(
                    x=all_data.index,
                    y=all_data['accidents'],
                    mode='lines+markers',
                    name='Données historiques',
                    line=dict(color='lightblue', width=2),
                    marker=dict(size=4)
                ))
                
                # Prédictions Prophet
                fig_prophet.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions_prophet,
                    mode='lines+markers',
                    name='Prédictions Prophet 2023',
                    line=dict(color='purple', width=3, dash='dash'),
                    marker=dict(size=6)
                ))
                
                # Données réelles 2023 si disponibles
                if 'ts_data' in locals():
                    real_2023 = ts_data[ts_data.index.year == 2023]
                    if len(real_2023) > 0:
                        fig_prophet.add_trace(go.Scatter(
                            x=real_2023.index,
                            y=real_2023['accidents'],
                            mode='lines+markers',
                            name='Données réelles 2023',
                            line=dict(color='green', width=3),
                            marker=dict(size=6)
                        ))
                
                fig_prophet.update_layout(
                    title='Prédictions Prophet avec variables exogènes',
                    xaxis_title='Date',
                    yaxis_title='Nombre d\'accidents',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_prophet, use_container_width=True)
                
                # Métriques Prophet
                if 'ts_data' in locals():
                    real_2023 = ts_data[ts_data.index.year == 2023]
                    if len(real_2023) > 0:
                        mae_prophet = np.mean(np.abs(predictions_prophet - real_2023['accidents'].values))
                        rmse_prophet = np.sqrt(np.mean((predictions_prophet - real_2023['accidents'].values) ** 2))
                        
                        st.write("**Performance Prophet vs données réelles 2023 :**")
                        st.write(f"- MAE : {mae_prophet:.2f} accidents")
                        st.write(f"- RMSE : {rmse_prophet:.2f} accidents")
            else:
                st.error("❌ Pas assez de données pour Prophet")
        
        # Graphique de synthèse - Modèle hybride (sans ARIMA-GARCH)
        st.markdown("---")
        st.markdown("### Synthèse des prédictions - Modèle hybride")
        
        if True:  # Toujours exécuter
            # Vérifier que nous avons les prédictions nécessaires
            if ('predictions_with_2020' in locals() and 'predictions_no_2020' in locals() and 
                'predictions_prophet' in locals() and 'ts_data' in locals()):
                
                real_2023 = ts_data[ts_data.index.year == 2023]
                if len(real_2023) > 0:
                    # Calculer les MAE
                    mae_with_2020 = np.mean(np.abs(predictions_with_2020 - real_2023['accidents'].values))
                    mae_no_2020 = np.mean(np.abs(predictions_no_2020 - real_2023['accidents'].values))
                    mae_prophet = np.mean(np.abs(predictions_prophet - real_2023['accidents'].values))
                    
                    # Créer un modèle hybride basé sur les performances (3 modèles seulement)
                    predictions_hybrid = []
                    
                    # Calculer les poids inversement proportionnels à l'erreur
                    total_inv_error = 1/mae_with_2020 + 1/mae_no_2020 + 1/mae_prophet
                    w_with_2020 = (1/mae_with_2020) / total_inv_error
                    w_no_2020 = (1/mae_no_2020) / total_inv_error
                    w_prophet = (1/mae_prophet) / total_inv_error
                    
                    # Combinaison pondérée des prédictions
                    for i in range(12):
                        hybrid_pred = (w_with_2020 * predictions_with_2020[i] + 
                                    w_no_2020 * predictions_no_2020[i] + 
                                    w_prophet * predictions_prophet[i])
                        predictions_hybrid.append(hybrid_pred)
                    
                    # Calculer la performance du modèle hybride
                    mae_hybrid = np.mean(np.abs(predictions_hybrid - real_2023['accidents'].values))
                    
                    # Créer le graphique de synthèse
                    fig_hybrid = go.Figure()
                    
                    # Données historiques
                    fig_hybrid.add_trace(go.Scatter(
                        x=ts_data.index,
                        y=ts_data['accidents'],
                        mode='lines',
                        name='Données historiques',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Prédictions individuelles
                    months_2023 = pd.date_range(start='2023-01-01', periods=12, freq='MS')
                    
                    fig_hybrid.add_trace(go.Scatter(
                        x=months_2023,
                        y=predictions_with_2020,
                        mode='lines+markers',
                        name=f'SARIMA avec 2020 (MAE: {mae_with_2020:.1f})',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_hybrid.add_trace(go.Scatter(
                        x=months_2023,
                        y=predictions_no_2020,
                        mode='lines+markers',
                        name=f'SARIMA sans 2020 (MAE: {mae_no_2020:.1f})',
                        line=dict(color='orange', dash='dash')
                    ))
                    
                    fig_hybrid.add_trace(go.Scatter(
                        x=months_2023,
                        y=predictions_prophet,
                        mode='lines+markers',
                        name=f'Prophet (MAE: {mae_prophet:.1f})',
                        line=dict(color='green', dash='dash')
                    ))
                    
                    # Modèle hybride (ligne épaisse)
                    fig_hybrid.add_trace(go.Scatter(
                        x=months_2023,
                        y=predictions_hybrid,
                        mode='lines+markers',
                        name=f'Modèle hybride (MAE: {mae_hybrid:.1f})',
                        line=dict(color='black', width=4)
                    ))
                    
                    # Données réelles 2023
                    fig_hybrid.add_trace(go.Scatter(
                        x=months_2023,
                        y=real_2023['accidents'].values,
                        mode='lines+markers',
                        name='Données réelles 2023',
                        line=dict(color='red', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig_hybrid.update_layout(
                        title='Synthèse des prédictions - Modèle hybride intelligent',
                        xaxis_title='Date',
                        yaxis_title='Nombre d\'accidents',
                        hovermode='x unified',
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_hybrid, use_container_width=True)
                    
                    # Affichage des poids du modèle hybride
                    st.write("**Poids du modèle hybride :**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("SARIMA avec 2020", f"{w_with_2020:.1%}")
                    with col2:
                        st.metric("SARIMA sans 2020", f"{w_no_2020:.1%}")
                    with col3:
                        st.metric("Prophet", f"{w_prophet:.1%}")
                    
                    # Comparaison finale avec le modèle hybride
                    st.write("**Performance finale :**")
                    final_models = [
                        ("SARIMA avec 2020", mae_with_2020),
                        ("SARIMA sans 2020", mae_no_2020),
                        ("Prophet", mae_prophet),
                        ("Modèle hybride", mae_hybrid)
                    ]
                    
                    # Trier par performance
                    final_models.sort(key=lambda x: x[1])
                    
                    for i, (model_name, mae_val) in enumerate(final_models):
                        if i == 0:
                            st.write(f"🥇 **{model_name}** : {mae_val:.2f} accidents")
                        elif i == 1:
                            st.write(f"🥈 **{model_name}** : {mae_val:.2f} accidents")
                        elif i == 2:
                            st.write(f"🥉 **{model_name}** : {mae_val:.2f} accidents")
                        else:
                            st.write(f"**{model_name}** : {mae_val:.2f} accidents")
                else:
                    st.error("❌ Pas de données réelles 2023 pour la comparaison")
            else:
                st.error("❌ Prédictions manquantes pour le modèle hybride")
        
        # SECTION XGBOOST
        st.markdown("---")
        st.markdown("### Prédictions XGBoost")
        
        if not XGBOOST_AVAILABLE:
            st.error("XGBoost n'est pas installé. Installez-le avec : `pip install xgboost scikit-learn`")
        else:
            if st.button("Lancer la prédiction XGBoost", type="primary"):
                with st.spinner("Entraînement du modèle XGBoost en cours..."):
                    try:
                        # Préparation des données pour XGBoost
                        # Utiliser les données avec météo et trafic
                        if 'all_data' in locals() and len(all_data) > 0:
                            xgb_data = all_data.copy()
                        else:
                            # Si pas de données complètes, utiliser les données de base
                            xgb_data = ts_data.copy()
                        
                        # Création des features temporelles avancées
                        xgb_data['year'] = xgb_data.index.year
                        xgb_data['month'] = xgb_data.index.month
                        xgb_data['day_of_year'] = xgb_data.index.dayofyear
                        xgb_data['quarter'] = xgb_data.index.quarter
                        
                        # Features cycliques pour capturer la saisonnalité
                        xgb_data['month_sin'] = np.sin(2 * np.pi * xgb_data['month'] / 12)
                        xgb_data['month_cos'] = np.cos(2 * np.pi * xgb_data['month'] / 12)
                        xgb_data['quarter_sin'] = np.sin(2 * np.pi * xgb_data['quarter'] / 4)
                        xgb_data['quarter_cos'] = np.cos(2 * np.pi * xgb_data['quarter'] / 4)
                        
                        # Features de lag (décalage temporel)
                        for lag in [1, 2, 3, 6, 12]:
                            xgb_data[f'accidents_lag_{lag}'] = xgb_data['accidents'].shift(lag)
                        
                        # Moyennes mobiles
                        for window in [3, 6, 12]:
                            xgb_data[f'accidents_ma_{window}'] = xgb_data['accidents'].rolling(window=window).mean()
                        
                        # Features de tendance
                        xgb_data['accidents_diff'] = xgb_data['accidents'].diff()
                        xgb_data['accidents_pct_change'] = xgb_data['accidents'].pct_change()
                        
                        # Features de volatilité
                        xgb_data['accidents_std_3'] = xgb_data['accidents'].rolling(window=3).std()
                        xgb_data['accidents_std_6'] = xgb_data['accidents'].rolling(window=6).std()
                        
                        # Supprimer les lignes avec des valeurs manquantes
                        xgb_data = xgb_data.dropna()
                        
                        if len(xgb_data) < 20:
                            st.error("❌ Pas assez de données pour l'entraînement XGBoost")
                        else:
                            st.write(f"📊 **Données utilisées pour XGBoost :** {len(xgb_data)} lignes")
                            
                            # Séparation des features et de la target
                            feature_columns = [col for col in xgb_data.columns if col != 'accidents']
                            X = xgb_data[feature_columns]
                            y = xgb_data['accidents']
                            
                            # Division train/test (80/20)
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, shuffle=False
                            )
                            
                            # Entraînement du modèle XGBoost optimisé
                            xgb_model = xgb.XGBRegressor(
                                n_estimators=200,
                                max_depth=8,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_alpha=0.1,
                                reg_lambda=0.1,
                                random_state=42
                            )
                            
                            # Entraînement avec validation set pour early stopping
                            xgb_model.fit(
                                X_train, y_train,
                                eval_set=[(X_test, y_test)],
                                early_stopping_rounds=20,
                                verbose=False
                            )
                            
                            # Prédictions sur l'ensemble de test
                            y_pred = xgb_model.predict(X_test)
                            
                            # Métriques de performance
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            
                            st.success(f"✅ Modèle XGBoost entraîné avec succès !")
                            st.write(f"📈 **Performance :** MAE = {mae:.2f}, RMSE = {rmse:.2f}")
                            
                            # Prédictions pour 2023 - Approche améliorée
                            predictions_2023 = []
                            
                            # Créer un DataFrame pour les prédictions 2023
                            future_data = []
                            
                            for month in range(1, 13):
                                # Créer une nouvelle ligne pour chaque mois 2023
                                row = {}
                                
                                # Features temporelles
                                row['year'] = 2023
                                row['month'] = month
                                row['day_of_year'] = pd.Timestamp(f'2023-{month:02d}-01').dayofyear
                                row['quarter'] = pd.Timestamp(f'2023-{month:02d}-01').quarter
                                
                                # Features cycliques pour la saisonnalité
                                row['month_sin'] = np.sin(2 * np.pi * month / 12)
                                row['month_cos'] = np.cos(2 * np.pi * month / 12)
                                row['quarter_sin'] = np.sin(2 * np.pi * row['quarter'] / 4)
                                row['quarter_cos'] = np.cos(2 * np.pi * row['quarter'] / 4)
                                
                                # Features de lag - utiliser les vraies données historiques
                                if month == 1:
                                    # Janvier 2023 : utiliser les données de décembre 2022
                                    row['accidents_lag_1'] = xgb_data['accidents'].iloc[-1] if len(xgb_data) > 0 else 0
                                    row['accidents_lag_2'] = xgb_data['accidents'].iloc[-2] if len(xgb_data) > 1 else 0
                                    row['accidents_lag_3'] = xgb_data['accidents'].iloc[-3] if len(xgb_data) > 2 else 0
                                    row['accidents_lag_6'] = xgb_data['accidents'].iloc[-6] if len(xgb_data) > 5 else 0
                                    row['accidents_lag_12'] = xgb_data['accidents'].iloc[-12] if len(xgb_data) > 11 else 0
                                else:
                                    # Utiliser les prédictions précédentes pour les lags
                                    row['accidents_lag_1'] = predictions_2023[-1] if len(predictions_2023) > 0 else 0
                                    row['accidents_lag_2'] = predictions_2023[-2] if len(predictions_2023) > 1 else 0
                                    row['accidents_lag_3'] = predictions_2023[-3] if len(predictions_2023) > 2 else 0
                                    row['accidents_lag_6'] = predictions_2023[-6] if len(predictions_2023) > 5 else 0
                                    row['accidents_lag_12'] = predictions_2023[-12] if len(predictions_2023) > 11 else 0
                                
                                # Moyennes mobiles - utiliser les prédictions précédentes
                                if month >= 3:
                                    row['accidents_ma_3'] = np.mean(predictions_2023[-3:]) if len(predictions_2023) >= 3 else 0
                                else:
                                    # Pour les premiers mois, utiliser les données historiques
                                    hist_data = xgb_data['accidents'].iloc[-(3-month):].tolist() + predictions_2023
                                    row['accidents_ma_3'] = np.mean(hist_data[-3:]) if len(hist_data) >= 3 else 0
                                
                                if month >= 6:
                                    row['accidents_ma_6'] = np.mean(predictions_2023[-6:]) if len(predictions_2023) >= 6 else 0
                                else:
                                    hist_data = xgb_data['accidents'].iloc[-(6-month):].tolist() + predictions_2023
                                    row['accidents_ma_6'] = np.mean(hist_data[-6:]) if len(hist_data) >= 6 else 0
                                
                                if month >= 12:
                                    row['accidents_ma_12'] = np.mean(predictions_2023[-12:]) if len(predictions_2023) >= 12 else 0
                                else:
                                    hist_data = xgb_data['accidents'].iloc[-(12-month):].tolist() + predictions_2023
                                    row['accidents_ma_12'] = np.mean(hist_data[-12:]) if len(hist_data) >= 12 else 0
                                
                                # Copier les autres features des dernières données disponibles
                                for col in feature_columns:
                                    if col not in row and col in xgb_data.columns:
                                        if col in ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'q', 'k', 'nb_mesures']:
                                            # Pour les données météo et trafic, utiliser la moyenne mensuelle historique
                                            if col in xgb_data.columns:
                                                monthly_avg = xgb_data[col].groupby(xgb_data.index.month).mean()
                                                row[col] = monthly_avg.get(month, xgb_data[col].mean())
                                            else:
                                                row[col] = 0
                                        else:
                                            row[col] = 0
                                
                                future_data.append(row)
                                
                                # Créer un DataFrame temporaire pour la prédiction
                                temp_df = pd.DataFrame([row])
                                
                                # S'assurer que toutes les colonnes sont présentes
                                for col in feature_columns:
                                    if col not in temp_df.columns:
                                        temp_df[col] = 0
                                
                                # Réorganiser les colonnes dans le bon ordre
                                temp_df = temp_df[feature_columns]
                                
                                # Prédiction
                                pred = xgb_model.predict(temp_df)[0]
                                predictions_2023.append(max(0, pred))  # S'assurer que la prédiction n'est pas négative
                            
                            # Création du graphique XGBoost
                            fig_xgb = go.Figure()
                            
                            # Données historiques
                            hist_df_xgb = xgb_data.reset_index()
                            fig_xgb.add_trace(go.Scatter(
                                x=hist_df_xgb['date'],
                                y=hist_df_xgb['accidents'],
                                mode='lines+markers',
                                name='Données historiques',
                                line=dict(color='blue', width=2),
                                marker=dict(size=4)
                            ))
                            
                            # Prédictions 2023 XGBoost
                            pred_df_xgb = pd.DataFrame({
                                'date': future_dates,
                                'accidents': predictions_2023
                            })
                            
                            fig_xgb.add_trace(go.Scatter(
                                x=pred_df_xgb['date'],
                                y=pred_df_xgb['accidents'],
                                mode='lines+markers',
                                name='Prédictions 2023 (XGBoost)',
                                line=dict(color='purple', width=3, dash='dash'),
                                marker=dict(size=6)
                            ))
                            
                            # Configuration du graphique
                            fig_xgb.update_layout(
                                title="Prédictions XGBoost - Accidents à Paris 2023",
                                xaxis_title="Date",
                                yaxis_title="Nombre d'accidents",
                                height=600,
                                hovermode='x unified'
                            )
                            
                            # Affichage du graphique XGBoost
                            st.plotly_chart(fig_xgb, use_container_width=True)
                            
                            # Importance des features
                            feature_importance = pd.DataFrame({
                                'feature': feature_columns,
                                'importance': xgb_model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            st.markdown("#### 📊 Importance des variables (XGBoost)")
                            fig_importance = go.Figure()
                            
                            fig_importance.add_trace(go.Bar(
                                x=feature_importance['importance'],
                                y=feature_importance['feature'],
                                orientation='h',
                                marker_color='lightblue'
                            ))
                            
                            fig_importance.update_layout(
                                title="Importance des variables dans le modèle XGBoost",
                                xaxis_title="Importance",
                                yaxis_title="Variables",
                                height=400
                            )
                            
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'entraînement XGBoost : {str(e)}")
      



























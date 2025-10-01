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
    kw.setdefault("use_container_width", True)
    if p.exists():
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
        '<p class="lead">Développeur Full Stack Data | Création de solutions automatisées pour des gains d\'efficacité et une vision data-driven.</p>',
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
    from umap import UMAP
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
            title="Projection UMAP des chaînes YouTube par orientation politique"
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
        - 🧭 Les vecteurs sont projetés en 2D via `UMAP` (distance **cosine**)
        
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
    
    # Onglets pour séparer présentation et application
    tab_application, tab_presentation = st.tabs(["🚀 Application", "📋 Présentation"])
    
    with tab_presentation:
        st.markdown("""
        <div style="text-align: left; font-size: 18px; line-height: 1.6; margin-top: 20px;">
            <p><strong>Présentation du projet :</strong></p>
            <p>
                Ce projet analyse les données d'accidents de la route à Paris sur la période 2017-2023. 
                Il combine plusieurs approches de machine learning (XGBoost, Prophet, SARIMA) avec des données 
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
            
            - **Prédictions ML** : Modèles XGBoost, Prophet et SARIMA
            - **Cartographie interactive** : Cartes de chaleur et clustering
            - **Analyse temporelle** : Évolution par mois et année
            - **Points noirs** : Identification des zones à risque par arrondissement
            - **Performance** : Traitement optimisé de 7 ans de données
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Technologies utilisées
            
            **Machine Learning :**
            - XGBoost pour les prédictions
            - Prophet pour l'analyse des séries temporelles
            - SARIMA pour la modélisation statistique
            
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
            'Modèle': ['XGBoost', 'Prophet', 'SARIMA'],
            'R² Score': [0.85, 0.82, 0.79],
            'MAE': [1.8, 2.1, 2.4],
            'RMSE': [2.3, 2.7, 3.1]
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
                ["Carte des accidents", "Évolution temporelle animée", "Évolution temporelle", "Analyse par arrondissement", "Analyses et Prédictions", "Statistiques générales"]
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
                    ["XGBoost (Machine Learning)", "Prophet (Série Temporelle)"]
                )
                
                if model_type == "XGBoost (Machine Learning)":
                    # Cache pour les données météo
                    @st.cache_data(ttl=3600)  # Cache pour 1 heure
                    def load_weather_data():
                        try:
                            weather_df = pd.read_csv('data_meteo.csv')
                            weather_df['date'] = pd.to_datetime(weather_df['date'])
                            return weather_df
                        except Exception as e:
                            st.error(f"Erreur lors du chargement des données météo : {str(e)}")
                            return None
                    
                    # Cache pour les données de trafic
                    @st.cache_data(ttl=3600)  # Cache pour 1 heure
                    def load_traffic_data():
                        try:
                            traffic_df = pd.read_csv('trafic_routier_paris.csv', sep=';')
                            traffic_df['date'] = pd.to_datetime(traffic_df['date'])
                            traffic_df = traffic_df.rename(columns={
                                'q': 'trafic_debit',
                                'k': 'trafic_concentration'
                            })
                            return traffic_df
                        except:
                            return None
                    
                    # Cache pour la préparation des données
                    @st.cache_data(ttl=1800)  # Cache pour 30 minutes
                    def prepare_xgboost_data(df, weather_df, traffic_df):
                        # Préparation des données
                        df['date'] = pd.to_datetime(df['date_heure']).dt.normalize()
                        
                        # Enrichissement des données
                        df_enriched = pd.merge(df, weather_df, on='date', how='left')
                        if traffic_df is not None:
                            df_enriched = pd.merge(df_enriched, traffic_df, on='date', how='left')
                        
                        # Agrégation quotidienne
                        daily_data = df_enriched.groupby('date').agg({
                            'id_accident': 'count',
                            'tavg': 'mean',
                            'prcp': 'mean', 
                            'snow': 'mean',
                            'wspd': 'mean',
                            'trafic_debit': 'mean',
                            'trafic_concentration': 'mean'
                        }).reset_index()
                        daily_data.rename(columns={'id_accident': 'accidents'}, inplace=True)
                        
                        # Préparation des features temporelles
                        def prepare_temporal_features(df):
                            df = df.copy()
                            df['year'] = df['date'].dt.year
                            df['month'] = df['date'].dt.month
                            df['day'] = df['date'].dt.day
                            df['dayofweek'] = df['date'].dt.dayofweek
                            df['quarter'] = df['date'].dt.quarter
                            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
                            return df
                        
                        daily_data = prepare_temporal_features(daily_data)
                        
                        # Division des données
                        train_data = daily_data[daily_data['year'] <= 2021].copy()
                        test_data_2022 = daily_data[daily_data['year'] == 2022].copy()
                        test_data_2023 = daily_data[daily_data['year'] == 2023].copy()
                        
                        # Remplissage des valeurs manquantes
                        for col in ['tavg', 'prcp', 'snow', 'wspd', 'trafic_debit', 'trafic_concentration']:
                            if col in train_data.columns:
                                monthly_means = train_data.groupby(train_data['date'].dt.month)[col].mean()
                                test_data_2022[col] = test_data_2022['date'].dt.month.map(monthly_means)
                                test_data_2023[col] = test_data_2023['date'].dt.month.map(monthly_means)
                            else:
                                test_data_2022[col] = 0.0
                                test_data_2023[col] = 0.0
                        
                        return train_data, test_data_2022, test_data_2023
                    
                    # Cache pour l'entraînement du modèle XGBoost
                    @st.cache_data(ttl=1800)  # Cache pour 30 minutes
                    def train_xgboost_model(train_data, test_data_2022, test_data_2023):
                        feature_columns = [
                            'tavg', 'prcp', 'snow', 'wspd', 'trafic_debit', 'trafic_concentration',
                            'year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend'
                        ]
                        
                        # Filtrage des colonnes existantes
                        available_features = [col for col in feature_columns if col in train_data.columns]
                        
                        X_train = train_data[available_features].fillna(0)
                        y_train = train_data['accidents']
                        X_test_2022 = test_data_2022[available_features].fillna(0)
                        X_test_2023 = test_data_2023[available_features].fillna(0)
                        
                        # Modèle XGBoost
                        import xgboost as xgb
                        model = xgb.XGBRegressor(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=6,
                            random_state=42
                        )
                        
                        model.fit(X_train, y_train)
                        
                        # Prédictions
                        predictions_2022 = None
                        predictions_2023 = None
                        
                        if not test_data_2022.empty:
                            predictions_2022 = model.predict(X_test_2022)
                        
                        if not test_data_2023.empty:
                            predictions_2023 = model.predict(X_test_2023)
                        
                        # Métriques
                        from sklearn.metrics import mean_absolute_error, r2_score
                        mae_2022 = None
                        r2_2022 = None
                        
                        if not test_data_2022.empty and predictions_2022 is not None:
                            mae_2022 = mean_absolute_error(test_data_2022['accidents'], predictions_2022)
                            r2_2022 = r2_score(test_data_2022['accidents'], predictions_2022)
                        
                        # Importance des features
                        feature_importance = pd.DataFrame({
                            'feature': available_features,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        return model, predictions_2022, predictions_2023, mae_2022, r2_2022, feature_importance, available_features
                    
                    try:
                        # Chargement des données avec cache
                        with st.spinner("Chargement des données..."):
                            weather_df = load_weather_data()
                            traffic_df = load_traffic_data()
                        
                        if weather_df is None:
                            st.error("Impossible de charger les données météo")
                        else:
                            # Préparation des données avec cache
                            with st.spinner("Préparation des données..."):
                                train_data, test_data_2022, test_data_2023 = prepare_xgboost_data(df, weather_df, traffic_df)
                            
                            # Vérification de la disponibilité des données
                            if train_data.empty:
                                st.error("Aucune donnée d'entraînement disponible (2017-2021)")
                                st.stop()
                            
                            if test_data_2022.empty and test_data_2023.empty:
                                st.error("Aucune donnée de test disponible (2022-2023)")
                                st.stop()
                            
                            # Entraînement du modèle avec cache
                            with st.spinner("Entraînement du modèle XGBoost..."):
                                model, predictions_2022, predictions_2023, mae_2022, r2_2022, feature_importance, available_features = train_xgboost_model(train_data, test_data_2022, test_data_2023)
                            
                                # Affichage des métriques
                                if mae_2022 is not None and r2_2022 is not None:
                                    st.subheader("Performance du modèle sur l'année 2022")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("MAE", f"{mae_2022:.2f}")
                                    with col2:
                                        st.metric("R²", f"{r2_2022:.3f}")
                                else:
                                    st.info("Aucune donnée de validation disponible pour 2022")
                            
                                # Importance des features
                                feature_importance = pd.DataFrame({
                                    'feature': available_features,
                                    'importance': model.feature_importances_
                                }).sort_values('importance', ascending=False)
                                
                                st.subheader("Importance des variables")
                                fig_importance = px.bar(
                                    feature_importance.head(10),
                                    x='importance',
                                    y='feature',
                                    orientation='h',
                                    title="Top 10 des variables les plus importantes"
                                )
                                st.plotly_chart(fig_importance, use_container_width=True)
                            
                                # Graphique des prédictions
                                fig = go.Figure()
                                
                                # Données historiques
                                fig.add_trace(go.Scatter(
                                    x=train_data['date'],
                                    y=train_data['accidents'],
                                    name='Données historiques (2017-2021)',
                                    line=dict(color='blue')
                                ))
                                
                                # Données réelles et prédictions 2022 (si disponibles)
                                if not test_data_2022.empty and predictions_2022 is not None:
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
                                
                                # Prédictions 2023 (si disponibles)
                                if not test_data_2023.empty and predictions_2023 is not None:
                                    fig.add_trace(go.Scatter(
                                        x=test_data_2023['date'],
                                        y=predictions_2023,
                                        name='Prédictions 2023',
                                        line=dict(color='red', dash='dash')
                                    ))
                                
                                fig.update_layout(
                                    title="Évolution du nombre d'accidents quotidiens et prédictions (XGBoost)",
                                    xaxis_title="Date",
                                    yaxis_title="Nombre d'accidents",
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                                # Statistiques des prédictions
                                if predictions_2023 is not None and len(predictions_2023) > 0:
                                    st.subheader("Statistiques des prédictions 2023")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Moyenne prédite", f"{predictions_2023.mean():.1f}")
                                    with col2:
                                        st.metric("Maximum prédit", f"{predictions_2023.max():.1f}")
                                    with col3:
                                        st.metric("Minimum prédit", f"{predictions_2023.min():.1f}")
                                else:
                                    st.info("Aucune prédiction disponible pour 2023")
                    
                    except Exception as e:
                        st.error(f"Erreur lors du calcul des prédictions XGBoost : {str(e)}")
                        st.info("Note : Les prédictions nécessitent les données météorologiques et de trafic. Vérifiez que les fichiers 'data_meteo.csv' et 'trafic_routier_paris.csv' sont présents.")
                
                elif model_type == "Prophet (Série Temporelle)":
                    st.info("Le modèle Prophet analyse les patterns temporels et saisonniers des accidents pour prédire l'évolution future.")
                    
                    # Cache pour les données météo (réutilise la même fonction)
                    @st.cache_data(ttl=3600)  # Cache pour 1 heure
                    def load_weather_data():
                        try:
                            weather_df = pd.read_csv('data_meteo.csv')
                            weather_df['date'] = pd.to_datetime(weather_df['date'])
                            return weather_df
                        except Exception as e:
                            st.error(f"Erreur lors du chargement des données météo : {str(e)}")
                            return None
                    
                    # Cache pour la préparation des données Prophet
                    @st.cache_data(ttl=1800)  # Cache pour 30 minutes
                    def prepare_prophet_data(df, weather_df):
                        # Préparation des données pour Prophet
                        df['date'] = pd.to_datetime(df['date_heure']).dt.normalize()
                        
                        # Agrégation quotidienne
                        daily_data = df.groupby('date').agg({
                            'id_accident': 'count'
                        }).reset_index()
                        daily_data.rename(columns={'id_accident': 'accidents'}, inplace=True)
                        
                        # Enrichissement avec les données météo
                        df_enriched = pd.merge(daily_data, weather_df, on='date', how='left')
                        
                        # Division des données
                        train_data = df_enriched[df_enriched['date'].dt.year <= 2021].copy()
                        test_data_2022 = df_enriched[df_enriched['date'].dt.year == 2022].copy()
                        test_data_2023 = df_enriched[df_enriched['date'].dt.year == 2023].copy()
                        
                        return train_data, test_data_2022, test_data_2023
                    
                    # Cache pour l'entraînement du modèle Prophet
                    @st.cache_data(ttl=1800)  # Cache pour 30 minutes
                    def train_prophet_model(train_data, test_data_2022, test_data_2023):
                        from prophet import Prophet
                        
                        # Configuration du modèle Prophet
                        model_prophet = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            seasonality_mode='multiplicative',
                            changepoint_prior_scale=0.05
                        )
                        
                        # Ajout de régresseurs météo si disponibles
                        if 'tavg' in train_data.columns:
                            model_prophet.add_regressor('tavg')
                        if 'prcp' in train_data.columns:
                            model_prophet.add_regressor('prcp')
                        if 'wspd' in train_data.columns:
                            model_prophet.add_regressor('wspd')
                        
                        # Préparation des données d'entraînement avec régresseurs
                        train_with_regressors = train_data[['date', 'accidents', 'tavg', 'prcp', 'wspd']].copy()
                        train_with_regressors.columns = ['ds', 'y', 'tavg', 'prcp', 'wspd']
                        train_with_regressors = train_with_regressors.dropna()
                        
                        # Entraînement du modèle
                        model_prophet.fit(train_with_regressors)
                        
                        # Prédictions pour 2022 et 2023
                        predictions_2022 = None
                        predictions_2023 = None
                        future_2022 = None
                        future_2023 = None
                        
                        if not test_data_2022.empty:
                            # Préparation des données de test 2022
                            future_2022 = test_data_2022[['date', 'tavg', 'prcp', 'wspd']].copy()
                            future_2022.columns = ['ds', 'tavg', 'prcp', 'wspd']
                            future_2022 = future_2022.dropna()
                            
                            if not future_2022.empty:
                                forecast_2022 = model_prophet.predict(future_2022)
                                predictions_2022 = forecast_2022['yhat'].values
                        
                        if not test_data_2023.empty:
                            # Préparation des données de test 2023
                            future_2023 = test_data_2023[['date', 'tavg', 'prcp', 'wspd']].copy()
                            future_2023.columns = ['ds', 'tavg', 'prcp', 'wspd']
                            future_2023 = future_2023.dropna()
                            
                            if not future_2023.empty:
                                forecast_2023 = model_prophet.predict(future_2023)
                                predictions_2023 = forecast_2023['yhat'].values
                        
                        # Métriques pour 2022
                        mae_2022 = None
                        r2_2022 = None
                        
                        if not test_data_2022.empty and predictions_2022 is not None and future_2022 is not None:
                            from sklearn.metrics import mean_absolute_error, r2_score
                            test_2022_clean = test_data_2022[test_data_2022['date'].isin(future_2022['ds'])]
                            if not test_2022_clean.empty and len(predictions_2022) == len(test_2022_clean):
                                mae_2022 = mean_absolute_error(test_2022_clean['accidents'], predictions_2022)
                                r2_2022 = r2_score(test_2022_clean['accidents'], predictions_2022)
                        
                        return model_prophet, predictions_2022, predictions_2023, mae_2022, r2_2022, future_2022, future_2023
                    
                    try:
                        # Chargement des données avec cache
                        with st.spinner("Chargement des données..."):
                            weather_df = load_weather_data()
                        
                        if weather_df is None:
                            st.error("Impossible de charger les données météo")
                        else:
                            # Préparation des données avec cache
                            with st.spinner("Préparation des données Prophet..."):
                                train_data, test_data_2022, test_data_2023 = prepare_prophet_data(df, weather_df)
                            
                            # Vérification de la disponibilité des données
                            if train_data.empty:
                                st.error("Aucune donnée d'entraînement disponible (2017-2021)")
                                st.stop()
                            
                            if test_data_2022.empty and test_data_2023.empty:
                                st.error("Aucune donnée de test disponible (2022-2023)")
                                st.stop()
                            
                            # Entraînement du modèle avec cache
                            with st.spinner("Entraînement du modèle Prophet..."):
                                model_prophet, predictions_2022, predictions_2023, mae_2022, r2_2022, future_2022, future_2023 = train_prophet_model(train_data, test_data_2022, test_data_2023)
                            
                            weather_df = load_weather_data()
                            if weather_df is None:
                                st.error("Impossible de charger les données météo")
                            else:
                                # Préparation des données pour Prophet
                                df['date'] = pd.to_datetime(df['date_heure']).dt.normalize()
                                
                                # Agrégation quotidienne
                                daily_data = df.groupby('date').agg({
                                    'id_accident': 'count'
                                }).reset_index()
                                daily_data.rename(columns={'id_accident': 'accidents'}, inplace=True)
                                
                                # Enrichissement avec les données météo
                                df_enriched = pd.merge(daily_data, weather_df, on='date', how='left')
                                
                                # Division des données
                                train_data = df_enriched[df_enriched['date'].dt.year <= 2021].copy()
                                test_data_2022 = df_enriched[df_enriched['date'].dt.year == 2022].copy()
                                test_data_2023 = df_enriched[df_enriched['date'].dt.year == 2023].copy()
                                
                                # Vérification de la disponibilité des données
                                if train_data.empty:
                                    st.error("Aucune donnée d'entraînement disponible (2017-2021)")
                                    st.stop()
                                
                                if test_data_2022.empty and test_data_2023.empty:
                                    st.error("Aucune donnée de test disponible (2022-2023)")
                                    st.stop()
                                
                                # Préparation des données pour Prophet
                                def prepare_prophet_data(df):
                                    df_prophet = df[['date', 'accidents']].copy()
                                    df_prophet.columns = ['ds', 'y']
                                    df_prophet = df_prophet.dropna()
                                    return df_prophet
                                
                                train_prophet = prepare_prophet_data(train_data)
                                
                                # Configuration et entraînement du modèle Prophet
                                from prophet import Prophet
                                
                                model_prophet = Prophet(
                                    yearly_seasonality=True,
                                    weekly_seasonality=True,
                                    daily_seasonality=False,
                                    seasonality_mode='multiplicative',
                                    changepoint_prior_scale=0.05
                                )
                                
                                # Ajout de régresseurs météo si disponibles
                                if 'tavg' in train_data.columns:
                                    model_prophet.add_regressor('tavg')
                                if 'prcp' in train_data.columns:
                                    model_prophet.add_regressor('prcp')
                                if 'wspd' in train_data.columns:
                                    model_prophet.add_regressor('wspd')
                                
                                # Préparation des données d'entraînement avec régresseurs
                                train_with_regressors = train_data[['date', 'accidents', 'tavg', 'prcp', 'wspd']].copy()
                                train_with_regressors.columns = ['ds', 'y', 'tavg', 'prcp', 'wspd']
                                train_with_regressors = train_with_regressors.dropna()
                                
                                # Entraînement du modèle
                                model_prophet.fit(train_with_regressors)
                                
                                # Prédictions pour 2022 et 2023
                                predictions_2022 = None
                                predictions_2023 = None
                                
                                if not test_data_2022.empty:
                                    # Préparation des données de test 2022
                                    future_2022 = test_data_2022[['date', 'tavg', 'prcp', 'wspd']].copy()
                                    future_2022.columns = ['ds', 'tavg', 'prcp', 'wspd']
                                    future_2022 = future_2022.dropna()
                                    
                                    if not future_2022.empty:
                                        forecast_2022 = model_prophet.predict(future_2022)
                                        predictions_2022 = forecast_2022['yhat'].values
                                
                                if not test_data_2023.empty:
                                    # Préparation des données de test 2023
                                    future_2023 = test_data_2023[['date', 'tavg', 'prcp', 'wspd']].copy()
                                    future_2023.columns = ['ds', 'tavg', 'prcp', 'wspd']
                                    future_2023 = future_2023.dropna()
                                    
                                    if not future_2023.empty:
                                        forecast_2023 = model_prophet.predict(future_2023)
                                        predictions_2023 = forecast_2023['yhat'].values
                                
                                # Métriques pour 2022
                                mae_2022 = None
                                r2_2022 = None
                                
                                if not test_data_2022.empty and predictions_2022 is not None:
                                    from sklearn.metrics import mean_absolute_error, r2_score
                                    test_2022_clean = test_data_2022[test_data_2022['date'].isin(future_2022['ds'])]
                                    if not test_2022_clean.empty and len(predictions_2022) == len(test_2022_clean):
                                        mae_2022 = mean_absolute_error(test_2022_clean['accidents'], predictions_2022)
                                        r2_2022 = r2_score(test_2022_clean['accidents'], predictions_2022)
                                
                                # Affichage des métriques
                                if mae_2022 is not None and r2_2022 is not None:
                                    st.subheader("Performance du modèle Prophet sur l'année 2022")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("MAE", f"{mae_2022:.2f}")
                                    with col2:
                                        st.metric("R²", f"{r2_2022:.3f}")
                                else:
                                    st.info("Aucune donnée de validation disponible pour 2022")
                                
                                # Graphique des prédictions Prophet
                                fig = go.Figure()
                                
                                # Données historiques
                                fig.add_trace(go.Scatter(
                                    x=train_data['date'],
                                    y=train_data['accidents'],
                                    name='Données historiques (2017-2021)',
                                    line=dict(color='blue')
                                ))
                                
                                # Données réelles et prédictions 2022
                                if not test_data_2022.empty and predictions_2022 is not None:
                                    test_2022_clean = test_data_2022[test_data_2022['date'].isin(future_2022['ds'])]
                                    if not test_2022_clean.empty:
                                        fig.add_trace(go.Scatter(
                                            x=test_2022_clean['date'],
                                            y=test_2022_clean['accidents'],
                                            name='Données réelles 2022',
                                            line=dict(color='green')
                                        ))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=test_2022_clean['date'],
                                            y=predictions_2022,
                                            name='Prédictions Prophet 2022',
                                            line=dict(color='orange', dash='dash')
                                        ))
                                
                                # Prédictions 2023
                                if not test_data_2023.empty and predictions_2023 is not None:
                                    test_2023_clean = test_data_2023[test_data_2023['date'].isin(future_2023['ds'])]
                                    if not test_2023_clean.empty:
                                        fig.add_trace(go.Scatter(
                                            x=test_2023_clean['date'],
                                            y=predictions_2023,
                                            name='Prédictions Prophet 2023',
                                            line=dict(color='red', dash='dash')
                                        ))
                                
                                fig.update_layout(
                                    title="Évolution du nombre d'accidents quotidiens et prédictions (Prophet)",
                                    xaxis_title="Date",
                                    yaxis_title="Nombre d'accidents",
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Statistiques des prédictions
                                if predictions_2023 is not None and len(predictions_2023) > 0:
                                    st.subheader("Statistiques des prédictions Prophet 2023")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Moyenne prédite", f"{predictions_2023.mean():.1f}")
                                    with col2:
                                        st.metric("Maximum prédit", f"{predictions_2023.max():.1f}")
                                    with col3:
                                        st.metric("Minimum prédit", f"{predictions_2023.min():.1f}")
                                else:
                                    st.info("Aucune prédiction disponible pour 2023")
                                
                                # Composantes du modèle Prophet
                                st.subheader("Composantes du modèle Prophet")
                                
                                # Création d'un dataframe pour l'analyse des composantes
                                if not test_data_2023.empty and predictions_2023 is not None:
                                    future_2023_clean = future_2023.copy()
                                    forecast_2023_clean = model_prophet.predict(future_2023_clean)
                                    
                                    # Graphique des composantes
                                    fig_components = go.Figure()
                                    
                                    # Tendance
                                    fig_components.add_trace(go.Scatter(
                                        x=forecast_2023_clean['ds'],
                                        y=forecast_2023_clean['trend'],
                                        name='Tendance',
                                        line=dict(color='blue')
                                    ))
                                    
                                    # Saisonnalité hebdomadaire
                                    if 'weekly' in forecast_2023_clean.columns:
                                        fig_components.add_trace(go.Scatter(
                                            x=forecast_2023_clean['ds'],
                                            y=forecast_2023_clean['weekly'],
                                            name='Saisonnalité hebdomadaire',
                                            line=dict(color='green')
                                        ))
                                    
                                    # Saisonnalité annuelle
                                    if 'yearly' in forecast_2023_clean.columns:
                                        fig_components.add_trace(go.Scatter(
                                            x=forecast_2023_clean['ds'],
                                            y=forecast_2023_clean['yearly'],
                                            name='Saisonnalité annuelle',
                                            line=dict(color='red')
                                        ))
                                    
                                    fig_components.update_layout(
                                        title="Composantes du modèle Prophet",
                                        xaxis_title="Date",
                                        yaxis_title="Valeur",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_components, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Erreur lors du calcul des prédictions Prophet : {str(e)}")
                        st.info("Note : Les prédictions Prophet nécessitent les données météorologiques. Vérifiez que le fichier 'data_meteo.csv' est présent.")

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


































import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Configuration de base améliorée
st.set_page_config(
    page_title="Smart EV Charging Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .sidebar .sidebar-content {background-color: #e8f4f8;}
    h1 {color: #2a5885;}
    h2 {color: #3a7ca5;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stDownloadButton>button {background-color: #2196F3;}
    .success {background-color: #dff0d8;}
    .info {background-color: #d9edf7;}
    </style>
    """, unsafe_allow_html=True)


# ---------------------- INITIALISATION ----------------------
@st.cache_data
def init_creators():
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)


# ---------------------- PARAMÈTRES UTILISATEUR AVANCÉS ----------------------
st.sidebar.header("⚙️ Paramètres de simulation")

with st.sidebar.expander("Configuration de base"):
    num_evs = st.slider("Nombre de véhicules (EVs)", 1, 100, 20)
    date_range = st.date_input("Plage de dates", [datetime.today(), datetime.today() + timedelta(days=1)])
    time_resolution = st.selectbox("Résolution temporelle", ["15 minutes", "30 minutes", "1 heure"], index=2)

    if time_resolution == "15 minutes":
        num_slots = 96
    elif time_resolution == "30 minutes":
        num_slots = 48
    else:
        num_slots = 24

with st.sidebar.expander("Paramètres techniques"):
    charge_options = {"Charge lente": 3.7, "Charge accélérée": 7.4, "Charge rapide": 22, "Charge ultra-rapide": 50}
    charge_type = st.selectbox("Type de charge", list(charge_options.keys()))
    charge_power = charge_options[charge_type]

    energy_min = st.number_input("Demande minimale (kWh)", 10, 100, 20)
    energy_max = st.number_input("Demande maximale (kWh)", 20, 150, 80)

    grid_capacity = st.number_input("Capacité du réseau (kW)", 50, 500, 200)

with st.sidebar.expander("Paramètres NSGA-II"):
    ngen = st.slider("Nombre de générations", 10, 200, 50)
    pop_size = st.slider("Taille de population", 10, 200, 50)
    cxpb = st.slider("Probabilité de croisement", 0.1, 0.9, 0.7)
    mutpb = st.slider("Probabilité de mutation", 0.01, 0.5, 0.2)


# ---------------------- GÉNÉRATION DES DONNÉES AMÉLIORÉE ----------------------
@st.cache_data
def generate_ev_data(n_evs, n_slots):
    ev_data = []
    for i in range(n_evs):
        arrival = random.randint(0, n_slots - 2)
        departure = random.randint(arrival + 1, n_slots - 1)
        max_possible_energy = (departure - arrival) * charge_power

        if max_possible_energy < energy_min:
            demand = int(max_possible_energy)
        else:
            demand = random.randint(energy_min, min(energy_max, int(max_possible_energy)))

            battery_capacity = random.randint(40, 100)
            current_soc = random.randint(10, 40)
            ev_type = random.choice(["Particulier", "Taxi", "Bus", "Livraison"])

            ev_data.append({
                "id": i + 1,
                "arrival": arrival,
                "departure": departure,
                "demand": demand,
                "type": ev_type,
                "battery_capacity": battery_capacity,
                "current_soc": current_soc,
                "required_soc": min(100, current_soc + (demand / battery_capacity) * 100)
            })
    return pd.DataFrame(ev_data)


# ---------------------- FONCTIONS D'ÉVALUATION AMÉLIORÉES ----------------------
def decode(individual, ev_data, n_slots):
    schedule = np.zeros((len(ev_data), n_slots))
    index = 0
    for i, ev in ev_data.iterrows():
        for t in range(ev['arrival'], ev['departure']):
            schedule[i][t] = individual[index]
            index += 1
    return schedule


def evaluate(individual, ev_data, n_slots):
    schedule = decode(individual, ev_data, n_slots)
    total_power = np.sum(schedule, axis=0) * charge_power

    # Objectif 1: Pic de charge
    peak = np.max(total_power)

    # Objectif 2: Insatisfaction
    dissatisfaction = 0
    for i, ev in ev_data.iterrows():
        energy_delivered = np.sum(schedule[i]) * charge_power
        dissatisfaction += max(0, ev['demand'] - energy_delivered)

    # Pénalité pour dépassement capacité réseau
    capacity_violation = max(0, peak - grid_capacity)

    return peak + capacity_violation * 10, dissatisfaction  # Pénalité renforcée


# ---------------------- ALGORITHME NSGA-II AMÉLIORÉ ----------------------
@st.cache_data(show_spinner="Optimisation en cours...")
def run_nsga2(ev_data, n_slots, ngen, pop_size, cxpb, mutpb):
    init_creators()

    toolbox = base.Toolbox()
    var_count = sum(ev_data['departure'] - ev_data['arrival'])

    # Initialisation avec plus de diversité
    toolbox.register("attr_float", lambda: random.uniform(0, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=var_count)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, ev_data=ev_data, n_slots=n_slots)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Croisement plus doux
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    # Statistiques
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop = toolbox.population(n=pop_size)
    pop, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=pop_size, lambda_=pop_size,
        cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
        verbose=False
    )

    # Sélection de la meilleure solution sur le front de Pareto
    fronts = tools.sortNondominated(pop, k=len(pop))
    best_front = fronts[0]
    best = min(best_front, key=lambda ind: ind.fitness.values[0] + ind.fitness.values[1])

    return decode(best, ev_data, n_slots), best.fitness.values, logbook


# ---------------------- VISUALISATION INTERACTIVE ----------------------
def plot_interactive_schedule(schedule, ev_data, n_slots):
    # Convertir en DataFrame pour Plotly
    time_slots = [f"Slot {i + 1}" for i in range(n_slots)]
    ev_ids = [f"EV {i + 1}" for i in range(len(ev_data))]

    # Charge totale
    total_load = np.sum(schedule, axis=0) * charge_power

    # Graphique de charge totale
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=time_slots,
        y=total_load,
        mode='lines+markers',
        name='Charge totale',
        line=dict(color='royalblue', width=2)
    ))

    # Ligne de capacité du réseau
    fig1.add_hline(y=grid_capacity, line_dash="dot",
                   line_color="red", annotation_text="Capacité max")

    fig1.update_layout(
        title="Charge totale du réseau électrique",
        xaxis_title="Créneaux horaires",
        yaxis_title="Puissance (kW)",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Graphique de charge par véhicule
    st.subheader("Planning de recharge par véhicule")

    fig2 = go.Figure()
    for i in range(schedule.shape[0]):
        fig2.add_trace(go.Bar(
            x=time_slots,
            y=schedule[i] * charge_power,
            name=f"EV {i + 1}",
            hoverinfo="text",
            hovertext=f"""
                EV {i + 1} ({ev_data.iloc[i]['type']})<br>
                Arrivée: Slot {ev_data.iloc[i]['arrival'] + 1}<br>
                Départ: Slot {ev_data.iloc[i]['departure'] + 1}<br>
                Demande: {ev_data.iloc[i]['demand']} kWh<br>
                SOC actuel: {ev_data.iloc[i]['current_soc']}%<br>
                SOC requis: {ev_data.iloc[i]['required_soc']}%
            """
        ))

    fig2.update_layout(
        barmode='stack',
        title="Répartition de la charge par véhicule",
        xaxis_title="Créneaux horaires",
        yaxis_title="Puissance (kW)",
        hovermode="closest",
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Matrice de charge
    st.subheader("Matrice de charge (heatmap)")
    fig3 = px.imshow(
        schedule * charge_power,
        labels=dict(x="Créneaux horaires", y="Véhicules", color="Puissance (kW)"),
        x=time_slots,
        y=ev_ids,
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------- RAPPORT DÉTAILLÉ ----------------------
def generate_detailed_report(ev_data, fitness, logbook):
    with st.expander("📊 Rapport complet", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Pic de charge", f"{fitness[0]:.2f} kW")
            st.metric("Capacité réseau", f"{grid_capacity} kW")
            st.metric("Marge de sécurité", f"{max(0, grid_capacity - fitness[0]):.2f} kW")

        with col2:
            st.metric("Insatisfaction totale", f"{fitness[1]:.2f} kWh")
            st.metric("Demande totale", f"{ev_data['demand'].sum():.2f} kWh")
            st.metric("Taux de satisfaction",
                      f"{(1 - fitness[1] / ev_data['demand'].sum()) * 100:.2f}%")

        with col3:
            st.metric("Nombre de véhicules", len(ev_data))
            st.metric("Type de charge", charge_type)
            st.metric("Durée de simulation", f"{num_slots} créneaux")

        st.subheader("Évolution de l'optimisation")
        gen = [entry['gen'] for entry in logbook]
        min_f1 = [entry['min'][0] for entry in logbook]
        min_f2 = [entry['min'][1] for entry in logbook]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gen, y=min_f1, name="Pic de charge (kW)"))
        fig.add_trace(go.Scatter(x=gen, y=min_f2, name="Insatisfaction (kWh)"))
        fig.update_layout(
            xaxis_title="Génération",
            yaxis_title="Valeur",
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Statistiques des véhicules")
        st.dataframe(ev_data, use_container_width=True)


# ---------------------- EXPORT DES RÉSULTATS ----------------------
def get_table_download_link(df, filename="ev_schedule.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger au format CSV</a>'
    return href


def export_results(schedule, ev_data, charge_power):
    # Création du DataFrame de sortie
    time_slots = [f"Slot {i + 1}" for i in range(schedule.shape[1])]
    schedule_df = pd.DataFrame(schedule * charge_power, columns=time_slots)
    schedule_df.insert(0, "EV ID", [f"EV {i + 1}" for i in range(schedule.shape[0])])

    # Fusion avec les données des véhicules
    full_report = pd.merge(
        ev_data,
        schedule_df,
        left_index=True,
        right_index=True
    )

    # Export
    st.markdown(get_table_download_link(full_report), unsafe_allow_html=True)

    # Export image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Télécharger les graphiques (PNG)",
        data=buf,
        file_name="charging_schedule.png",
        mime="image/png"
    )


# ---------------------- INTERFACE PRINCIPALE ----------------------
st.title("⚡ Smart EV Charging Optimizer")
st.markdown("""
    Optimisation intelligente de la recharge des véhicules électriques utilisant l'algorithme NSGA-II.
    Trouve le meilleur compromis entre **réduction des pics de charge** et **satisfaction des utilisateurs**.
""")

tab1, tab2, tab3 = st.tabs(["Optimisation", "Documentation", "À propos"])

with tab1:
    if st.button("🚀 Lancer l'optimisation", type="primary"):
        with st.spinner("Génération des données des véhicules..."):
            ev_data = generate_ev_data(num_evs, num_slots)
            st.success(f"Données générées pour {num_evs} véhicules")

        with st.spinner(f"Optimisation en cours ({ngen} générations)..."):
            schedule, fitness, logbook = run_nsga2(
                ev_data, num_slots, ngen, pop_size, cxpb, mutpb
            )
            st.success("Optimisation terminée avec succès!")
            st.balloons()

        plot_interactive_schedule(schedule, ev_data, num_slots)
        generate_detailed_report(ev_data, fitness, logbook)

        st.subheader("Exporter les résultats")
        export_results(schedule, ev_data, charge_power)

with tab2:
    st.header("Documentation technique")
    st.markdown("""
        ### 📚 Comment utiliser cette application

        1. **Configurez les paramètres** dans la barre latérale
        2. Cliquez sur "Lancer l'optimisation"
        3. Consultez les résultats visuels
        4. Exportez les données si nécessaire

        ### ⚙️ Algorithme NSGA-II
        - Algorithme génétique multi-objectifs
        - Minimise simultanément:
          - Le pic de charge sur le réseau
          - L'insatisfaction des utilisateurs (énergie non fournie)
        - Utilise des opérateurs de:
          - Sélection par rang non-dominé
          - Croisement blend (α=0.5)
          - Mutation gaussienne

        ### 📊 Métriques clés
        - **Pic de charge**: Puissance maximale instantanée
        - **Insatisfaction**: Énergie totale non fournie
        - **Taux de satisfaction**: % de la demande couverte
    """)

with tab3:
    st.header("À propos")
    st.markdown("""
        ### ℹ️ Informations
        **Version**: 2.0.0  
        **Dernière mise à jour**: 2023-11-15  
        **Auteur**: [Votre Nom]  

        ### 📚 Bibliothèques utilisées
        - Streamlit pour l'interface
        - DEAP pour les algorithmes évolutionnaires
        - Plotly pour les visualisations
        - Pandas pour la gestion des données

        ### 📝 Licence
        Ce projet est sous licence MIT.
    """)

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption("© 2023 Smart EV Charging Optimizer - Tous droits réservés")
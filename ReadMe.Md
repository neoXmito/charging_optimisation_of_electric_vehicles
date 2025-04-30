

# Optimisation Intelligente de la Recharge des VE avec NSGA-II

## Description du Projet

Ce projet se concentre sur l'optimisation des plannings de recharge pour une flotte de Véhicules Électriques (VE) afin de relever les défis posés par leur adoption croissante sur les réseaux électriques[cite: 2]. Il utilise l'Algorithme Génétique de Tri Non-Dominé II (NSGA-II), un algorithme génétique multi-objectifs, pour trouver des stratégies de recharge optimales[cite: 1, 37].

Les objectifs principaux sont :
1.  **Minimiser le Pic de Charge :** Réduire la demande de puissance maximale simultanée sur le réseau électrique[cite: 3, 12].
2.  **Maximiser la Satisfaction Utilisateur :** Assurer que les VE reçoivent l'énergie requise pendant leurs temps de connexion disponibles[cite: 3, 14].

Le système modélise la recharge des VE en fonction des heures d'arrivée, des heures de départ, des demandes d'énergie et de la puissance des bornes de recharge[cite: 11]. Il génère des solutions Pareto-optimales, représentant les meilleurs compromis possibles entre la minimisation de la charge du réseau et la satisfaction des besoins de recharge des utilisateurs[cite: 18, 43].

## Fonctionnalités

* **Optimisation Multi-Objectifs :** Utilise NSGA-II pour équilibrer des objectifs conflictuels (pic de charge vs satisfaction utilisateur)[cite: 17, 37].
* **Modélisation Mathématique :** Définition formelle des variables de décision (taux de charge $s_i(t)$), des fonctions objectifs ($f_1$ pour le pic de charge, $f_2$ pour la demande non satisfaite) et des contraintes (disponibilité temporelle, limites de puissance)[cite: 8, 12, 14, 16].
* **Visualisation Interactive :** Une interface Streamlit permet aux utilisateurs de configurer les paramètres de simulation (nombre de VE, plage horaire, capacité du réseau), d'exécuter l'optimisation et de visualiser les résultats, y compris les plannings de recharge par véhicule et la charge globale du réseau[cite: 4, 33, 45, 48].
* **Simulation de Données :** Capacité à générer des scénarios de recharge de VE aléatoires pour les tests[cite: 29].

## Technologies Utilisées

* **Algorithme :** NSGA-II [cite: 37]
* **Visualisation/UI :** Streamlit [cite: 4]
* **(Probablement) Langage :** Python (suggéré par l'utilisation de bibliothèques NSGA-II et Streamlit)

## Fonctionnement

1.  **Entrée des Données :** Définir les paramètres tels que le nombre de VE, l'horizon temporel, la capacité du réseau, la puissance de charge et les contraintes individuelles des VE (arrivée, départ, énergie nécessaire)[cite: 11, 30, 34].
2.  **Initialisation :** Générer une population initiale de plannings de recharge potentiels[cite: 39].
3.  **Évaluation :** Calculer le pic de charge ($f_1$) et la demande d'énergie totale non satisfaite ($f_2$) pour chaque planning dans la population[cite: 12, 14, 31].
4.  **Opérations NSGA-II :** Appliquer le tri non-dominé, le calcul de la distance de peuplement (crowding distance), la sélection, le croisement et la mutation pour générer une nouvelle population, améliorant itérativement vers le front de Pareto[cite: 38, 39, 50].
5.  **Sortie :** Présenter les solutions Pareto-optimales, montrant les compromis entre les objectifs, et visualiser les plannings de recharge sélectionnés[cite: 42, 43, 48].

## Équipe Projet [cite: 1]

* Ismaili Amina
* Chouacha Marwa
* Chali Malama
* Traore Souleymane
* Akakro Koffi Moise

# Comment utiliser le code?


### 1. Créer un environnement virtuel: python -m venv env
### 2. Activez l'environnement virtuel: env/Scripts/actvate (for windows users)
### 3. Installer les requirements dans le fichier requirements.txt en faisant: pip install -r requirements.txt
### 4. Lancer le streamlit: streamlit run streamlit.py

**Encadrant :** Mr. Ramchoun [cite: 1]
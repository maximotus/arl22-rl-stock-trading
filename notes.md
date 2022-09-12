# Meeting 21.07.2022

## TODO
- Coursera Kurs durcharbeiten: https://www.coursera.org/learn/trading-strategies-reinforcement-learning/home/week/1
- Recherche bezüglich existierenden Stock TRading Gym-Environments
- Literaturrecherche:
  - Was für RL-Algorithmen für Stock-Trading existieren bereits?
  - Sind existierende Algorithmen auch schon in StableBaselines3 vertreten?
  - Welche Entscheidungs-Indikatoren existieren und sind geeignet für unser Projekt?  

## Fragen an Prof
- Was genau ist mit Kapselung der Datenquelle als OpenAI-Gym Environment gemeint?
  - RL-Policy
  - Agent muss trainiert werden
  - wir können auch existierende environments nutzen (github)
  - ess muss icht unbedingt ein gym-environment verwendet werden (also wir können auch selbst einen simulator programmierne)
  - wifi netz!?
  - MetaTrader5 nicht verpflichtend
- Können wir auch Arbitrage-Trading (fällt unter HFT) verwenden?
  - nein
  - lieber über normale features gehen

## Tipps von Prof
- rechenintensiv wird das Training
- wir sollten mehrere Features verwenden
  - verschiedene
  - z.B. Lufthansa braucht Öl
  - DB abhängig vom STrompreis
  - Firma mit Abhängigkeiten
- Tipp zum Indikator: Differenz (z.B. Abweichung vom 30-Tage-Mittelwert)

# Meeting 16.08.2022

## TODO
- Coursera Kurs machen
- Poetry installieren / getting started

## Tipps von Prof
- Modellierung des Kurses mittels LSTM
- Coursera Kurs fertig machen
- PPO1 / PPO2 oder ACKTR? Oder beides?
- Nächste Woche sollten wir was handfestes haben...
- Market Volume könnte auch ein wichtiges Feature sein

# Meeting 24.08.2022

## Topics
- Agent
- Models (PPO1, PPO2, ACKTR, DQN)
- Why do we need an own DQN implementation?
- Maybe we should better start with pre-implemented models and test it on our own gym-env and run experiments

## TODO
- [] Korbi und ich integrieren stable-baselines
- Jonas macht ggf. bei uns mit
- Andreas debuggt das gym env
- Updates heute Nachmittag
- GPU usage integrieren


INFO: Jonas ist Wochenende weg...
Voll da ab 03.09.

# Meeting 26.08.2022 mit Prof

## Tipps

- Alternatives RL framework zu stable-baselines3: d3rlpy

## Anforderungen an Dokumentation

- So wenig wie nötig, so viel wie möglich
- Aufbau
  - Related Work
  - Architektur / Code
  - Experimente
    - Setup
    - Ergebnisse
    - Benchmarks
    - Performance
    - Plots und Tabellen
  - Result

# Meeting 30.08.2022 mit Prof
- Rohdaten über Callbacks speichern (in beliebigem Format)
- Parametervariantionen (werden bei uns in den .yaml Dateien gespeichert)
- Zusätzlich Tensorboard

# Anytrading
- Actions: Sell=0, Buy=1
- Positions: Short=0, Long=1
- Attributes:
  - prices (real prices over time, used for render)

# Meeting am 06.09.2022
- Vergleichen immer nur auf gleichen Datasets
- Tendenz immer steigend oder immer fallend --> Mehr Daten
- Multiexperimente?
  - n x selbes Experiment auf gleicher Sequenz
  - nichts größeres
  - wenn zeit, dann rechnerei
- graphen?
  - kerzengröße, aktuell 1 minute
  - trainingsdauer
  - kovergenz??? --> loss kurve
  - normalisierte daten verwenden! (nur die trainingsdaten normalisieren)

# TODO

## Max
- [ ] Email an Tokic mit Bitte um Aufschiebung
- [ ] Problem mit  linear steigenden Preisen in Historie lösen
- [ ] Experimente auf Cluster starten
- [ ] Bericht schreiben
# World Cup 2026 — Machine Learning Prediction Model

**Author:** Hien Nguyen  
**Notebook:** `worldcup26.ipynb`  
**Goal:** Predict match outcomes and simulate the FIFA World Cup 2026 tournament using historical World Cup data, team ratings, and an ensemble of machine learning classifiers.

---

## What This Project Does

This notebook builds a full ML prediction pipeline for the 2026 FIFA World Cup, held across the USA, Canada, and Mexico. Starting from historical match records dating back to 1930, the model engineers 35 features per match — including Elo ratings, xG (expected goals), FIFA 2026 rankings, form metrics, and contextual factors — then trains an ensemble classifier to predict win/draw/loss probabilities for any matchup. Those probabilities feed a Poisson-based tournament simulator that runs 2,000 Monte Carlo simulations to produce championship win probabilities for all 48 teams.

---

## Datasets

| File | Description |
|------|-------------|
| `matches_1930_2022.csv` | 964 World Cup matches with scorelines, xG (128 matches, 2018+), red cards, stage, venue |
| `schedule_2026.csv` | All 2026 group-stage fixtures |
| `fifa_ranking_2026-06-08.csv` | FIFA world rankings as of June 8, 2026 — current team strength |
| `fifa_ranking_2022-10-06.csv` | October 2022 rankings (reference) |
| `world_cup.csv` | Tournament metadata |

---

## Cell-by-Cell Breakdown

### Cell 1 — Imports & Data Loading (`cell_imports`)
Loads all five datasets and imports the ML stack: pandas, numpy, scikit-learn (LogisticRegression, RandomForest, CalibratedClassifierCV, TimeSeriesSplit), LabelEncoder, and Counter for Monte Carlo tallying. Warning suppression is applied globally so sklearn feature-name warnings do not pollute outputs.

**Output:** `Successfully loaded all datasets.`

---

### Cell 2 — Sort by Date (`cell_sort`)
Converts `Date` to datetime and sorts all 964 matches chronologically from 1930–2022. Chronological order is critical — every feature we compute is a rolling window that must only look backward in time.

**Output:** Confirms sort with earliest match: France vs Mexico, July 13, 1930.

---

### Cell 3 — Team Name Standardisation (`cell_names`)
Applies a name mapping dictionary to unify inconsistent spellings across the dataset and the 2026 schedule (e.g., `"Czechia"` → `"Czech Republic"`, `"United States"` → `"USA"`, `"Curaçao"` → `"Curacao"`). Also strips whitespace with `clean_team()`.

**Output:** `Team names cleaned.`

---

### Cell 4 — Time-Decayed Elo with Margin-of-Victory Multiplier (`cell_elo`)
**Upgrade from baseline:** The original notebook used a flat K=30 Elo with no time weighting, meaning a 1954 win was treated identically to a 2022 win.

This cell implements:
- **Time decay:** K-factor shrinks exponentially with age, halving every 4 years relative to June 2026, so recent results carry full weight and 1930 results approach zero
- **Margin-of-victory multiplier:** Uses `log(goal_diff + 1) + 1`, dampened when a heavy favourite wins big, to prevent Elo inflation from lopsided mismatches

**Key result — Top 10 Elo ratings after processing all 964 matches:**

| Team | Elo |
|------|-----|
| France | 1598.7 |
| Netherlands | 1570.4 |
| Argentina | 1563.0 |
| Brazil | 1554.9 |
| England | 1543.4 |
| Belgium | 1536.3 |
| Croatia | 1530.0 |
| Germany | 1526.3 |
| Portugal | 1524.1 |
| Colombia | 1520.3 |

Note: Unlike the flat Elo baseline (which ranked Netherlands #1 at 1701 due to decades of historical wins), the time-decayed version correctly elevates France, Argentina, and England based on their 2018–2022 dominance.

---

### Cell 5 — xG Preprocessing + Stage Flag (`cell_xg_preprocess`)
Parses `home_xg` / `away_xg` as numerics (available for 128 matches from 2018 onwards) and falls back to actual goals for older matches. Red cards are similarly coerced with 0-fill. A binary `is_knockout` column is derived from the `Round` field to distinguish group-stage from knockout matches.

**Key results:**
- xG data available for **128 matches** (2018 Qatar + 2022 Qatar tournaments)
- **270 knockout matches** identified across all tournaments since 1930

---

### Cell 6 — Exponentially-Weighted Form Function (`cell_form_fn`)
**Upgrade from baseline:** The original used a simple 5-game average. This cell defines `calculate_form_weighted(history, window=10)` which applies exponential weights (`0.9^(n-1-i)`) over the last 10 games, giving the most recent match ~3× the influence of a match 10 games ago.

The function returns: win rate, draw rate, loss rate, goals for, goals against, xG for, xG against, and red cards — all as weighted averages.

---

### Cell 7 — Rebuild Team History with xG + Red Cards (`cell_history`)
Iterates all 964 matches in chronological order to build `team_history` — a dictionary of every team's game-by-game record. At each match, it first reads the rolling form **before** the match (no leakage), then appends the result. New columns written to the `matches` DataFrame include rolling xG averages, red card averages, and derived diff features (`xg_diff`, `form_diff`, `attack_diff`, `defense_diff`, `draw_rate_diff`).

**Key result:** `Team history built for 86 teams.`

---

### Cell 8 — FIFA 2026 Ranking Injection (`cell_fifa`)
**Upgrade from baseline:** The baseline had no current-tournament team strength signal. This cell loads the June 8, 2026 FIFA rankings and maps them onto team names using an inverse dictionary, then derives a `rank_score` (0–1, higher = stronger) for use as a feature. Every team entry in `team_dict` is enriched with `fifa_rank`, `fifa_pts`, and `fifa_score`.

**Key result — FIFA Top 10 as of June 2026:**

| Team | Rank | Points |
|------|------|--------|
| Argentina | 1 | 1876.1 |
| Spain | 2 | 1873.0 |
| France | 3 | 1869.4 |
| England | 4 | 1827.0 |
| Portugal | 5 | 1766.2 |
| Brazil | 6 | 1765.9 |
| Morocco | 7 | 1755.1 |
| Netherlands | 8 | 1751.1 |
| Belgium | 9 | 1742.2 |
| Germany | 10 | 1735.8 |

**Coverage gap:** Cape Verde, Curacao, and Czech Republic were missing from the FIFA file and defaulted to rank 150.

---

### Cell 9 — Contextual Features: Rest Days + Confederation (`cell_context`)
**New features not in baseline:** Adds three contextual signals:
- **Rest days:** Days since each team's previous World Cup match, capped at 365. Captures scheduling fatigue, particularly relevant in the compressed 2026 format
- **Confederation strength score:** UEFA=5, CONMEBOL=4, CONCACAF=3, CAF/AFC=2, OFC=1 — encodes the competitive tier each team developed in
- **FIFA rank on matches:** Directly maps the current ranking onto historical match rows for training

**Output:** `Contextual features added.`

---

### Cell 10 — Feature List + `build_match_features` (`cell_features`)
Defines the complete 35-feature vector and the `build_match_features(home_team, away_team, is_knockout)` function that assembles a single-row DataFrame for prediction. Features are organised into five groups: Elo signals (3), form/goals (14), xG + red cards (5), FIFA rank (4), stage + context (9).

**Output:** `Total features: 35`

---

### Cell 11 — Time-Aware Cross-Validation + Model Training (`cell_training`)
**Upgrade from baseline:** The original used a single random 80/20 split, which risks data leakage. This cell uses `TimeSeriesSplit(n_splits=5)` so each fold trains on past matches and validates on future matches — the correct evaluation protocol for temporal data.

**Cross-validation results (Logistic Regression):**

| Fold | Accuracy |
|------|----------|
| 1 | 29.4% |
| 2 | 33.8% |
| 3 | 41.9% |
| 4 | 38.1% |
| 5 | 39.4% |
| **Mean** | **36.5% ± 4.4%** |

The rising accuracy across folds is expected — earlier folds train on very little data (pre-1970 matches). The held-out test set (most recent 20% of matches) produces 39% accuracy for base LR. Football match prediction is inherently noisy: draw frequency (~22% of outcomes) and the near-random nature of individual goals creates a theoretical ceiling around 50–55% for binary win/loss prediction.

---

### Cell 12 — Probability Calibration (`cell_calibration`)
**Upgrade from baseline:** Raw classifier probabilities tend to cluster too close to the base rate (33/33/33 in a 3-class problem). `CalibratedClassifierCV(method='isotonic', cv=5)` fits a non-parametric calibration curve so stated probabilities better reflect true frequencies.

**Key result:**

| | Away | Draw | Home |
|--|------|------|------|
| Raw probs mean | 48.8% | 29.6% | 21.6% |
| Calibrated probs mean | 39.1% | 25.2% | 35.6% |

The raw model was systematically overconfident in Away outcomes. Calibration corrects this, producing more balanced and trustworthy probabilities for the Monte Carlo simulation.

---

### Cell 13 — Ensemble Model (`cell_ensemble`)
**Upgrade from baseline:** The original used a single logistic regression. This cell trains four models and soft-votes (averages) their probabilities:

| Model | Input type |
|-------|-----------|
| Calibrated Logistic Regression | Scaled numpy |
| Random Forest (500 trees, depth 10) | Named DataFrame |
| XGBoost (300 estimators, lr=0.05) | Scaled numpy |
| LightGBM (300 estimators, lr=0.05) | Scaled numpy |

**Ensemble test-set performance:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Away | 0.45 | 0.68 | 0.54 |
| Draw | 0.08 | 0.05 | 0.06 |
| Home | 0.52 | 0.40 | 0.45 |
| **Accuracy** | | | **42%** |

**Top 10 most predictive features (Random Forest importances):**

| Feature | Importance |
|---------|-----------|
| elo_diff | 7.4% |
| away_elo | 5.9% |
| home_elo | 5.3% |
| home_xg_for | 4.6% |
| home_last5_goals_for | 4.5% |
| xg_diff | 3.8% |
| attack_diff | 3.8% |
| form_diff | 3.7% |
| home_last5_loss_rate | 3.5% |
| draw_rate_diff | 3.3% |

Elo-based features dominate, confirming that long-run rating is the strongest predictor. xG (where available) ranks 4th, validating its inclusion. Draw prediction remains the hardest class — an industry-wide challenge in football modelling.

---

### Cell 14 — Poisson Goal Model (`cell_poisson`)
**Upgrade from baseline:** The original simulation converted classifier probabilities to xG via a hardcoded formula (`1.2 + (prob[2] - prob[0]) * 1.5`), which was fragile and uncalibrated. This cell builds a proper Dixon-Coles-style attack/defense rating for each team:

```
home_xg = (home_attack × away_defense / league_avg) × 1.10  [home advantage]
away_xg = (away_attack × home_defense / league_avg)
```

**League average:** 1.78 goals/game across all 964 World Cup matches.

**Team attack/defense ratings (weighted last 10 games):**

| Team | Attack | Defense |
|------|--------|---------|
| France | 2.31 | 1.12 |
| Argentina | 2.16 | 1.52 |
| England | 1.84 | 0.98 |
| Spain | 1.83 | 1.02 |
| Brazil | 1.61 | 0.66 |

Brazil's defense (0.66 goals conceded per game) is the strongest of any top team, while France leads in attack rate.

---

### Cell 15 — Stage-Aware Simulation Functions (`cell_sim_fns`)
Implements the full simulation stack:
- `simulate_match()` — Poisson draws from attack/defense ratings; group stage allows draws
- `simulate_knockout_match()` — adds extra time (Poisson λ=0.3 per side) and Elo-tilted penalty shootout for draws
- `simulate_group()` — runs all fixtures, tracks points/GD/GF
- `simulate_knockout()` — generic bracket reducer that works for any power-of-2 team count

**Key design:** Extra-time scoring uses a reduced λ=0.3 (lower than normal play) reflecting real-world ET goal rates. Penalty outcomes tilt toward the Elo-stronger team by at most ±20% (clipped to 30–70%).

---

### Cell 16 — Group Stage Setup (`cell_groups`)
Defines all 12 groups (A–L) matching the official 2026 draw, maps each team to its group, and slices `schedule_2026` into per-group fixture lists. The WC 2026 format: 12 groups of 4, 6 matches per group, top 2 guaranteed to advance.

---

### Cell 17 — Monte Carlo Tournament Simulation (`cell_montecarlo`)
Runs 2,000 complete tournament simulations. Each simulation:
1. Simulates all 6 group-stage matches per group using Poisson draws
2. Ranks groups by points → GD → GF
3. Takes top 2 from each group (24 teams) + best 8 third-place finishers = **32 teams** (correct WC 2026 format)
4. Runs a 5-round knockout bracket to a champion

**Championship win probabilities (2,000 simulations):**

| Team | Win % |
|------|-------|
| **Netherlands** | **15.2%** |
| Brazil | 13.8% |
| France | 12.2% |
| Colombia | 7.0% |
| England | 6.2% |
| Belgium | 5.9% |
| Spain | 5.5% |
| Argentina | 4.2% |
| Portugal | 3.4% |
| Germany | 2.9% |

---

### Cell 18 — Group Match Probability Table (`cell_predictions`)
Uses the ensemble model to predict win/draw/loss percentages for every group-stage fixture. Notable results:

| Match | Home Win % | Draw % | Away Win % |
|-------|-----------|--------|-----------|
| Spain vs Saudi Arabia | 64.8% | 23.4% | 11.8% |
| Belgium vs IR Iran | 62.7% | 32.1% | 5.2% |
| Brazil vs Haiti | 59.0% | 32.1% | 8.9% |
| Panama vs Croatia | 7.3% | 19.6% | **73.1%** |
| Tunisia vs Netherlands | 7.3% | 15.5% | **77.2%** |
| Norway vs France | 12.1% | 10.4% | **77.5%** |
| Scotland vs Brazil | 8.8% | 19.5% | **71.6%** |

---

## Conclusion

This notebook represents a complete first-generation ML pipeline for international football prediction, built and refined through 10 systematic upgrades over a statistical baseline.

**What the model learned:**

The three most important predictors — `elo_diff`, `away_elo`, `home_elo` — confirm the core intuition that accumulated historical performance (rating) is more predictive than recent form alone. xG (4th most important) adds genuine signal for modern matches where it is available. Draw prediction remains the hardest problem in football modelling: with 22% base rate and high stochasticity, the ensemble achieves only 8% precision on draws, consistent with academic benchmarks.

**Tournament conclusion:**

After 2,000 Monte Carlo simulations, **Netherlands (15.2%)**, **Brazil (13.8%)**, and **France (12.2%)** emerge as co-favourites — a tight three-way cluster. The notable finding is **Colombia at 7.0%**, ranking 4th ahead of traditional powers Spain, Argentina, and Germany. This reflects Colombia's strong recent Elo trajectory and favourable group draw (Group K with Congo, Uzbekistan, Portugal — a group the model expects them to top). The FIFA rankings (Spain #2, Argentina #1) disagree with the simulation rankings; the gap is explained by the Poisson goal model using weighted recent World Cup form rather than all competitive matches, and Argentina's relatively high defense concession rate (1.52) hurting their knockout survival probability.

**Limitations and honest caveats:**

- 42% test accuracy is the expected range for this problem. Academic papers on football prediction typically report 45–55% for match outcomes; World Cup data (964 matches over 92 years) is smaller than club datasets, and the distribution shift across eras limits how much historical data helps
- Draw prediction is structurally difficult — the model should be used for win probability ranking, not draw forecasting
- The xG data only covers 128 matches; for older matches the model falls back to goals, reducing the quality of that feature for most training examples
- Cape Verde, Curacao, and Czech Republic have no FIFA ranking data and default to rank 150, which may undervalue them

**For future versions:** Adding squad-level data (player club minutes, injury lists, age profiles), qualifying campaign statistics, and pre-tournament betting market odds as a calibration anchor would likely push test accuracy into the 47–52% range.

---

## How to Run

```bash
cd "WC prediction"
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
jupyter notebook worldcup26.ipynb
```

Run all cells top to bottom. Cells 11–13 take ~60 seconds (model training). Cell 17 (Monte Carlo, 2000 sims) takes ~30–60 seconds depending on hardware.

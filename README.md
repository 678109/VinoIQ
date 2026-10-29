# VinoIQ 🍷

En interaktiv Streamlit-app som estimerer kvaliteten på rødvin basert på kjemiske målinger.

## Windows – kom i gang

Åpne prosjektmappen i VS Code og start en terminal (Terminal → New Terminal).

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train_model
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Åpne adressen terminalen viser (vanligvis http://localhost:8501).

## Datasett

UCI Wine Quality (rødvin): https://archive.ics.uci.edu/dataset/186/wine+quality

Første treningskjøring laster ned `winequality-red.csv` fra UCI. For full reproduserbarhet etter nedlasting kan datasettet beholdes i `data/`.

## Metode

- Separat testsett (20 %) og valideringssett (20 %).
- DummyRegressor som baseline, Ridge Regression og Random Forest.
- Modellvalg etter lavest MAE på valideringssettet.
- Sluttmodell trenes på 80 % og evalueres én gang på testsettet.
- Resultatene lagres i `models/metrics.json` og modellen i `models/wine_model.joblib`.

## Begrensninger

Prediksjonen er et statistisk estimat av sensoriske vurderinger, ikke en garanti for faktisk kvalitet. Modellen er kun trent på rødviner fra dette datasettet.

## Kilder

- Cortez, P. et al. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*.
- UCI Machine Learning Repository: Wine Quality.
- Scikit-learn: https://scikit-learn.org/
- Streamlit: https://streamlit.io/

## AI-assistanse

Kodeutkast og dokumentasjon er utviklet med hjelp fra OpenAI ChatGPT. Kjøring, testing, resultater og endelig innlevering må kontrolleres av prosjektgruppen.

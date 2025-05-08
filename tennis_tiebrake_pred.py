import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Veriyi oku ve ön işlemeleri yap
df = pd.read_csv('atp_matches_2024.csv')
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
df['score'] = df['score'].astype(str)

# Score stringini parse et
def parse_score(score_str):
    score_str = score_str.strip()
    if score_str in ['W/O', 'WO']:
        return {'sets': [], 'result': 'walkover'}
    if 'RET' in score_str:
        score_str = score_str.replace('RET', '').strip()
        result_type = 'retired'
    elif 'ABD' in score_str:
        score_str = score_str.replace('ABD', '').strip()
        result_type = 'abandoned'
    else:
        result_type = 'normal'
    set_scores = re.findall(r'\d+-\d+(?:\(\d+\))?', score_str)
    parsed = {f'set{i+1}': val for i, val in enumerate(set_scores)}
    parsed['num_sets'] = len(set_scores)
    parsed['result'] = result_type
    return parsed

score_parsed_df = df['score'].apply(parse_score).apply(pd.Series)
df = pd.concat([df, score_parsed_df], axis=1)

# Tiebreak içeriğini hesapla
def summarize_sets(row):
    sets = [row.get(f'set{i}') for i in range(1, 6) if pd.notna(row.get(f'set{i}'))]
    has_tb = any('(' in s for s in sets if isinstance(s, str))
    return pd.Series({'has_tiebreak': int(has_tb)})

df[['has_tiebreak']] = df.apply(summarize_sets, axis=1)

# Gerekli sayısal özellikler
num_cols = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
            'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
            'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# Eğitim verisini oluştur
df_model = df[df['result'] == 'normal'].dropna(subset=num_cols + ['has_tiebreak'])

X = df_model[num_cols]
y = df_model['has_tiebreak']

# Ölçekleme ve modelleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Rapor
print("\nClassification Report for 'has_tiebreak':")

print(classification_report(y_test, y_pred, zero_division=0))

# Özellik önemini hesapla
import pandas as pd
import matplotlib.pyplot as plt

feature_names = X.columns
importances = model.feature_importances_

feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nEn Önemli Özellikler ('has_tiebreak') için:")
print(feature_importances.head(10).to_string(index=False))

# Grafikle göster
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'][:10], feature_importances['Importance'][:10])
plt.xlabel("Önem Skoru")
plt.title("En Önemli Özellikler (has_tiebreak)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
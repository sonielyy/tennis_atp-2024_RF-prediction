import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Veriyi oku
df = pd.read_csv('atp_matches_2024.csv')

# Sayısal dönüşümler
int_cols = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
            'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
            'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']
df[int_cols] = df[int_cols].apply(pd.to_numeric, errors='coerce')

# Temel filtreleme (normal maçlar)
df = df[df['score'].notna()]
df['score'] = df['score'].astype(str)
df = df[~df['score'].str.contains('RET|W/O|ABD')].copy()

# Etiket oluştur (kazanan p1, kaybeden p2)
df_transformed = pd.DataFrame()
for col in df.columns:
    if col.startswith('w_'):
        df_transformed['p1_' + col[2:]] = df[col]
    elif col.startswith('l_'):
        df_transformed['p2_' + col[2:]] = df[col]
df_transformed['p1_win'] = 1
df_transformed['p2_win'] = 0

# Veri augmentasyonu: p1 ve p2 swap
df_swapped = df_transformed.copy()
p1_cols = [col for col in df_swapped.columns if col.startswith('p1_')]
p2_cols = [col for col in df_swapped.columns if col.startswith('p2_')]
mask = np.random.rand(len(df_swapped)) > 0.5
for p1_col in p1_cols:
    p2_col = 'p2_' + p1_col[3:]
    temp = df_swapped.loc[mask, p1_col].copy()
    df_swapped.loc[mask, p1_col] = df_swapped.loc[mask, p2_col]
    df_swapped.loc[mask, p2_col] = temp
df_swapped.loc[mask, 'p1_win'] = 0
df_swapped.loc[mask, 'p2_win'] = 1

# Modelleme
X = df_swapped.drop(columns=['p1_win', 'p2_win'])
y = df_swapped['p1_win']

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nClassification Report for 'p1_win':")
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

print("\nEn Önemli Özellikler ('p1_win') için:")
print(feature_importances.head(10).to_string(index=False))

# Grafikle göster
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'][:10], feature_importances['Importance'][:10])
plt.xlabel("Önem Skoru")
plt.title("En Önemli Özellikler (p1_win)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
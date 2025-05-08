import pandas as pd

# CSV dosyasını DataFrame'e aktar
df = pd.read_csv('atp_matches_2024.csv')  # 'dosya_adi.csv' yerine kendi dosya adını yaz

import numpy as np

# CSV dosyasını DataFrame'e aktar
df = pd.read_csv('atp_matches_2024.csv')  # 'dosya_adi.csv' yerine kendi dosya adını yaz

# Tarih sütununu datetime formatına çevir
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')

# Aşağıdaki sütunları tam sayıya çevir (gerekiyorsa eksik değerler için float)
int_columns = ['draw_size', 'match_num', 'winner_id', 'loser_id', 'winner_seed', 'loser_seed',
               'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
               'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn',
               'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'winner_rank',
               'winner_rank_points', 'loser_rank', 'loser_rank_points']

for col in int_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Float'a çevrilecek sütunlar (yaş, boy gibi kesirli değerler)
float_columns = ['winner_ht', 'winner_age', 'loser_ht', 'loser_age']
for col in float_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Kategorik veri türüne çevrilecek sütunlar
category_columns = ['tourney_id', 'tourney_name', 'surface', 'tourney_level', 'round',
                    'winner_entry', 'loser_entry', 'winner_hand', 'loser_hand',
                    'winner_name', 'loser_name', 'winner_ioc', 'loser_ioc']

for col in category_columns:
    df[col] = df[col].astype('category')

# Skor gibi karma metin verileri string olarak kalabilir
df['score'] = df['score'].astype(str)

# Kontrol: Dönüşüm sonrası veri tiplerini yazdır
print(df.dtypes)

# DataFrame'i tablo formatında yazdır
print(df.head(5).to_string(index=False))

# Kategorik sütunları listeleyelim + score
categorical_columns = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_level', 'round',
    'winner_name', 'loser_name', 'winner_hand', 'loser_hand',
    'winner_ioc', 'loser_ioc', 'winner_entry', 'loser_entry',
    'score'
]
# Yeni bir DataFrame oluşturalım
df_categorical = df[categorical_columns].copy()

# DataFrame'i tablo formatında yazdır
print(df_categorical.head(5).to_string(index=False))

# Unique değerler
unique_counts = df_categorical.nunique()
print(unique_counts)

# Çıkarılacaklar
# tourney_id, tourney_name, winner_name, loser_name,

# Binary encodelar
# surface, tourney_level, winner_hand, loser_hand

# Label encodelar
# round

# Target encoding: O ülkelerin ortalama sayı ortalamaları
# winner_ioc, loser_ioc (ülkeler)

from sklearn.preprocessing import LabelEncoder

# Label Encoding: round
le_round = LabelEncoder()
df['round_encoded'] = le_round.fit_transform(df['round'])

# Binary Encoding (One-hot): surface, tourney_level, winner_hand, loser_hand
df = pd.get_dummies(df, columns=['surface', 'tourney_level', 'winner_hand', 'loser_hand'], drop_first=True)

# Target Encoding: winner_ioc, loser_ioc
# Burada 'target' olarak örnek bir sayı kullanmalıyız, örneğin oyuncunun kazanıp kazanmadığı gibi.
# Kazanan oyuncunun ülkesine göre ortalama galibiyet elde edeceğiz.

# DataFrame'i tablo formatında yazdır
print(df["score"].head(300).to_string(index=False))

import re


def parse_score(score_str):
    score_str = score_str.strip()

    if score_str in ['W/O', 'WO']:
        return {'sets': [], 'result': 'walkover'}
    elif 'RET' in score_str:
        score_str = score_str.replace('RET', '').strip()
        result_type = 'retired'
    elif 'ABD' in score_str:
        score_str = score_str.replace('ABD', '').strip()
        result_type = 'abandoned'
    else:
        result_type = 'normal'

    set_scores = re.findall(r'\d+-\d+(?:\(\d+\))?', score_str)

    parsed = {}
    for i, set_score in enumerate(set_scores):
        parsed[f'set{i + 1}'] = set_score

    parsed['num_sets'] = len(set_scores)
    parsed['result'] = result_type
    return parsed

score_parsed_df = df['score'].apply(parse_score).apply(pd.Series)

df = pd.concat([df, score_parsed_df], axis=1)

import re

def summarize_sets(row):
    sets = [row.get(f'set{i}') for i in range(1, 6) if pd.notna(row.get(f'set{i}'))]
    sets_won, sets_lost, has_tb, max_diff = 0, 0, 0, 0

    for s in sets:
        match = re.match(r'(\d+)-(\d+)', s)
        if not match:
            continue
        w, l = int(match.group(1)), int(match.group(2))
        if '(' in s:
            has_tb = 1
        diff = abs(w - l)
        max_diff = max(max_diff, diff)
        if w > l:
            sets_won += 1
        else:
            sets_lost += 1

    return pd.Series({
        'sets_won': sets_won,
        'sets_lost': sets_lost,
        'has_tiebreak': has_tb
    })
# 1. Kazanan ülkelerin sayısı (winner tarafı)
winner_freq = df['winner_ioc'].value_counts()
df['winner_ioc_freq'] = df['winner_ioc'].map(winner_freq)

# 2. Kaybeden ülkelerin sayısı (loser tarafı)
loser_freq = df['loser_ioc'].value_counts()
df['loser_ioc_freq'] = df['loser_ioc'].map(loser_freq)

# Uygula ve df'e ekle
df[['sets_won', 'sets_lost', 'has_tiebreak']] = df.apply(summarize_sets, axis=1)
print(df.head(300).to_string(index=False))

# Gereksiz Kategorik Sütunların Çıkarılması
df = df.drop(columns=[
    'tourney_id', 'tourney_name', 'draw_size', 'tourney_date', 'winner_id', 'winner_seed', 'loser_id', 'loser_seed',
    'winner_name', 'loser_name',
    'round', 'winner_ioc', 'loser_ioc',
    'winner_entry', 'loser_entry',
    'score',
    'set1', 'set2', 'set3', 'set4', 'num_sets', 'set5'
])
print(df.head(10).to_string(index=False))


df = df[df['result'] == 'normal'].reset_index(drop=True)
df = df.drop(columns=[
    'result', 'sets_won', 'sets_lost', 'sets'
])

print(df.head(10).to_string(index=False))

###################################################################

# Orijinal df'in kopyasını al
df_transformed = pd.DataFrame()

# p1 (winner) sütunlarını yeniden adlandır
for col in df.columns:
    if col.startswith('winner_'):
        df_transformed[col.replace('winner_', 'p1_')] = df[col]
    elif col.startswith('w_'):
        df_transformed[col.replace('w_', 'p1_')] = df[col]

# p2 (loser) sütunlarını yeniden adlandır
for col in df.columns:
    if col.startswith('loser_'):
        df_transformed[col.replace('loser_', 'p2_')] = df[col]
    elif col.startswith('l_'):
        df_transformed[col.replace('l_', 'p2_')] = df[col]

# Natural (tarafsız) sütunları ekle
winner_loser_prefixes = ('winner_', 'loser_', 'w_', 'l_')
natural_cols = [col for col in df.columns if not col.startswith(winner_loser_prefixes)]
df_transformed[natural_cols] = df[natural_cols]

# Etiket sütunlarını ekle
df_transformed['p1_win'] = 1
df_transformed['p2_win'] = 0

#########################################################################################

import numpy as np

# Kopya al
df_swapped = df_transformed.copy()

# p1_ ve p2_ ile başlayan sütunları ayır
p1_cols = [col for col in df_swapped.columns if col.startswith('p1_')]
p2_cols = [col for col in df_swapped.columns if col.startswith('p2_')]

# Rastgele olarak hangi satırlarda takas yapılacağını belirle
swap_mask = np.random.rand(len(df_swapped)) > 0.5

# p1_ ve p2_ sütunlarının isimlerini eşleştirerek yer değiştir
for p1_col in p1_cols:
    p2_col = 'p2_' + p1_col[3:]  # p1_x → p2_x eşleştirmesi
    temp = df_swapped.loc[swap_mask, p1_col].copy()
    df_swapped.loc[swap_mask, p1_col] = df_swapped.loc[swap_mask, p2_col]
    df_swapped.loc[swap_mask, p2_col] = temp

# Etiket sütunlarını da güncelle
df_swapped.loc[swap_mask, 'p1_win'] = 0
df_swapped.loc[swap_mask, 'p2_win'] = 1

print(df_swapped.head(10).to_string(index=False))

########################################################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# 1. match_num sütununu çıkar
df_model = df_swapped.drop(columns=['match_num'])

# 2. Hedef değişken: p1_win (veya istersen p2_win, ama genellikle biri yeterlidir)
y = df_model['p1_win']
X = df_model.drop(columns=['p1_win', 'p2_win'])  # p2_win de çıkarılır çünkü p1_win zaten sonucu gösteriyor

# 3. Özellikleri ölçekle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Veriyi eğitim ve test olarak ayır (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Random Forest ile eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Test verisi ile tahmin
y_pred = model.predict(X_test)

# 7. Confusion Matrix hesapla
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 2. Diğer metrikler
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nClassification Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# 3. Daha detaylı çıktı için:
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


import pandas as pd
import matplotlib.pyplot as plt

# Özellik isimlerini al
feature_names = X.columns

# Özellik önemlerini al
importances = model.feature_importances_

# Önemleri bir DataFrame'e aktar
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# En önemli 20 özelliği yazdır
print("\nEn önemli özellik:")
print(feature_importances.to_string(index=False))

# Grafik olarak da göster (isteğe bağlı)
plt.figure(figsize=(12, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel("Önem Skoru")
plt.title("En Önemli Özellik (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

################################################################################

# Hedefi değiştir: has_tiebreak
y = df_model['has_tiebreak']
X = df_model.drop(columns=['has_tiebreak', 'p1_win', 'p2_win'])

# Özellikleri ölçekle
X_scaled = scaler.fit_transform(X)

# Eğitim-test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Modeli eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Tahmin ve metrikler
y_pred = model.predict(X_test)

print("\nClassification Report for 'has_tiebreak':")
print(classification_report(y_test, y_pred, zero_division=0))


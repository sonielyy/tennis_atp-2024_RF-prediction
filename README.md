# 🎾 ATP 2024 Match Outcome Prediction with Random Forest

This project uses **ATP 2024 tennis match statistics** to predict the winner of a match using a **Random Forest Classifier**. The pipeline includes data cleaning, transformation, augmentation, and model training with performance evaluation.

![image](https://github.com/user-attachments/assets/e46e9530-ed06-4ac9-80f9-b9b2593170bd)

---

## Dataset

- **Source:** `atp_matches_2024.csv`
- **Reference:** ATP data collected by Jeff Sackmann, available at [Tennis Abstract GitHub repo](https://github.com/JeffSackmann/tennis_atp?tab=readme-ov-file#atp-tennis-rankings-results-and-stats), used under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
- **Content:** Match-by-match performance statistics of players (e.g., aces, double faults, serve points).
- **Preprocessing:**
  - Matches with missing scores or early terminations (`RET`, `W/O`, `ABD`) are removed.
  - Statistical columns are converted to numeric types, coercing errors to NaN.

---

## Data Processing

- Winners are labeled as `p1`, losers as `p2` to create a consistent feature space.
- Data augmentation applied by **randomly swapping `p1` and `p2`** features to prevent bias.
- Binary classification target: `p1_win` (1 if p1 won, 0 otherwise).

---

## Model Details

- **Model:** `RandomForestClassifier` from `scikit-learn`
- **Scaling:** `StandardScaler` applied to features
- **Train-Test Split:** 70% training, 30% testing (stratified)
- **Evaluation:** `classification_report` (precision, recall, F1-score)

---

## Feature Importance

The most influential features used by the model:

![Figure_1](https://github.com/user-attachments/assets/92a0082d-edf6-4a88-b2aa-b9154daaf66e)

*Note: To regenerate this plot, ensure matplotlib is installed and run the full script.*

---

## Example Output

```text
Classification Report for 'p1_win':
              precision    recall  f1-score   support

           0       0.87      0.88      0.87       450
           1       0.88      0.86      0.87       443

    accuracy                           0.87       893
   macro avg       0.87      0.87      0.87       893
weighted avg       0.87      0.87      0.87       893
```


## License

This project includes tennis data compiled by Jeff Sackmann, available at  
[Tennis Abstract GitHub repository](https://github.com/JeffSackmann), and is licensed under the  
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

In other words:  
- Attribution is required  
- Commercial use is not permitted  
- Any derived work must be shared under the same license

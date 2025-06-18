import sys
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ─────────── CONFIGURATION ───────────
FILE_PATH     = '/Users/sorbet/Desktop/Dev/2048-AI-Solver/data.csv'
CLEAN_PATH    = '/Users/sorbet/Desktop/Dev/2048-AI-Solver/clean_dataset.csv'
MODEL_PATH    = '/Users/sorbet/Desktop/Dev/2048-AI-Solver/model.joblib'
TARGET_COL    = 'bepp2_move'
DROP_COLS     = ['bepp2_val']
SKIP_COLS     = 4
ALLOWED_MOVES = ['up', 'down', 'left', 'right']
TEST_SIZE     = 0.1   # 90/10 split
CV_FOLDS      = 5    # Utiliser 5 splis
# ───────────────────────────────────────

def make_usecols(path, skip):
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    kept = cols[skip:]
    if TARGET_COL in cols[:skip]:
        kept.append(TARGET_COL)
    return [c for c in kept if c not in DROP_COLS]

def load_dataframe(path, skip):
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    usecols = make_usecols(path, skip)
    return pd.read_csv(path, usecols=usecols, low_memory=False)

def preprocess_df(df):
    df = df[df[TARGET_COL].isin(ALLOWED_MOVES)]
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype('category')
    obj_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    non_target = [col for col in obj_cols if col != TARGET_COL]
    for col in non_target:
        orig = df[col]
        coerced = pd.to_numeric(orig, errors='coerce')
        bad_vals = orig[coerced.isna()].unique()[:10]
        if len(bad_vals) > 0:
            print(f"❌ '{col}' non-convertibles: {list(bad_vals)}")
        dec_mask = coerced.notna() & (coerced % 1 != 0)
        if dec_mask.any():
            print(f"⚠️ '{col}' décimales: {orig[dec_mask].unique()[:10].tolist()}")
        mask_drop = coerced.isna() | dec_mask
        if mask_drop.any():
            before = df.shape[0]
            df = df.loc[~mask_drop]
            after = df.shape[0]
            print(f"🗑️  Suppr. {before-after} lignes dans '{col}'")
            coerced = coerced.loc[df.index]
        df[col] = coerced.astype('Int64')
    return df


def main():
    timestamps = {}
    # Lecture
    t0 = time.time()
    print(f"➡️ Lecture de : {FILE_PATH}")
    try:
        df = load_dataframe(FILE_PATH, SKIP_COLS)
    except Exception as e:
        print(f"❌ Erreur de lecture : {e}")
        sys.exit(1)
    timestamps['load'] = time.time() - t0
    print(f"📋 Colonnes initiales ({len(df.columns)}): {df.columns.tolist()}")
    print(f"🆕 Lignes avant prétrait.: {df.shape[0]}")

    # Prétraitement
    t1 = time.time()
    df = preprocess_df(df)
    timestamps['preprocess'] = time.time() - t1
    print(f"🔍 Colonnes après prétrait.: {df.columns.tolist()}")
    print(f"✅ Lignes après prétrait.: {df.shape[0]}")

    # Sauvegarde dataset clean
    t2 = time.time()
    df.to_csv(CLEAN_PATH, index=False)
    timestamps['save_clean'] = time.time() - t2
    print(f"💾 Dataset nettoyé enregistré dans {CLEAN_PATH}")

    # Split
    t3 = time.time()
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    timestamps['split'] = time.time() - t3
    print(f"✔️ Split train/test : {X_train.shape[0]}/{X_test.shape[0]}")

    # Validation croisée 5 splis
    t4 = time.time()
    min_count = y_train.value_counts().min()
    n_splits = min(CV_FOLDS, min_count)
    if n_splits >= 2:
        print(f"🔄 CV à {n_splits} plis (min classe={min_count})…")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = HistGradientBoostingClassifier()
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring='accuracy', n_jobs=-1, error_score='raise')
        print(f"   Scores CV: {scores}")
        print(f"   Moyenne CV Acc.: {scores.mean():.4f}")
    else:
        print(f"⚠️ CV ignorée (min classe={min_count} <2)")
    timestamps['cv'] = time.time() - t4

    # Entraînement final
    t5 = time.time()
    print("🏋️ Entraînement final…")
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)
    timestamps['train'] = time.time() - t5

    # Sauvegarde du modèle
    t6 = time.time()
    joblib.dump(model, MODEL_PATH)
    timestamps['save_model'] = time.time() - t6
    print(f"💾 Modèle enregistré dans {MODEL_PATH}")

    # Prédiction et évaluation
    t7 = time.time()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = (cm.astype('float') / cm.sum(axis=1)[:, None] * 100).round(2)
    timestamps['eval'] = time.time() - t7

    # Affichage temps
    print("⏱️ Temps par étape (s):")
    for k, v in timestamps.items(): print(f"  {k}: {v:.2f}")

    # Résultats
    print(f"\n✅ Test Accuracy: {acc:.4f}\n")
    print("📊 Matrice de confusion (valeurs brutes):")
    print(cm)
    print("\n📊 Matrice de confusion (% par ligne):")
    print(cm_pct)
    print("\n📈 Rapport de classification :")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    main()

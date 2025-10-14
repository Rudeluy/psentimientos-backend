import kagglehub, os, pandas as pd, glob, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# === Descargar dataset IMDb español ===
path = kagglehub.dataset_download("luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish")
candidates = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
if not candidates:
    candidates = glob.glob(os.path.join(path, "**/*.tsv"), recursive=True)
file = candidates[0]
sep = "\t" if file.endswith(".tsv") else ","
df_raw = pd.read_csv(file, sep=sep, encoding="utf-8")

# === Selección y limpieza ===
possible_text_cols = ["review", "text", "texto", "review_es", "reviewText", "review_body"]
possible_label_cols = ["sentiment", "label", "polarity", "sentimiento", "target", "sentiment_es"]

def pick_first(cols, options):
    for c in options:
        if c in cols:
            return c
    return None

text_col = pick_first(df_raw.columns.tolist(), possible_text_cols)
label_col = pick_first(df_raw.columns.tolist(), possible_label_cols)
df = df_raw[[text_col, label_col]].dropna()
df = df.rename(columns={text_col: "texto", label_col: "sentimiento"})

# === Normalizar etiquetas ===
def to_binary(y):
    if isinstance(y, str):
        y2 = y.strip().lower()
        if y2 in ["positive","pos","positivo"]: return 1
        if y2 in ["negative","neg","negativo"]: return 0
    return None

df["sentimiento"] = df["sentimiento"].apply(to_binary)
df = df.dropna()

# === Entrenamiento ===
X_train, X_test, y_train, y_test = train_test_split(
    df["texto"], df["sentimiento"], test_size=0.2, random_state=42, stratify=df["sentimiento"]
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=50000, min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
pipe.fit(X_train, y_train)

# === Guardar modelo ===
joblib.dump(pipe, "sentiment_model.pkl")
print("✅ Modelo de sentimientos guardado como sentiment_model.pkl")

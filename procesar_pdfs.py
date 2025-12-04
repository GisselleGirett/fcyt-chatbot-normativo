import os
import pypdf
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# === CONFIGURACIÓN ===
# Carpeta donde están tus PDFs
PDF_DIR = "docs"   # asegurate que tus 2 PDFs estén dentro de ./docs/


def extraer_texto(pdf_path):
    """Extrae texto de un PDF página por página."""
    reader = pypdf.PdfReader(pdf_path)
    texto = ""
    for page in reader.pages:
        try:
            texto += page.extract_text() + "\n"
        except Exception:
            pass
    return texto


def dividir_en_chunks(texto, max_chars=500):
    """Divide texto largo en bloques (~500 caracteres aprox)."""
    palabras = texto.split()
    chunks = []
    actual = []
    count = 0

    for p in palabras:
        actual.append(p)
        count += len(p) + 1
        if count > max_chars:
            chunks.append(" ".join(actual))
            actual = []
            count = 0

    if actual:
        chunks.append(" ".join(actual))

    return chunks


# === PROCESAMIENTO PRINCIPAL ===
documentos = []

print("Buscando PDFs en la carpeta:", PDF_DIR)

for archivo in os.listdir(PDF_DIR):
    if archivo.lower().endswith(".pdf"):
        ruta = os.path.join(PDF_DIR, archivo)
        print(f"\nProcesando PDF: {archivo}")

        texto = extraer_texto(ruta)
        chunks = dividir_en_chunks(texto)

        for chunk in chunks:
            documentos.append({"texto": chunk, "fuente": archivo})

print(f"\nTotal de fragmentos: {len(documentos)}")

texts = [d["texto"] for d in documentos]

# === CREAR MATRIZ TF-IDF ===
print("\nGenerando matriz TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),        # unigrams + bigrams para algo de "semántica"
    stop_words=None            # podrías poner 'spanish' si querés
)

X = vectorizer.fit_transform(texts)  # matriz (n_docs x vocab)

# Para simplificar, la guardamos como matriz densa float32
embeddings = X.toarray().astype("float32")

print("Dimensión del espacio vectorial:", embeddings.shape[1])

# === GUARDAR ARCHIVOS ===
print("\nGuardando índice...")
data = {
    "documentos": documentos,
    "vectorizer": vectorizer,
    "embeddings": embeddings,
}

with open("indice_tfidf.pkl", "wb") as f:
    pickle.dump(data, f)

print("\n✔ ¡Proceso completado!")
print("  - indice_tfidf.pkl generado")
print("Listo para usar con el chatbot.")

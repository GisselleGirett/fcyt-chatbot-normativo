import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar índice TF-IDF y metadatos
with open("indice_tfidf.pkl", "rb") as f:
    data = pickle.load(f)

documentos = data["documentos"]    # lista de dicts {texto, fuente}
vectorizer = data["vectorizer"]    # TfidfVectorizer entrenado
embeddings = data["embeddings"]    # matriz (n_docs x dim)


def buscar_respuesta(pregunta: str, k: int = 3):
    """Dada una pregunta, devuelve los k fragmentos más similares."""
    # Vectorizar la pregunta con el mismo TF-IDF
    q_vec = vectorizer.transform([pregunta]).toarray().astype("float32")

    # Similaridad coseno entre pregunta y todos los fragmentos
    sims = cosine_similarity(q_vec, embeddings)[0]  # shape: (n_docs,)

    # Índices de los k mejores, ordenados de mayor a menor similitud
    idxs = np.argsort(sims)[::-1][:k]

    resultados = []
    for idx in idxs:
        doc = documentos[idx]
        resultados.append(
            {
                "score": float(sims[idx]),
                "texto": doc["texto"],
                "fuente": doc["fuente"],
            }
        )
    return resultados


if __name__ == "__main__":
    print("=== Chatbot normativo FCyT (modo consola) ===")
    print("Escribe tu pregunta sobre reglamentos. Escribe 'salir' para terminar.")

    while True:
        q = input("\nPregunta: ").strip()
        if not q:
            continue
        if q.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break

        resultados = buscar_respuesta(q, k=3)

        print("\n--- Resultados ---")
        for i, r in enumerate(resultados, start=1):
            print(f"\n[{i}] Fuente: {r['fuente']} (score: {r['score']:.3f})")
            # Mostramos solo los primeros 400 caracteres para no inundar
            texto_corto = r["texto"][:400].replace("\n", " ")
            print(texto_corto + ("..." if len(r['texto']) > 400 else ""))

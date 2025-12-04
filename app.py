import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

# ==== CARGA DEL ÍNDICE ====

with open("indice_tfidf.pkl", "rb") as f:
    data = pickle.load(f)

documentos = data["documentos"]    # lista de dicts {texto, fuente}
vectorizer = data["vectorizer"]    # TfidfVectorizer entrenado
embeddings = data["embeddings"]    # matriz (n_docs x dim)

app = FastAPI(title="Chatbot normativo FCyT")


def buscar_respuesta(pregunta: str, k: int = 3):
    """Devuelve los k fragmentos más similares a la pregunta."""
    q_vec = vectorizer.transform([pregunta]).toarray().astype("float32")
    sims = cosine_similarity(q_vec, embeddings)[0]

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


# ==== MODELOS DE ENTRADA / SALIDA ====

class Question(BaseModel):
    question: str


# ==== RUTAS ====

@app.get("/", response_class=HTMLResponse)
def home():
    # HTML minimalista con JS para llamar al endpoint /ask
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8" />
        <title>Chatbot normativo FCyT</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
            textarea { width: 100%; height: 80px; font-size: 1rem; }
            button { padding: 0.5rem 1rem; margin-top: 0.5rem; cursor: pointer; }
            .resultado { margin-top: 1rem; padding: 0.75rem; border-radius: 6px; border: 1px solid #ddd; background: #f9f9f9; }
            .fuente { font-size: 0.85rem; color: #555; }
            .score { font-size: 0.8rem; color: #999; }
        </style>
    </head>
    <body>
        <h1>Chatbot normativo FCyT</h1>
        <p>Preguntá sobre reglamento académico o de PFG. (Versión demo basada en búsqueda TF-IDF).</p>

        <label for="question">Pregunta:</label>
        <textarea id="question" placeholder="Ej: ¿Cuál es la función del docente de la materia PFG?"></textarea>
        <br />
        <button onclick="enviarPregunta()">Consultar</button>

        <div id="resultados"></div>

        <script>
            async function enviarPregunta() {
                const q = document.getElementById("question").value.trim();
                const contResultados = document.getElementById("resultados");
                contResultados.innerHTML = "";

                if (!q) {
                    contResultados.innerHTML = "<p>Escribí una pregunta.</p>";
                    return;
                }

                contResultados.innerHTML = "<p>Buscando...</p>";

                try {
                    const resp = await fetch("/ask", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({ question: q })
                    });

                    if (!resp.ok) {
                        contResultados.innerHTML = "<p>Error al consultar el servidor.</p>";
                        return;
                    }

                    const data = await resp.json();
                    contResultados.innerHTML = "";

                    if (!data.resultados || data.resultados.length === 0) {
                        contResultados.innerHTML = "<p>No se encontraron resultados.</p>";
                        return;
                    }

                    data.resultados.forEach((r, i) => {
                        const div = document.createElement("div");
                        div.className = "resultado";

                        const textoCorto = r.texto.length > 500
                            ? r.texto.slice(0, 500) + "..."
                            : r.texto;

                        div.innerHTML = `
                            <div class="fuente"><strong>[${i+1}] ${r.fuente}</strong>
                                <span class="score">(score: ${r.score.toFixed(3)})</span>
                            </div>
                            <div>${textoCorto.replace(/\\n/g, " ")}</div>
                        `;
                        contResultados.appendChild(div);
                    });

                } catch (e) {
                    contResultados.innerHTML = "<p>Error en la petición: " + e + "</p>";
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/ask")
def ask(q: Question):
    resultados = buscar_respuesta(q.question, k=3)
    return {"resultados": resultados}

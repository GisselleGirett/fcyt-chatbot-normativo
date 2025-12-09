"""
app.py - Sistema Normativo FCyT UNCA
Backend mejorado con panel de administración y búsqueda híbrida
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import pickle
import shutil
from pathlib import Path
from datetime import datetime

# Importar el motor de búsqueda mejorado
from search_engine import SearchEngine

app = FastAPI(title="FCyT Chatbot Normativo")

# Configuración de rutas
DOCS_DIR = Path("docs")
INDEX_FILE = Path("indice_tfidf.pkl")
UPLOAD_DIR = Path("uploads")

# Crear directorios necesarios
DOCS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Inicializar motor de búsqueda
search_engine = None

def load_search_engine():
    """Cargar o crear el motor de búsqueda"""
    global search_engine
    try:
        if INDEX_FILE.exists():
            search_engine = SearchEngine.load(INDEX_FILE)
            print(f"✅ Índice cargado: {len(search_engine.chunks)} fragmentos")
        else:
            print("⚠️  No se encontró índice. Ejecute: python procesar_pdfs.py")
            search_engine = None
    except Exception as e:
        print(f"❌ Error al cargar índice: {e}")
        search_engine = None

# Cargar al iniciar
load_search_engine()


# ===== MODELOS PYDANTIC =====

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResult(BaseModel):
    text: str
    document: str
    score: float
    chunk_id: int

class DocumentInfo(BaseModel):
    id: str
    name: str
    size: str
    date: str
    status: str
    path: str


# ===== ENDPOINTS DE ADMINISTRACIÓN =====

@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    """Listar todos los documentos PDF disponibles"""
    documents = []
    
    if not DOCS_DIR.exists():
        return documents
    
    for pdf_file in DOCS_DIR.glob("*.pdf"):
        stat = pdf_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        
        documents.append(DocumentInfo(
            id=str(pdf_file.stem),
            name=pdf_file.name,
            size=f"{size_mb:.2f} MB",
            date=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
            status="active",
            path=str(pdf_file)
        ))
    
    return sorted(documents, key=lambda x: x.date, reverse=True)


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Subir un nuevo documento PDF"""
    
    # Validar extensión
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    
    # Guardar archivo
    file_path = DOCS_DIR / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Regenerar índice automáticamente
        from procesar_pdfs import procesar_pdfs
        procesar_pdfs()
        
        # Recargar motor de búsqueda
        load_search_engine()
        
        return {
            "message": f"Documento '{file.filename}' subido exitosamente",
            "filename": file.filename,
            "status": "indexed"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al subir documento: {str(e)}")


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Eliminar un documento PDF"""
    
    file_path = DOCS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    
    try:
        # Eliminar archivo
        file_path.unlink()
        
        # Regenerar índice
        from procesar_pdfs import procesar_pdfs
        procesar_pdfs()
        
        # Recargar motor de búsqueda
        load_search_engine()
        
        return {
            "message": f"Documento '{filename}' eliminado exitosamente",
            "status": "reindexed"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar documento: {str(e)}")


@app.post("/api/reindex")
async def reindex_documents():
    """Regenerar el índice TF-IDF manualmente"""
    
    try:
        from procesar_pdfs import procesar_pdfs
        result = procesar_pdfs()
        
        # Recargar motor de búsqueda
        load_search_engine()
        
        return {
            "message": "Índice regenerado exitosamente",
            "documents_processed": result.get("total_docs", 0),
            "chunks_created": result.get("total_chunks", 0),
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al regenerar índice: {str(e)}")


# ===== ENDPOINTS DE BÚSQUEDA =====

@app.post("/api/search", response_model=List[SearchResult])
async def search_documents(request: QueryRequest):
    """Realizar búsqueda en los documentos"""
    
    if not search_engine:
        raise HTTPException(
            status_code=503,
            detail="Motor de búsqueda no inicializado. Ejecute: python procesar_pdfs.py"
        )
    
    try:
        results = search_engine.search(request.query, top_k=request.top_k)
        
        return [
            SearchResult(
                text=r["text"],
                document=r["document"],
                score=float(r["score"]),
                chunk_id=r["chunk_id"]
            )
            for r in results
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


@app.get("/api/stats")
async def get_statistics():
    """Obtener estadísticas del sistema"""
    
    if not search_engine:
        return {
            "status": "not_initialized",
            "total_documents": 0,
            "total_chunks": 0,
            "index_size": "0 MB"
        }
    
    index_size = INDEX_FILE.stat().st_size / (1024 * 1024) if INDEX_FILE.exists() else 0
    
    return {
        "status": "ready",
        "total_documents": len(set(c["document"] for c in search_engine.chunks)),
        "total_chunks": len(search_engine.chunks),
        "index_size": f"{index_size:.2f} MB",
        "search_method": "TF-IDF + Cosine Similarity"
    }


# ===== INTERFAZ WEB =====

@app.get("/", response_class=HTMLResponse)
async def root():
    """Interfaz principal del sistema"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistema Normativo FCyT - UNCA</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { opacity: 0.9; font-size: 1.1em; }
            .content { padding: 40px; }
            .tabs {
                display: flex;
                border-bottom: 2px solid #e0e0e0;
                margin-bottom: 30px;
            }
            .tab {
                flex: 1;
                padding: 15px;
                text-align: center;
                cursor: pointer;
                font-weight: 600;
                color: #666;
                transition: all 0.3s;
            }
            .tab.active {
                color: #667eea;
                border-bottom: 3px solid #667eea;
                background: #f8f9ff;
            }
            .tab:hover { background: #f8f9ff; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .search-box {
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
            }
            .search-box input {
                flex: 1;
                padding: 15px 20px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                transition: all 0.3s;
            }
            .search-box input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            .btn {
                padding: 15px 30px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            .btn-primary {
                background: #667eea;
                color: white;
            }
            .btn-primary:hover { background: #5568d3; transform: translateY(-2px); }
            .result {
                background: #f8f9ff;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin-bottom: 15px;
                border-radius: 10px;
                transition: all 0.3s;
            }
            .result:hover { box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .result-meta {
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
                font-size: 14px;
            }
            .doc-name { color: #667eea; font-weight: 600; }
            .score { background: #667eea; color: white; padding: 3px 10px; border-radius: 20px; }
            .result-text { color: #333; line-height: 1.6; }
            .loading {
                text-align: center;
                padding: 40px;
                color: #666;
            }
            .admin-actions {
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
            }
            .doc-list {
                background: #f8f9ff;
                border-radius: 10px;
                padding: 20px;
            }
            .doc-item {
                background: white;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .stat-value { font-size: 2.5em; font-weight: bold; }
            .stat-label { opacity: 0.9; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎓 Sistema Normativo FCyT - UNCA</h1>
                <p>Búsqueda inteligente en documentos normativos con TF-IDF</p>
            </div>
            
            <div class="content">
                <div class="tabs">
                    <div class="tab active" onclick="switchTab('search')">🔍 Búsqueda</div>
                    <div class="tab" onclick="switchTab('admin')">⚙️ Administración</div>
                </div>
                
                <!-- PESTAÑA DE BÚSQUEDA -->
                <div id="search" class="tab-content active">
                    <div class="search-box">
                        <input 
                            type="text" 
                            id="queryInput" 
                            placeholder="Ej: ¿Cuál es la función del docente en PFG?"
                            onkeypress="if(event.key === 'Enter') searchDocuments()"
                        >
                        <button class="btn btn-primary" onclick="searchDocuments()">Buscar</button>
                    </div>
                    <div id="results"></div>
                </div>
                
                <!-- PESTAÑA DE ADMINISTRACIÓN -->
                <div id="admin" class="tab-content">
                    <div class="admin-actions">
                        <input type="file" id="fileInput" accept=".pdf" style="display:none" onchange="uploadFile()">
                        <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">
                            📤 Subir PDF
                        </button>
                        <button class="btn btn-primary" onclick="reindexDocuments()">
                            🔄 Regenerar Índice
                        </button>
                        <button class="btn btn-primary" onclick="loadDocuments()">
                            📋 Actualizar Lista
                        </button>
                    </div>
                    <div id="docList" class="doc-list"></div>
                    <div class="stats" id="stats"></div>
                </div>
            </div>
        </div>
        
        <script>
            function switchTab(tabName) {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                event.target.classList.add('active');
                document.getElementById(tabName).classList.add('active');
                
                if (tabName === 'admin') {
                    loadDocuments();
                    loadStats();
                }
            }
            
            async function searchDocuments() {
                const query = document.getElementById('queryInput').value.trim();
                if (!query) return;
                
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<div class="loading">🔍 Buscando...</div>';
                
                try {
                    const response = await fetch('/api/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, top_k: 5 })
                    });
                    
                    const results = await response.json();
                    
                    if (results.length === 0) {
                        resultsDiv.innerHTML = '<div class="loading">No se encontraron resultados</div>';
                        return;
                    }
                    
                    resultsDiv.innerHTML = results.map(r => `
                        <div class="result">
                            <div class="result-meta">
                                <span class="doc-name">📄 ${r.document}</span>
                                <span class="score">${(r.score * 100).toFixed(0)}%</span>
                            </div>
                            <div class="result-text">${r.text}</div>
                        </div>
                    `).join('');
                } catch (error) {
                    resultsDiv.innerHTML = '<div class="loading">❌ Error en la búsqueda</div>';
                }
            }
            
            async function loadDocuments() {
                const docListDiv = document.getElementById('docList');
                docListDiv.innerHTML = '<div class="loading">Cargando documentos...</div>';
                
                try {
                    const response = await fetch('/api/documents');
                    const docs = await response.json();
                    
                    if (docs.length === 0) {
                        docListDiv.innerHTML = '<div class="loading">No hay documentos cargados</div>';
                        return;
                    }
                    
                    docListDiv.innerHTML = docs.map(d => `
                        <div class="doc-item">
                            <div>
                                <strong>${d.name}</strong><br>
                                <small>${d.size} | ${d.date}</small>
                            </div>
                            <button class="btn" onclick="deleteDocument('${d.name}')" style="background:#e74c3c;color:white">
                                🗑️ Eliminar
                            </button>
                        </div>
                    `).join('');
                } catch (error) {
                    docListDiv.innerHTML = '<div class="loading">❌ Error al cargar documentos</div>';
                }
            }
            
            async function loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    
                    document.getElementById('stats').innerHTML = `
                        <div class="stat-card">
                            <div class="stat-value">${stats.total_documents}</div>
                            <div class="stat-label">Documentos</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.total_chunks}</div>
                            <div class="stat-label">Fragmentos</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${stats.index_size}</div>
                            <div class="stat-label">Tamaño índice</div>
                        </div>
                    `;
                } catch (error) {
                    console.error('Error al cargar estadísticas', error);
                }
            }
            
            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    alert('✅ ' + result.message);
                    loadDocuments();
                    loadStats();
                } catch (error) {
                    alert('❌ Error al subir archivo');
                }
                
                fileInput.value = '';
            }
            
            async function deleteDocument(filename) {
                if (!confirm(`¿Eliminar "${filename}"?`)) return;
                
                try {
                    const response = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
                        method: 'DELETE'
                    });
                    
                    const result = await response.json();
                    alert('✅ ' + result.message);
                    loadDocuments();
                    loadStats();
                } catch (error) {
                    alert('❌ Error al eliminar documento');
                }
            }
            
            async function reindexDocuments() {
                if (!confirm('¿Regenerar el índice TF-IDF?')) return;
                
                try {
                    const response = await fetch('/api/reindex', { method: 'POST' });
                    const result = await response.json();
                    alert('✅ ' + result.message);
                    loadStats();
                } catch (error) {
                    alert('❌ Error al regenerar índice');
                }
            }
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
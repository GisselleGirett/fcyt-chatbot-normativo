"""
search_engine.py - Motor de búsqueda híbrido TF-IDF + Embeddings
Mejora técnica para el proyecto FCyT
"""

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import re

# Intentar importar sentence-transformers (opcional)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers no disponible. Usando solo TF-IDF")


class SearchEngine:
    """
    Motor de búsqueda híbrido que combina:
    1. TF-IDF (baseline)
    2. Embeddings densos (opcional, si hay GPU)
    3. Re-ranking inteligente
    """
    
    def __init__(self, use_embeddings: bool = True, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Inicializar motor de búsqueda
        
        Args:
            use_embeddings: Usar embeddings densos además de TF-IDF
            model_name: Modelo de Sentence Transformers a usar
        """
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=self._get_spanish_stopwords()
        )
        
        self.chunks = []
        self.tfidf_matrix = None
        self.embedding_matrix = None
        self.embedding_model = None
        
        # Cargar modelo de embeddings si está disponible
        if self.use_embeddings:
            try:
                print(f"🚀 Cargando modelo de embeddings: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
                print("✅ Modelo de embeddings cargado (GPU detectada)" if self._has_gpu() else 
                      "✅ Modelo de embeddings cargado (CPU)")
            except Exception as e:
                print(f"⚠️  Error al cargar embeddings: {e}")
                self.use_embeddings = False
    
    @staticmethod
    def _get_spanish_stopwords():
        """Palabras comunes en español a ignorar"""
        return [
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
            'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo',
            'pero', 'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese',
            'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy', 'sin',
            'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno', 'mismo', 'yo',
            'también', 'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'primero',
            'desde', 'grande', 'eso', 'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella',
            'les', 'cual', 'os', 'donde', 'tan', 'mientras', 'menos', 'mis', 'ese'
        ]
    
    def _has_gpu(self):
        """Detectar si hay GPU disponible"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def normalize_text(self, text: str) -> str:
        """Normalización avanzada del texto"""
        # Minúsculas
        text = text.lower()
        
        # Quitar caracteres especiales pero mantener espacios
        text = re.sub(r'[^a-záéíóúñü\s]', ' ', text)
        
        # Quitar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def build_index(self, chunks: List[Dict]):
        """
        Construir índice de búsqueda
        
        Args:
            chunks: Lista de diccionarios con 'text', 'document', 'chunk_id'
        """
        print(f"📚 Construyendo índice con {len(chunks)} fragmentos...")
        
        self.chunks = chunks
        texts = [self.normalize_text(c['text']) for c in chunks]
        
        # 1. Índice TF-IDF
        print("🔤 Construyendo índice TF-IDF...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"✅ Matriz TF-IDF: {self.tfidf_matrix.shape}")
        
        # 2. Índice de embeddings (si está habilitado)
        if self.use_embeddings and self.embedding_model:
            print("🧠 Generando embeddings densos...")
            try:
                # Los embeddings se generan con el texto original (no normalizado)
                original_texts = [c['text'] for c in chunks]
                self.embedding_matrix = self.embedding_model.encode(
                    original_texts,
                    show_progress_bar=True,
                    batch_size=32,
                    convert_to_numpy=True
                )
                print(f"✅ Matriz de embeddings: {self.embedding_matrix.shape}")
            except Exception as e:
                print(f"❌ Error al generar embeddings: {e}")
                self.use_embeddings = False
        
        print("✅ Índice construido exitosamente")
    
    def search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict]:
        """
        Realizar búsqueda híbrida
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados a devolver
            alpha: Peso de TF-IDF (1-alpha = peso de embeddings)
        
        Returns:
            Lista de resultados ordenados por relevancia
        """
        if not self.chunks:
            raise ValueError("Índice no construido. Ejecute build_index() primero")
        
        normalized_query = self.normalize_text(query)
        
        # 1. Búsqueda TF-IDF
        query_tfidf = self.vectorizer.transform([normalized_query])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # 2. Búsqueda por embeddings (si está disponible)
        if self.use_embeddings and self.embedding_model and self.embedding_matrix is not None:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            embedding_scores = cosine_similarity(query_embedding, self.embedding_matrix)[0]
            
            # Combinar scores (híbrido)
            combined_scores = alpha * tfidf_scores + (1 - alpha) * embedding_scores
        else:
            # Solo TF-IDF
            combined_scores = tfidf_scores
        
        # 3. Obtener top-k resultados
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        # 4. Formatear resultados
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0:  # Solo resultados con alguna relevancia
                results.append({
                    'text': self.chunks[idx]['text'],
                    'document': self.chunks[idx]['document'],
                    'chunk_id': self.chunks[idx]['chunk_id'],
                    'score': float(combined_scores[idx]),
                    'tfidf_score': float(tfidf_scores[idx]),
                    'embedding_score': float(embedding_scores[idx]) if self.use_embeddings else 0.0
                })
        
        return results
    
    def save(self, filepath: str):
        """Guardar índice en disco"""
        data = {
            'chunks': self.chunks,
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'embedding_matrix': self.embedding_matrix,
            'use_embeddings': self.use_embeddings,
            'model_name': self.embedding_model.model_card_data.model_name if self.embedding_model else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"💾 Índice guardado en: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Cargar índice desde disco"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Recrear instancia
        instance = cls(use_embeddings=data.get('use_embeddings', False))
        instance.chunks = data['chunks']
        instance.vectorizer = data['vectorizer']
        instance.tfidf_matrix = data['tfidf_matrix']
        instance.embedding_matrix = data.get('embedding_matrix')
        
        # Recargar modelo de embeddings si es necesario
        if instance.use_embeddings and data.get('model_name'):
            try:
                instance.embedding_model = SentenceTransformer(data['model_name'])
            except:
                print("⚠️  No se pudo recargar el modelo de embeddings")
                instance.use_embeddings = False
        
        return instance


# ===== FUNCIONES DE UTILIDAD =====

def create_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Dividir texto en fragmentos con overlapping
    
    Args:
        text: Texto completo
        chunk_size: Tamaño de cada fragmento (en caracteres)
        overlap: Solapamiento entre fragmentos
    
    Returns:
        Lista de fragmentos
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Evitar cortar palabras
        if end < text_length:
            last_space = chunk.rfind(' ')
            if last_space > 0:
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if c]


def benchmark_search_methods(search_engine: SearchEngine, test_queries: List[str]):
    """
    Comparar rendimiento de diferentes métodos de búsqueda
    
    Args:
        search_engine: Instancia del motor de búsqueda
        test_queries: Lista de consultas de prueba
    """
    import time
    
    print("\n" + "="*60)
    print("🏁 BENCHMARK DE MÉTODOS DE BÚSQUEDA")
    print("="*60)
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        
        # TF-IDF solo
        start = time.time()
        results_tfidf = search_engine.search(query, alpha=1.0, top_k=3)
        time_tfidf = time.time() - start
        
        print(f"⏱️  TF-IDF: {time_tfidf*1000:.2f}ms")
        print(f"   Top resultado: {results_tfidf[0]['score']:.3f} - {results_tfidf[0]['text'][:100]}...")
        
        # Híbrido (si está disponible)
        if search_engine.use_embeddings:
            start = time.time()
            results_hybrid = search_engine.search(query, alpha=0.7, top_k=3)
            time_hybrid = time.time() - start
            
            print(f"⏱️  Híbrido: {time_hybrid*1000:.2f}ms")
            print(f"   Top resultado: {results_hybrid[0]['score']:.3f} - {results_hybrid[0]['text'][:100]}...")


if __name__ == "__main__":
    # Ejemplo de uso
    print("🔍 Motor de Búsqueda Híbrido - FCyT")
    print(f"GPU disponible: {SearchEngine()._has_gpu()}")
    print(f"Embeddings disponibles: {EMBEDDINGS_AVAILABLE}")
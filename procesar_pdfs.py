"""
procesar_pdfs.py - Script mejorado para procesar documentos PDF
Incluye chunking inteligente y soporte para embeddings
"""

import os
import re
from pathlib import Path
from typing import List, Dict
import pickle

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from search_engine import SearchEngine, create_chunks


# ===== CONFIGURACIÓN =====

DOCS_DIR = Path("docs")
INDEX_FILE = Path("indice_tfidf.pkl")
CHUNK_SIZE = 500  # caracteres por fragmento
OVERLAP = 100     # solapamiento entre fragmentos
USE_EMBEDDINGS = True  # Cambiar a False si no hay GPU o para más velocidad


# ===== FUNCIONES DE PROCESAMIENTO =====

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extraer texto de un PDF con limpieza mejorada
    
    Args:
        pdf_path: Ruta al archivo PDF
    
    Returns:
        Texto extraído y limpio
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        # Limpieza básica
        text = clean_text(text)
        
        return text
    
    except Exception as e:
        print(f"❌ Error al leer {pdf_path.name}: {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Limpiar y normalizar texto extraído de PDF
    
    Args:
        text: Texto crudo
    
    Returns:
        Texto limpio
    """
    # Quitar saltos de línea dentro de párrafos
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Quitar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Quitar caracteres especiales problemáticos
    text = re.sub(r'[^\w\s\náéíóúñÁÉÍÓÚÑ.,;:¿?¡!()\-]', '', text)
    
    return text.strip()


def extract_metadata(text: str, filename: str) -> Dict:
    """
    Extraer metadatos del documento (artículos, capítulos, etc.)
    
    Args:
        text: Texto del documento
        filename: Nombre del archivo
    
    Returns:
        Diccionario con metadatos
    """
    metadata = {
        'filename': filename,
        'has_articles': False,
        'has_chapters': False,
        'word_count': len(text.split()),
        'char_count': len(text)
    }
    
    # Detectar artículos
    article_pattern = r'(?:artículo|art\.|art )\s*\d+'
    articles = re.findall(article_pattern, text.lower())
    if articles:
        metadata['has_articles'] = True
        metadata['article_count'] = len(set(articles))
    
    # Detectar capítulos
    chapter_pattern = r'(?:capítulo|cap\.|cap )\s*\d+'
    chapters = re.findall(chapter_pattern, text.lower())
    if chapters:
        metadata['has_chapters'] = True
        metadata['chapter_count'] = len(set(chapters))
    
    return metadata


def intelligent_chunking(text: str, document_name: str) -> List[Dict]:
    """
    Chunking inteligente que respeta estructura del documento
    
    Args:
        text: Texto completo del documento
        document_name: Nombre del documento
    
    Returns:
        Lista de chunks con metadatos
    """
    chunks_data = []
    
    # Intentar dividir por artículos primero
    article_pattern = r'((?:artículo|art\.|art )\s*\d+[^.]*\.)'
    articles = re.split(article_pattern, text, flags=re.IGNORECASE)
    
    if len(articles) > 3:  # Si hay múltiples artículos
        print(f"   📑 Detectados {len(articles)//2} artículos")
        for i in range(1, len(articles), 2):
            article_title = articles[i]
            article_content = articles[i+1] if i+1 < len(articles) else ""
            
            # Si el artículo es muy largo, dividirlo
            if len(article_content) > CHUNK_SIZE:
                sub_chunks = create_chunks(article_content, CHUNK_SIZE, OVERLAP)
                for j, chunk in enumerate(sub_chunks):
                    chunks_data.append({
                        'text': f"{article_title} {chunk}",
                        'document': document_name,
                        'chunk_id': len(chunks_data),
                        'type': 'article_section'
                    })
            else:
                chunks_data.append({
                    'text': f"{article_title} {article_content}",
                    'document': document_name,
                    'chunk_id': len(chunks_data),
                    'type': 'article'
                })
    else:
        # Chunking estándar con overlapping
        text_chunks = create_chunks(text, CHUNK_SIZE, OVERLAP)
        print(f"   📄 Creados {len(text_chunks)} fragmentos estándar")
        
        for i, chunk in enumerate(text_chunks):
            chunks_data.append({
                'text': chunk,
                'document': document_name,
                'chunk_id': i,
                'type': 'standard'
            })
    
    return chunks_data


def procesar_pdfs() -> Dict:
    """
    Procesar todos los PDFs y generar índice mejorado
    
    Returns:
        Diccionario con estadísticas del procesamiento
    """
    print("="*70)
    print("🚀 PROCESAMIENTO DE DOCUMENTOS PDF - Sistema FCyT")
    print("="*70)
    
    if not DOCS_DIR.exists():
        print(f"❌ Error: No existe la carpeta '{DOCS_DIR}'")
        return {"error": "docs folder not found"}
    
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No se encontraron archivos PDF en '{DOCS_DIR}'")
        return {"error": "no PDFs found"}
    
    print(f"📚 Encontrados {len(pdf_files)} documentos PDF")
    print(f"⚙️  Configuración:")
    print(f"   - Tamaño de chunk: {CHUNK_SIZE} caracteres")
    print(f"   - Overlap: {OVERLAP} caracteres")
    print(f"   - Embeddings: {'✅ Activado' if USE_EMBEDDINGS else '❌ Desactivado'}")
    print()
    
    # Procesar cada PDF
    all_chunks = []
    stats = {
        'total_docs': 0,
        'total_chunks': 0,
        'total_chars': 0,
        'documents': []
    }
    
    for pdf_file in pdf_files:
        print(f"📖 Procesando: {pdf_file.name}")
        
        # Extraer texto
        text = extract_text_from_pdf(pdf_file)
        
        if not text:
            print(f"   ⚠️  Documento vacío o no procesable")
            continue
        
        # Extraer metadatos
        metadata = extract_metadata(text, pdf_file.name)
        print(f"   ℹ️  {metadata['word_count']} palabras, {metadata['char_count']} caracteres")
        
        if metadata['has_articles']:
            print(f"   📋 Contiene {metadata.get('article_count', 0)} artículos")
        
        # Crear chunks inteligentes
        doc_chunks = intelligent_chunking(text, pdf_file.name)
        all_chunks.extend(doc_chunks)
        
        # Actualizar estadísticas
        stats['total_docs'] += 1
        stats['total_chunks'] += len(doc_chunks)
        stats['total_chars'] += metadata['char_count']
        stats['documents'].append({
            'name': pdf_file.name,
            'chunks': len(doc_chunks),
            'words': metadata['word_count'],
            'metadata': metadata
        })
        
        print(f"   ✅ Generados {len(doc_chunks)} fragmentos\n")
    
    # Construir índice de búsqueda
    print("="*70)
    print("🔨 CONSTRUYENDO ÍNDICE DE BÚSQUEDA")
    print("="*70)
    
    search_engine = SearchEngine(use_embeddings=USE_EMBEDDINGS)
    search_engine.build_index(all_chunks)
    
    # Guardar índice
    search_engine.save(INDEX_FILE)
    
    # Resumen final
    print("\n" + "="*70)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("="*70)
    print(f"📊 Estadísticas:")
    print(f"   - Documentos procesados: {stats['total_docs']}")
    print(f"   - Total de fragmentos: {stats['total_chunks']}")
    print(f"   - Total de caracteres: {stats['total_chars']:,}")
    print(f"   - Promedio fragmentos/doc: {stats['total_chunks']/stats['total_docs']:.1f}")
    print(f"   - Archivo índice: {INDEX_FILE}")
    print()
    
    # Detalle por documento
    print("📋 Detalle por documento:")
    for doc in stats['documents']:
        print(f"   • {doc['name']}: {doc['chunks']} fragmentos ({doc['words']} palabras)")
    
    print("\n🎉 ¡Sistema listo para búsquedas!\n")
    
    return stats


def test_search_engine():
    """Probar el motor de búsqueda con consultas de ejemplo"""
    
    if not INDEX_FILE.exists():
        print("❌ No existe índice. Ejecute primero: python procesar_pdfs.py")
        return
    
    print("\n" + "="*70)
    print("🧪 PRUEBA DEL MOTOR DE BÚSQUEDA")
    print("="*70)
    
    # Cargar índice
    search_engine = SearchEngine.load(INDEX_FILE)
    
    # Consultas de prueba
    test_queries = [
        "función del docente en PFG",
        "requisitos para proyecto final",
        "reglamento de investigación",
        "evaluación de trabajos académicos"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Consulta: '{query}'")
        print("-" * 70)
        
        results = search_engine.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Relevancia: {result['score']:.3f} | {result['document']}")
            print(f"   {result['text'][:200]}...")
    
    print("\n" + "="*70)


# ===== EJECUCIÓN PRINCIPAL =====

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Modo prueba
        test_search_engine()
    else:
        # Modo procesamiento
        result = procesar_pdfs()
        
        # Ofrecer ejecutar pruebas
        if result.get('total_docs', 0) > 0:
            print("\n💡 Para probar el sistema, ejecute: python procesar_pdfs.py --test")
            print("💡 Para iniciar el servidor: uvicorn app:app --reload --port 8000")
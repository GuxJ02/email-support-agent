# query_email.py

import os
import json
import requests
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup

# — Prompts para análisis y generación de emails/incidencias —

PROMPT_ANALISIS_INCIDENCIA = """
Se te va a proporcionar un email que contenga tanto un asunto como el cuerpo del email, tu objetivo principal va a ser
determinar respondiendo con una única palabra: si consideras que ese email es una incidencia, responde exactamente con "incidencia". En caso contrario, responde exactamente con "no incidencia".

Email proporcionado:
{email}
"""

PROMPT_EXTRACCION_ASUNTOS = """
Recibirás un bloque con emails de contexto (asunto + cuerpo) y un email de incidencia. De todos los emails de contexto, responde solo con los asuntos de aquellos que consideres más útiles para resolver la nueva incidencia. Lista los asuntos más relevantes, uno por línea.

Emails de contexto:
{context}

Email de incidencia:
{email}
"""

PROMPT_TEMPLATE_EMAIL = """
<|SYSTEM|>
Eres **SoporteAI**, un agente de soporte técnico experto. Se te proporcionará un email de incidencia y un contexto histórico de incidencias resueltas. Debes generar **solo** el borrador que se enviará al cliente, con exactamente esta estructura, IMPORTANTE: las partes que no estén marcadas entre < > es obligatorio que las pongas literales:

---
Asunto: Análisis de incidencia:
 <ASUNTO DE LA NUEVA INCIDENCIA QUE HAS RCIBIDO EN EL APARTADO Nueva incidencia a resolver (asunto + cuerpo)>
Cuerpo:
1. Resumen de la incidencia entrante:
 <Breve resumen de la incidencia que se va a resolver>

2. Pasos a seguir para resolver la incidencia basados en el contexto histórico:
<Lista de pasos detallados para resolver la incidencia, basados en el contexto histórico proporcionado>
---
A continuación se proporcionara el contexto y la nueva incidencia a resolver:
Contexto histórico proporcionado:
{context}

Nueva incidencia a resolver (asunto + cuerpo):
{question}
"""

# — Configuración del cross-encoder para reranking —
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model     = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def rerank_candidates(query: str, candidates: list) -> list:
    scores = []
    for cand in candidates:
        inputs = tokenizer.encode_plus(query, cand,
                                       truncation=True,
                                       padding=True,
                                       return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        scores.append(logits.item())
    # Devuelve los índices ordenados de mayor a menor score
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

def ollama_completion(prompt: str) -> str:
    data = {"model": "mistral", "prompt": prompt}
    resp = requests.post("http://localhost:11434/api/generate",
                         json=data, stream=True, timeout=30)
    resp.raise_for_status()
    buf = ""
    for chunk in resp.iter_content(1024):
        txt = chunk.decode("utf-8")
        try:
            buf += json.loads(txt).get("response", "")
        except json.JSONDecodeError:
            buf += txt
    # Cortamos cualquier token sobrante de sistema
    return buf.split('{"model":')[0].strip()

def query_rag_email(query_text: str):
    """
    1) Determina si el email es una incidencia.
    2) Si no lo es, devuelve inmediatamente.
    3) Si lo es, recupera contexto, extrae asuntos, genera borrador y lo guarda en salida.txt.
    Devuelve: (respuesta_completa, asunto_extraído, cuerpo_extraído, asuntos_importantes)
    """
    # Inicializo variables de salida
    asuntos_importantes = ""
    asunto = ""
    cuerpo = ""

    # 1) Detección de incidencia
    prompt1 = ChatPromptTemplate.from_template(PROMPT_ANALISIS_INCIDENCIA).format(
        email=query_text
    )
    respuesta = ollama_completion(prompt1).strip()
    clave = respuesta.lower().rstrip(".")

    if clave != "incidencia":
        # No es incidencia: salgo sin más
        return respuesta, asunto, cuerpo, asuntos_importantes

    # 2) RAG: recupero contexto
    db = Chroma(
        persist_directory="chromaDB",
        embedding_function=get_embedding_function()
    )
    results = db.similarity_search_with_score(query_text, k=8)
    if not results:
        return "⚠️ No se encontraron incidencias similares.", asunto, cuerpo, asuntos_importantes

    texts = [doc.page_content for doc, _ in results]
    top5_idxs = rerank_candidates(query_text, texts)[:5]
    context = "\n\n---\n\n".join(texts[i] for i in top5_idxs)

    # 3) Extraer asuntos importantes
    prompt2 = ChatPromptTemplate.from_template(PROMPT_EXTRACCION_ASUNTOS).format(
        context=context,
        email=query_text
    )
    asuntos_importantes = ollama_completion(prompt2)

    # 4) Generar borrador de respuesta
    prompt3 = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_EMAIL).format(
        context=context,
        question=query_text
    )
    borrador = ollama_completion(prompt3)

    # 5) Guardar en salida.txt
    with open("salida.txt", "w", encoding="utf-8") as f:
        f.write(borrador)
        f.write("\n\n=== Asuntos utilizados del historico ===\n")
        f.write(asuntos_importantes)
        f.write("\n=== Fin de asuntos utilizados del historico ===\n")

    # 6) Extraer asunto y cuerpo del borrador
    asunto, cuerpo = leer_email("salida.txt")
    return borrador, asunto, cuerpo, asuntos_importantes

def clean_email(raw_body: str) -> str:
    """Convierte HTML a texto y elimina líneas de cabecera redundantes."""
    if "<html" in raw_body.lower():
        raw_body = BeautifulSoup(raw_body, "html.parser").get_text(" ", strip=True)
    lines = [
        ln for ln in raw_body.splitlines()
        if not ln.lower().startswith(("de:", "from:", "el", "-----original"))
    ]
    return "\n".join(lines).strip()

def leer_email(ruta_archivo: str):
    """
    Lee salida.txt y devuelve:
      - asunto: texto tras 'Asunto:' (maneja sujeto en la misma línea o en la siguiente)
      - cuerpo: TODO lo que hay tras 'Cuerpo:' hasta el final (incluye Asuntos importantes)
    """
    asunto = None
    cuerpo_lines = []
    in_body = False
    # flag para detectar sujeto multilinea
    esperando_sujeto = False

    with open(ruta_archivo, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            stripped = line.strip()

            # 1) Si estamos esperando la siguiente línea para el sujeto
            if esperando_sujeto and stripped:
                asunto = stripped
                esperando_sujeto = False
                continue

            # 2) Extraer asunto en la misma línea
            if asunto is None and stripped.startswith("Asunto:"):
                resto = stripped[len("Asunto:"):].strip()
                if resto and not resto.endswith(":"):
                    # caso: Asunto: texto completo en la misma línea
                    asunto = resto
                else:
                    # caso: línea de “Asunto:” sin texto o termina en “:”
                    esperando_sujeto = True
                continue

            # 3) Detectar inicio de cuerpo
            if stripped.startswith("Cuerpo:"):
                in_body = True
                continue

            # 4) Si estamos en cuerpo, acumular TODAS las líneas
            if in_body:
                cuerpo_lines.append(line)

    cuerpo = "\n".join(cuerpo_lines).strip()
    return asunto, cuerpo


if __name__ == "__main__":
    # Prueba rápida leyendo lo guardado en salida.txt
    asu, cue = leer_email("salida.txt")
    print("Asunto:", asu)
    print("\nCuerpo:\n", cue)

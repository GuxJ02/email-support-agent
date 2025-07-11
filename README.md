# 📧 email-support-agent  
*Email-support-agent* es un agente autónomo que monitoriza tu bandeja de entrada de **Gmail** en tiempo real, identifica automáticamente correos que contienen **incidencias técnicas** y genera un borrador de respuesta utilizando un **modelo LLM local** (vía [Ollama](https://ollama.com/)) y técnicas de **RAG (Retrieval-Augmented Generation)**.  
El sistema consulta una base de datos vectorial de incidencias anteriores (indexada con **ChromaDB** y embebida con **Sentence Transformers**) para proponer soluciones basadas en casos previos, y finalmente reenvía el borrador al correo que tú determines.

Tecnologías clave: `Python`, `LangChain`, `RAG`, `Ollama`, `ChromaDB`, `IMAPClient`, `transformers`, `hugging-face`.


---

## ⚙️ Requisitos rápidos

| Necesitas | Detalles |
|-----------|----------|
| **Python** | 3.10 o superior |
| **Ollama** | Servidor local de LLM (`ollama serve`) |
| **Cuenta Gmail** | IMAP habilitado + **App Password** |
| **CUDA 11.8** *(opc.)* | Para acelerar PyTorch con GPU NVIDIA |

---

## 🧠 Modelo LLM local (Ollama)

Este proyecto llama a **Ollama** en `http://localhost:11434`.  
Sigue estos pasos una sola vez:

```bash
# Instala Ollama (macOS / Linux / Windows)
 https://ollama.com/  # Descargalo desde la propia web

# Este proyecto utiliza el modelo Mistral pero se puede cambiar
# a gusto del usuario (descarga mistral para poder utilizarlo sin cambiar nada)
ollama pull mistral
```
---

## 🔑 Variables de entorno (.env.example)
 Modifica estas variables para adaptarlas a tus correos
| Variable        | Descripción                                 |
| --------------- | ------------------------------------------  |
| `GMAIL_USER`    | Dirección de email que recibira incidencias |
| `GMAIL_APP_PWD` | App Password de 16 caracteres(`GMAIL_USER`) |
| `EMAIL_DEST`    | Destino al que reenviar los reportes        |

---

## 🚀 Instalación & primer arranque

```bash
# 1. Clona el repositorio
git clone https://github.com/GuxJ02/email-support-agent.git
cd email-support-agent

# 2. Crea y activa un entorno virtual
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows PowerShell
venv\Scripts\activate

# 3. Instala dependencias
pip install -r requirements.txt

# 4. Copia la plantilla de variables y edítala
cp .env.example .env
# → pon GMAIL_USER, GMAIL_APP_PWD y DEST_EMAIL a tu gusto

# 5. Indexa tus e-mails históricos (Data/*.txt) en ChromaDB, dentro de la carpeta Datos hay un *.txt de ejemplo con emails de ejemplo,
# puedes cambiarlo a tu gusto por un *.txt propio y ejecutar la siguiente sentencia para indexar los datos en la BBDDV
python insertarBD.py --reset

# 6. Arranca el listener de Gmail
python gmail_listener_email.py
```
Una vez iniciado el listener ya podrás enviar un mensaje que en caso de considerarse una incidencia pasará al llm para analizarlo 
y mandar un reporte al correo indicad.

---

## ⚡️ Acelerar con GPU (opcional)

Si dispones de una GPU NVIDIA, puedes instalar la build CUDA 11.8 de PyTorch para multiplicar la velocidad de inferencia:

1. **Desinstalar la versión CPU-only**  
   ```bash
   pip uninstall -y torch torchvision torchaudio
   ```
2. **Instalar la build CUDA 11.8**  
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Comprobar que PyTorch detecta la GPU (Opcional)**  
   ```bash
   python -c "import torch; print('torch:', torch.__version__); print('CUDA version:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
   ```
   

---




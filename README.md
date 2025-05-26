# 📧 email-support-agent  
*Agente RAG que detecta incidencias en tu bandeja de Gmail, genera un borrador con LLM y lo reenvía al destinatario que elijas.*

---

## ⚙️ Requisitos rápidos

| Necesitas | Detalles |
|-----------|----------|
| **Python** | 3.10 o superior |
| **Cuenta Gmail** | Con IMAP habilitado y **App Password** |
| **CUDA 11.8** *(opcional)* | Para acelerar con GPU NVIDIA |

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

# 5. Indexa tus e-mails históricos (Data/*.txt) en ChromaDB
python insertarBD.py --reset

# 6. Arranca el listener de Gmail
python gmail_listener_email.py
```
---

## 🔑 Variables de entorno

| Variable        | Descripción                            |
| --------------- | -------------------------------------- |
| `GMAIL_USER`    | Tu dirección Gmail                     |
| `GMAIL_APP_PWD` | App Password de 16 caracteres          |
| `DEST_EMAIL`    | Destino al que reenviar los borradores |

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




# gmail_listener_email.py
from dotenv import load_dotenv
load_dotenv()
import os
import email
import smtplib
from email.message import EmailMessage
from imapclient import IMAPClient
from query_email import clean_email, query_rag_email

HOST       = 'imap.gmail.com'
USER       = os.environ.get('GMAIL_USER')
PASS       = os.environ.get('GMAIL_APP_PWD')
DEST_EMAIL = os.environ.get('GMAIL_DESTINATARIO') # Este email es donde se reenvían los resultados de los analisis de las incidencias

def idle_listener():
    with IMAPClient(HOST) as client:
        client.login(USER, PASS)
        client.select_folder('INBOX')
        existing_uids = client.search(['ALL'])
        last_uid = max(existing_uids) if existing_uids else 0

        print(f"▶️  Conectado a Gmail IMAP; último UID inicial: {last_uid}")
        while True:
            client.idle()
            events = client.idle_check(timeout=60)
            client.idle_done()
            if not events:
                continue

            all_uids = client.search(['ALL'])
            new_uids = [uid for uid in all_uids if uid > last_uid]
            if not new_uids:
                continue

            for uid in sorted(new_uids):
                raw = client.fetch([uid], ['RFC822'])[uid][b'RFC822']
                msg = email.message_from_bytes(raw)

                subj_orig = msg.get('Subject', '(sin asunto)')
                print(f"\n--- Nuevo e-mail (UID={uid}) ---")
                print(f"Asunto original: {subj_orig}")

                # extraigo body text/plain
                body = ""
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break

                texto = clean_email(body)
                borrador, asunto, cuerpo, _ = query_rag_email(texto)

                if not asunto:
                    print("→ No es incidencia, no envío nada.")
                    last_uid = max(all_uids)
                    continue

                # Construyo y envío email final
                msg_out = EmailMessage()
                msg_out["From"]    = USER
                msg_out["To"]      = DEST_EMAIL
                msg_out["Subject"] = asunto
                msg_out.set_content(cuerpo)

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login(USER, PASS)
                    smtp.send_message(msg_out)

                print(f"✉️  Incidencia reenviada a {DEST_EMAIL}: {asunto}")

            last_uid = max(all_uids)

if __name__ == '__main__':
    idle_listener()

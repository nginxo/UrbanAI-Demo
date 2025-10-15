import os
import json
from google import genai
from typing import List, Dict

class GeminiChat:
    def __init__(self, api_key: str, context_file: str = "context.txt"):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        self.context = self._load_context(context_file)
        self.conversation_history = []
        self.chat = None
        self._initialize_chat()
    
    def _load_context(self, context_file: str) -> str:
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                context = f.read()
            print(f"✓ Contesto caricato da {context_file}")
            return context
        except FileNotFoundError:
            print(f"⚠ File {context_file} non trovato, uso contesto vuoto")
            return "Sei un assistente AI utile e cordiale."
    
    def _initialize_chat(self):
        config = {
            "system_instruction": self.context,
            "temperature": 0.7,
        }
        self.chat = self.client.chats.create(
            model=self.model,
            config=config
        )
        print(f"✓ Chat UrbanAI inizializzata con modello {self.model}")
    
    def send_message(self, user_input: str) -> str:
        try:
            response = self.chat.send_message(user_input)
            ai_response = response.text
            
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            self.conversation_history.append({
                "role": "model",
                "content": ai_response
            })
            
            return ai_response
            
        except Exception as e:
            return f"Errore nella comunicazione con Gemini: {str(e)}"
    
    def get_history(self) -> List[Dict]:
        try:
            history = []
            for message in self.chat.get_history():
                history.append({
                    "role": message.role,
                    "content": message.parts[0].text if message.parts else ""
                })
            return history
        except:
            return self.conversation_history
    
    def save_conversation(self, filename: str = "conversation_log.json"):
        history = self.get_history()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "model": self.model,
                "context": self.context,
                "messages": history
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Conversazione salvata in {filename}")
    
    def reset_chat(self):
        self._initialize_chat()
        self.conversation_history = []
        print("✓ Chat resettata")


def main():
    print("=" * 60)
    print("  UrbanAI (test con Gemini)")
    print("=" * 60)
    
    # Carica la API key da variabile d'ambiente
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠ GEMINI_API_KEY non trovata nelle variabili d'ambiente")
        api_key = input("Inserisci la tua API key Gemini: ").strip()
    
    # Inizializza il chatbot
    try:
        chatbot = GeminiChat(api_key=api_key, context_file="context.txt")
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione: {e}")
        return
    
    print("\n📝 Comandi disponibili:")
    print("  - 'esci' o 'quit' per terminare")
    print("  - 'salva' per salvare la conversazione")
    print("  - 'reset' per resettare la chat")
    print("  - 'history' per vedere la cronologia")
    print("=" * 60)
    
    # Loop principale di conversazione
    while True:
        try:
            # Input utente
            print("\n" + "─" * 60)
            user_input = input("👤 Tu: ").strip()
            
            # Gestione comandi speciali
            if user_input.lower() in ['esci', 'quit', 'exit']:
                print("\n🍃 UrbanAI: Arrivederci! 👋")
                chatbot.save_conversation()
                break
            
            elif user_input.lower() == 'salva':
                chatbot.save_conversation()
                continue
            
            elif user_input.lower() == 'reset':
                chatbot.reset_chat()
                continue
            
            elif user_input.lower() == 'history':
                history = chatbot.get_history()
                print("\n📜 Cronologia conversazione:")
                for msg in history:
                    role = "👤 Tu" if msg['role'] == 'user' else "🍃 UrbanAI"
                    print(f"\n{role}: {msg['content'][:100]}...")
                continue
            
            elif not user_input:
                print("⚠ Inserisci un messaggio valido")
                continue
            
            # Invia il messaggio e ricevi risposta
            print("\n🍃 UrbanAI: ", end="", flush=True)
            response = chatbot.send_message(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n🍃 UrbanAI: Interruzione ricevuta. Salvataggio in corso...")
            chatbot.save_conversation()
            print("Arrivederci! 👋")
            break
        except Exception as e:
            print(f"\n❌ Errore: {e}")
            continue


if __name__ == "__main__":
    main()

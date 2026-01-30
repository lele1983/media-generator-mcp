# Media Generator MCP

Un server MCP (Model Context Protocol) per generare immagini e video utilizzando AI all'avanguardia.

## Funzionalita

- **Generazione Immagini**: Crea immagini da descrizioni testuali usando Nano Banana Pro (Gemini 3 Pro Image Preview) via OpenRouter
- **Editing Immagini**: Modifica immagini esistenti con istruzioni in linguaggio naturale
- **Animazione Video**: Trasforma immagini statiche in video con Kling 2.6 Pro via fal.ai
- **Workflow Completo**: Genera video direttamente da testo (text → image → video)

## Modelli Utilizzati

| Funzione | Provider | Modello |
|----------|----------|---------|
| Generazione/Editing Immagini | OpenRouter | `google/gemini-3-pro-image-preview` (Nano Banana Pro) |
| Animazione Video | fal.ai | `fal-ai/kling-video/v2.6/pro/image-to-video` |

## Requisiti

- Python 3.10+
- Account [OpenRouter](https://openrouter.ai) con API key
- Account [fal.ai](https://fal.ai) con API key

## Installazione

### 1. Clona o scarica il progetto

```bash
cd /path/to/media-generator-mcp
```

### 2. Crea ambiente virtuale (consigliato)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# oppure
.\venv\Scripts\activate   # Windows
```

### 3. Installa dipendenze

```bash
pip install -r requirements.txt
```

### 4. Configura le API keys

```bash
cp .env.example .env
# Modifica .env con le tue chiavi API
```

Oppure esporta le variabili d'ambiente:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export FAL_KEY="..."
```

## Configurazione Claude Desktop

Aggiungi al tuo `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "media-generator": {
      "command": "python",
      "args": ["/path/to/media-generator-mcp/server.py"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-...",
        "FAL_KEY": "..."
      }
    }
  }
}
```

### Con uv (alternativa)

```json
{
  "mcpServers": {
    "media-generator": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/media-generator-mcp",
        "run", "server.py"
      ],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-...",
        "FAL_KEY": "..."
      }
    }
  }
}
```

## Tools Disponibili

### 1. `image_generate` - Genera Immagine da Testo

Crea un'immagine da una descrizione testuale dettagliata.

**Parametri:**
- `prompt` (required): Descrizione dell'immagine
- `aspect_ratio`: `1:1`, `16:9`, `9:16`, `4:3`, `21:9` (default: `1:1`)
- `image_size`: `1K`, `2K`, `4K` (default: `2K`)
- `style_prompt`: Istruzioni di stile aggiuntive

**Esempio:**
```
Genera un'immagine di una citta futuristica al tramonto con grattacieli di vetro e macchine volanti
```

### 2. `image_edit` - Modifica Immagine

Modifica un'immagine esistente seguendo istruzioni testuali.

**Parametri:**
- `image_url` o `image_base64` (required): Immagine da modificare
- `edit_instructions` (required): Cosa modificare
- `aspect_ratio`, `image_size`: Come sopra

**Esempio:**
```
Modifica l'immagine cambiando il cielo in un tramonto drammatico con colori arancioni e viola
```

### 3. `video_animate` - Anima Immagine in Video

Trasforma un'immagine statica in un video animato.

**Parametri:**
- `image_url` (required): URL dell'immagine da animare
- `prompt` (required): Descrizione del movimento desiderato
- `duration`: `5` o `10` secondi (default: `5`)
- `generate_audio`: Genera audio sincronizzato (default: `true`)
- `negative_prompt`: Cosa evitare

**Esempio:**
```
Anima questa immagine con un lento zoom della camera, nuvole che si muovono nel cielo
```

### 4. `media_generate` - Workflow Completo Text-to-Video

Genera un video completo partendo solo da testo.

**Parametri:**
- `prompt` (required): Descrizione della scena
- `animation_prompt`: Istruzioni specifiche per l'animazione
- `duration`: `5` o `10` secondi
- `generate_audio`: Genera audio
- `aspect_ratio`, `image_size`, `style_prompt`

**Esempio:**
```
Crea un video di un gatto seduto su un davanzale che guarda la pioggia fuori dalla finestra.
Il gatto gira lentamente la testa e sbatte le palpebre.
```

## Costi Stimati

| Operazione | Costo Approssimativo |
|------------|---------------------|
| Generazione immagine | ~$0.01-0.05 |
| Video 5s senza audio | ~$0.35 |
| Video 5s con audio | ~$0.70 |
| Video 10s senza audio | ~$0.70 |
| Video 10s con audio | ~$1.40 |

## Output

### Immagini
Le immagini vengono restituite in formato **base64 PNG**. Puoi salvarle decodificando il base64:

```python
import base64
with open("image.png", "wb") as f:
    f.write(base64.b64decode(image_base64))
```

### Video
I video vengono restituiti come **URL diretti** al file MP4 (ospitati su fal.ai). Gli URL hanno una scadenza temporale.

## Troubleshooting

### Errore "OPENROUTER_API_KEY not found"
Assicurati di aver configurato la variabile d'ambiente o il file `.env`.

### Errore "FAL_KEY not found"
Verifica la configurazione della chiave fal.ai.

### Errore 401/403
Le API keys potrebbero essere invalide o non avere i permessi necessari.

### Errore 402
Crediti insufficienti. Ricarica il tuo account OpenRouter o fal.ai.

### Timeout durante generazione video
I video possono richiedere 30-120 secondi. Se il timeout persiste, prova:
- Ridurre la durata a 5 secondi
- Disabilitare l'audio
- Usare un'immagine piu semplice

## Licenza

MIT License

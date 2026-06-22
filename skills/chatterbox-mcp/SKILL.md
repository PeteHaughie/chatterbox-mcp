---
name: chatterbox-mcp
description: Text-to-speech with voice cloning via the chatterbox-mcp MCP server. Use when the user wants to generate speech audio, clone a voice from a sample, adjust voice personality (emotion, exaggeration, speed), or create text-to-speech content with a specific voice.
---

# Chatterbox TTS: Voice Cloning MCP Server

An MCP tool called **`tts`** and a streaming variant **`tts_stream_start`** (on the `chatterbox_mcp` server) are available for generating speech from text with cloned voices. Additional tools let you manage voice profiles.

> Do NOT run shell commands for TTS generation — use the `tts`, `tts_stream_start`, `create_voice_profile`, `list_voice_profiles`, and `delete_voice_profile` tools directly.
>
> **If the MCP server fails to start** — run `pip install -e /Users/petehaughie/Projects/chatterbox` then `pip install -e /Users/petehaughie/Projects/chatterbox-mcp` to install the local fork with MLX support (the PyPI version of `chatterbox-tts` doesn't include the required `t3_mlx` module).

## Tools

### `tts` — Generate speech to a WAV file

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `text` | `str` | required | — | Text to speak. Will be auto-capitalized and punctuation-normalized. |
| `voice` | `str` | required | — | Voice profile name (list available with `list_voice_profiles`) |
| `emotion_adv` | `float` | `0.5` | `0.0`–`1.0` | Exaggeration/emotion level. `0.0` = flat/monotone, `1.0` = very expressive |
| `temperature` | `float` | `0.8` | `0.1`–`1.5` | Sampling randomness. Lower = consistent, higher = creative |
| `top_p` | `float` | `0.95` | `0.5`–`1.0` | Nucleus sampling threshold |
| `repetition_penalty` | `float` | `1.2` | `1.0`–`1.5` | Penalty for repeated tokens |
| `max_tokens` | `int` | `500` | `1`–`2000` | Max speech tokens (~25 tokens/sec). `500` ≈ 20 seconds |

Returns: `{"path": "/path/to/output.wav", "duration_seconds": 3.2, "sample_rate": 24000, "voice": "alice"}`

### `tts_stream_start` — Start a realtime audio stream

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `text` | `str` | required | — | Text to speak |
| `voice` | `str` | required | — | Voice profile name |
| `emotion_adv` | `float` | `0.5` | `0.0`–`1.0` | Exaggeration level |
| `temperature` | `float` | `0.8` | `0.1`–`1.5` | Sampling randomness |
| `top_p` | `float` | `0.95` | `0.5`–`1.0` | Nucleus sampling threshold |
| `repetition_penalty` | `float` | `1.2` | `1.0`–`1.5` | Repetition penalty |
| `max_tokens` | `int` | `500` | `1`–`2000` | Max speech tokens |
| `chunk_tokens` | `int` | `25` | — | Tokens per decode chunk (~1 second of audio each) |

Returns:
```json
{"stream_url": "http://127.0.0.1:53425/stream/a1b2c3d4", "session_id": "a1b2c3d4"}
```

The port is assigned dynamically at startup — no pre-configuration needed. The `stream_url` is internal to `127.0.0.1` and is consumed by piping the SSE stream to an audio player on the same machine. The agent reads the `stream_url` from the response and uses it directly.

**How it works**: the server generates all speech tokens on the main thread (MLX), then decodes audio chunks on a background thread (PyTorch). Each chunk (~1s of audio) is pushed as an SSE event. The stream ends with `event: end`. Audio format is raw PCM s16le at 24kHz, base64-encoded in the SSE data field.

### `create_voice_profile` — Clone a voice from reference audio

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique profile name (e.g. `"alice"`, `"john_smith"`) |
| `reference_audio` | `str` | required | Path to WAV/MP3 file of speaker's voice |
| `description` | `str` | `""` | Optional notes about this voice |
| `emotion_adv` | `float` | `0.5` | Default exaggeration level for this voice |

Best results: **10–30 seconds of clean speech** with no background noise or music.

### `list_voice_profiles` — List cloned voices

No parameters. Returns JSON array of profile metadata.

### `delete_voice_profile` — Remove a voice profile

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Profile name to delete |

## Streaming Playback

Pipe the SSE stream to an audio player for realtime playback. Two options:

### Option A: Python consumer (recommended for assistant integration)

Save as `play_tts.py`:

```python
#!/usr/bin/env python3
"""Pipe TTS SSE stream to audio output."""
import sys, json, base64, subprocess as sp, httpx

url = sys.argv[1]
proc = sp.Popen(["play", "-r", "24000", "-b", "16", "-c", "1",
                 "-e", "signed", "-t", "raw", "-"],
                stdin=sp.PIPE)

with httpx.Client() as c:
    with c.stream("GET", url) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "data" in data:
                    proc.stdin.write(base64.b64decode(data["data"]))
                elif data.get("status") == "complete":
                    break
proc.stdin.close()
proc.wait()
```

Usage: `python3 play_tts.py "http://127.0.0.1:PORT/stream/SESSION_ID"`

Requires `httpx` and SoX (`brew install sox`).

### Option B: curl + player (minimal deps)

```bash
curl -sN "$STREAM_URL" | while IFS= read -r line; do
  case "$line" in
    data:\ * )
      data="${line#data: }"
      echo "$data" | python3 -c "
import sys, base64
sys.stdout.buffer.write(base64.b64decode(sys.stdin.read().strip()))
" | play -q -r 24000 -b 16 -c 1 -e signed -t raw -
      ;;
  esac
done
```

Note: Option B spawns a Python process per chunk and may have audible gaps. Option A is preferred.

## Voice Personality & Expression

The most important factor is **reference audio quality**. These parameters shape output personality:

### Emotion / Exaggeration (`emotion_adv`)

| Value | Effect | Use case |
|---|---|---|
| `0.0`–`0.3` | Flat, monotone | Narration, technical content, calming voices |
| `0.4`–`0.6` | Natural, conversational | Default for most voices |
| `0.7`–`0.9` | Expressive, lively | Storytelling, engaging content |
| `1.0` | Maximum exaggeration | Character voices, dramatic effect |

### Temperature (`temperature`)

| Value | Effect |
|---|---|
| `0.1`–`0.4` | Deterministic, consistent — good for brand voices |
| `0.6`–`0.9` | Natural variation — sweet spot is default `0.8` |
| `1.0`–`1.5` | Creative, unpredictable — experimental use |

### Repetition Penalty (`repetition_penalty`)

| Value | Effect |
|---|---|
| `1.0` | No penalty — may loop on short text |
| `1.1`–`1.2` | Mild penalty (default `1.2`, good balance) |
| `1.3`–`1.5` | Strong penalty — can suppress natural repeats |

### Reference Audio Guidelines

1. **Clean recording**: no background noise, music, or reverb
2. **Consistent volume**: avoid clipping or very quiet sections
3. **Natural speech**: conversational tone works better than reading
4. **Duration**: 10–30 seconds; >= 15 seconds preferred
5. **Variety**: natural pitch range and pauses help capture prosodic patterns

### Quick Personality Presets

| Personality | `emotion_adv` | `temperature` | `repetition_penalty` | `top_p` |
|---|---|---|---|---|
| Calm narrator | `0.2` | `0.4` | `1.2` | `0.9` |
| Default | `0.5` | `0.8` | `1.2` | `0.95` |
| Energetic | `0.8` | `0.9` | `1.1` | `0.95` |
| Dramatic | `1.0` | `0.7` | `1.2` | `0.9` |
| Whisper/soft | `0.1` | `0.3` | `1.3` | `0.8` |

## Example Sessions

### File-based generation

```
User: create a voice profile named "alice" from ~/samples/alice_conversation.wav
Agent: calls create_voice_profile(name="alice", reference_audio="~/samples/alice_conversation.wav")
       → Profile created

User: have alice say "Welcome to the future of voice cloning" with a calm persona
Agent: calls tts(text="Welcome to the future of voice cloning", voice="alice",
                 emotion_adv=0.2, temperature=0.4, repetition_penalty=1.2)
       → {"path": "~/.chatterbox-mcp/output/alice_abc12345.wav", "duration_seconds": 3.2, ...}
```

### Streaming (realtime assistant voice)

```
User: use the lcars voice for your responses
Agent: calls tts_stream_start(voice="lcars", text="Hello, I am online and ready to assist.")
       → {"stream_url": "http://127.0.0.1:53425/stream/a1b2c3d4", "session_id": "a1b2c3d4"}
Agent: spawns python3 play_tts.py "http://127.0.0.1:53425/stream/a1b2c3d4"
       → Audio plays through speakers as chunks arrive

User: (asks a question)
Agent: calls tts_stream_start(voice="lcars", text="Let me think about that...")
       → {"stream_url": "http://127.0.0.1:53425/stream/e5f6g7h8", "session_id": "e5f6g7h8"}
Agent: spawns python3 play_tts.py "http://127.0.0.1:53425/stream/e5f6g7h8"
       → Audio plays through speakers
```

Each `tts_stream_start` call creates a new independent session. The agent should spawn a fresh player process per utterance. The SSE server is persistent — the first call starts it, subsequent calls reuse it.

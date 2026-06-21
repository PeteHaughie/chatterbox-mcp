---
name: chatterbox-mcp
description: Text-to-speech with voice cloning via the chatterbox-mcp MCP server. Use when the user wants to generate speech audio, clone a voice from a sample, adjust voice personality (emotion, exaggeration, speed), or create text-to-speech content with a specific voice.
---

# Chatterbox TTS: Voice Cloning MCP Server

An MCP tool called **`tts`** (on the `chatterbox_mcp` server) is available for generating speech from text with cloned voices. Additional tools let you manage voice profiles.

> Do NOT run shell commands for TTS generation — use the `tts`, `create_voice_profile`, `list_voice_profiles`, and `delete_voice_profile` tools directly.

## Tools

### `tts` — Generate speech from text

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

## Example Session

```
User: create a voice profile named "alice" from ~/samples/alice_conversation.wav
Agent: calls create_voice_profile(name="alice", reference_audio="~/samples/alice_conversation.wav")
       → Profile created

User: have alice say "Welcome to the future of voice cloning" with a calm persona
Agent: calls tts(text="Welcome to the future of voice cloning", voice="alice",
                 emotion_adv=0.2, temperature=0.4, repetition_penalty=1.2)
       → {"path": "~/.chatterbox-mcp/output/alice_abc12345.wav", "duration_seconds": 3.2, ...}
```

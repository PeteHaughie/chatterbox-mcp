"""MCP server for Chatterbox TTS — voice cloning and generation."""

import json
import uuid
import base64
import asyncio
import threading
from pathlib import Path
from contextlib import asynccontextmanager

import soundfile as sf

from mcp.server.fastmcp import FastMCP

from .model import ChatterboxModel
from .voices import VoiceManager
from .model import S3GEN_SR
from .streaming import StreamManager, StreamSSEServer

OUTPUT_DIR = Path.home() / ".chatterbox-mcp" / "output"


@asynccontextmanager
async def app_lifespan(server):
    model = ChatterboxModel(device="cpu")
    model.load()
    voices = VoiceManager()
    streams = StreamManager()
    sse_server = StreamSSEServer(streams)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield {"model": model, "voices": voices, "streams": streams, "sse_server": sse_server}
    finally:
        sse_server.stop()


mcp = FastMCP("chatterbox_mcp", lifespan=app_lifespan)


@mcp.tool(
    name="tts",
    annotations={
        "title": "Text-to-Speech Generation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def tts(
    text: str = ...,
    voice: str = ...,
    emotion_adv: float = 0.5,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    max_tokens: int = 500,
) -> str:
    """Generate speech audio from text using a cloned voice profile.

    Produces a WAV file of synthesized speech at 24 kHz sample rate.
    The caller should specify an existing voice profile name (created
    beforehand with create_voice_profile). Emotion and sampling parameters
    control the expressiveness and naturalness of the output.

    Args:
        text (str): The text to speak (required).
        voice (str): Name of the voice profile to use (required, created
          via create_voice_profile).
        emotion_adv (float): Exaggeration/emotion level from 0.0 (flat)
          to 1.0 (expressive). Default 0.5.
        temperature (float): Sampling temperature from 0.1 (deterministic)
          to 1.5 (creative). Default 0.8.
        top_p (float): Nucleus sampling threshold from 0.5 to 1.0.
          Default 0.95.
        repetition_penalty (float): Penalty for repeated tokens from 1.0
          (none) to 1.5 (strong). Default 1.2.
        max_tokens (int): Maximum speech tokens to generate (approx 25/sec).
          Default 500 (~20 seconds). Max 2000.

    Returns:
        str: JSON with path to the generated WAV file and duration.
    """
    ctx = mcp.get_context()
    state = ctx.request_context.lifespan_context
    model: ChatterboxModel = state["model"]
    voices: VoiceManager = state["voices"]

    profile = voices.get_profile(voice)
    if not profile:
        return json.dumps({
            "error": f"Voice profile '{voice}' not found. "
                     f"Available: {[p.name for p in voices.list_profiles()]}"
        })

    try:
        wav, sr = model.generate(
            text=text,
            speaker_embedding=profile.speaker_embedding,
            prompt_tokens=profile.prompt_tokens,
            ref_audio_path=profile.ref_audio_path if profile.ref_audio_path.exists() else None,
            emotion_adv=emotion_adv,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=min(max_tokens, 2000),
        )
    except Exception as e:
        return json.dumps({"error": f"Generation failed: {type(e).__name__}: {e}"})

    out_name = f"{voice}_{uuid.uuid4().hex[:8]}.wav"
    out_path = OUTPUT_DIR / out_name
    sf.write(str(out_path), wav, sr)

    duration = len(wav) / sr
    return json.dumps({
        "path": str(out_path),
        "duration_seconds": round(duration, 1),
        "sample_rate": sr,
        "voice": voice,
    })


@mcp.tool(
    name="tts_stream_start",
    annotations={
        "title": "Start Streaming TTS",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def tts_stream_start(
    text: str = ...,
    voice: str = ...,
    emotion_adv: float = 0.5,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    max_tokens: int = 500,
    chunk_tokens: int = 25,
) -> str:
    """Generate speech and stream it progressively over SSE.

    Returns a URL to an SSE endpoint that delivers PCM audio chunks
    as they're decoded. The client connects to the URL and plays
    each chunk sequentially.

    Args:
        text: Text to speak.
        voice: Name of the voice profile (created via create_voice_profile).
        emotion_adv: Exaggeration from 0.0 (flat) to 1.0 (expressive).
        temperature: Sampling temperature from 0.1 to 1.5.
        top_p: Nucleus sampling threshold from 0.5 to 1.0.
        repetition_penalty: Repetition penalty from 1.0 to 1.5.
        max_tokens: Max speech tokens (approx 25/sec).
        chunk_tokens: Tokens per decode chunk (25 ≈ 1 second).

    Returns:
        JSON with stream_url and session_id.
    """
    ctx = mcp.get_context()
    state = ctx.request_context.lifespan_context
    model: ChatterboxModel = state["model"]
    voices: VoiceManager = state["voices"]
    streams: StreamManager = state["streams"]
    sse_server: StreamSSEServer = state["sse_server"]

    profile = voices.get_profile(voice)
    if not profile:
        return json.dumps({
            "error": f"Voice profile '{voice}' not found. "
                     f"Available: {[p.name for p in voices.list_profiles()]}"
        })

    if not text.strip():
        return json.dumps({"error": "Text cannot be empty"})

    if max_tokens > 2000:
        max_tokens = 2000

    session = streams.create({
        "voice": voice,
        "max_tokens": max_tokens,
        "chunk_tokens": chunk_tokens,
    })

    if not sse_server.port:
        await asyncio.to_thread(sse_server.start)

    ref_path = profile.ref_audio_path if profile.ref_audio_path.exists() else None

    # Generate speech tokens on the main thread (MLX requires this).
    speech_tokens = model.generate_tokens(
        text=text,
        speaker_embedding=profile.speaker_embedding,
        prompt_tokens=profile.prompt_tokens,
        emotion_adv=emotion_adv,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )

    def _decode_chunks():
        try:
            for pcm_bytes, chunk_idx in model.decode_chunked(
                speech_tokens=speech_tokens,
                ref_audio_path=ref_path,
                chunk_tokens=chunk_tokens,
            ):
                session.push("audio", {
                    "index": chunk_idx,
                    "data": base64.b64encode(pcm_bytes).decode(),
                    "chunk_tokens": chunk_tokens,
                })
            session.push("end", {"status": "complete"})
        except Exception as e:
            session.push("error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            session.state = "done"

    thread = threading.Thread(target=_decode_chunks, daemon=True)
    thread.start()

    return json.dumps({
        "stream_url": f"http://127.0.0.1:{sse_server.port}/stream/{session.id}",
        "session_id": session.id,
    })


@mcp.tool(
    name="create_voice_profile",
    annotations={
        "title": "Create Voice Profile",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def create_voice_profile(
    name: str = ...,
    reference_audio: str = ...,
    description: str = "",
    emotion_adv: float = 0.5,
) -> str:
    """Create a new voice profile from a reference audio file.

    The reference audio should be clean (low noise, no background music)
    and ideally 10-30 seconds of a single speaker. The extracted speaker
    embedding and speech prompt tokens are stored for later use with tts().

    Args:
        name (str): Unique name for the voice profile (e.g., 'my_voice',
          'john_smith'). Must not already exist.
        reference_audio (str): Path to a WAV or MP3 file of the speaker's
          voice. 10-30 seconds of clean speech recommended.
        description (str): Optional description or notes about this voice.
          Default empty.
        emotion_adv (float): Default exaggeration/emotion level for this
          voice, from 0.0 (flat) to 1.0 (expressive). Default 0.5. Can
          be overridden at generation time.

    Returns:
        str: JSON with profile details or error message.
    """
    ctx = mcp.get_context()
    state = ctx.request_context.lifespan_context
    model: ChatterboxModel = state["model"]
    voices: VoiceManager = state["voices"]

    audio_path = Path(reference_audio)
    if not audio_path.exists():
        return json.dumps({"error": f"Reference audio not found: {reference_audio}"})

    if voices.get_profile(name) is not None:
        return json.dumps({"error": f"Voice profile '{name}' already exists"})

    try:
        profile = voices.create_profile(
            name=name,
            reference_audio=str(audio_path.resolve()),
            model=model,
            description=description,
            emotion_adv=emotion_adv,
        )
    except Exception as e:
        return json.dumps({"error": f"Failed to create profile: {type(e).__name__}: {e}"})

    return json.dumps({
        "name": profile.name,
        "description": profile.description,
        "emotion_adv": profile.emotion_adv,
        "created_at": profile.created_at,
    })


@mcp.tool(
    name="list_voice_profiles",
    annotations={
        "title": "List Voice Profiles",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_voice_profiles() -> str:
    """List all available voice profiles.

    Returns:
        str: JSON array of profile names and metadata.
    """
    ctx = mcp.get_context()
    voices: VoiceManager = ctx.request_context.lifespan_context["voices"]
    profiles = voices.list_profiles()
    return json.dumps([p.meta for p in profiles], indent=2)


@mcp.tool(
    name="delete_voice_profile",
    annotations={
        "title": "Delete Voice Profile",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def delete_voice_profile(name: str = ...) -> str:
    """Delete a voice profile and all its stored data.

    Args:
        name (str): Name of the voice profile to delete.

    Returns:
        str: Confirmation or error message.
    """
    ctx = mcp.get_context()
    voices: VoiceManager = ctx.request_context.lifespan_context["voices"]
    if voices.delete_profile(name):
        return json.dumps({"status": "deleted", "name": name})
    return json.dumps({"error": f"Voice profile '{name}' not found"})


def main():
    mcp.run()


if __name__ == "__main__":
    main()

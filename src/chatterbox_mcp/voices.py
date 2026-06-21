"""Voice profile management for Chatterbox MCP server."""

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import librosa

from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.s3gen import S3GEN_SR

VOICES_DIR = Path.home() / ".chatterbox-mcp" / "voices"
ENC_COND_LEN = 6 * S3_SR
DEC_COND_LEN = 10 * S3GEN_SR


class VoiceProfile:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self._load_meta()

    def _load_meta(self):
        meta_path = self.path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._meta = {"name": self.name, "description": ""}

    @property
    def description(self) -> str:
        return self._meta.get("description", "")

    @property
    def emotion_adv(self) -> float:
        return self._meta.get("emotion_adv", 0.5)

    @property
    def created_at(self) -> str:
        return self._meta.get("created_at", "")

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    @property
    def speaker_embedding(self) -> np.ndarray:
        return np.load(self.path / "speaker_embedding.npy")

    @property
    def prompt_tokens(self) -> np.ndarray:
        return np.load(self.path / "prompt_tokens.npy")

    @property
    def ref_audio_path(self) -> Path:
        return self.path / "ref_audio_24k.wav"

    def exists(self) -> bool:
        return self.path.exists()


class VoiceManager:
    def __init__(self, voices_dir: str | Path | None = None):
        self.voices_dir = Path(voices_dir or VOICES_DIR)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> list[VoiceProfile]:
        if not self.voices_dir.exists():
            return []
        profiles = []
        for child in sorted(self.voices_dir.iterdir()):
            if child.is_dir() and (child / "meta.json").exists():
                try:
                    profiles.append(VoiceProfile(child))
                except Exception:
                    continue
        return profiles

    def get_profile(self, name: str) -> VoiceProfile | None:
        path = self.voices_dir / name
        if path.exists() and (path / "meta.json").exists():
            return VoiceProfile(path)
        return None

    def create_profile(
        self,
        name: str,
        reference_audio: str | Path,
        model,
        description: str = "",
        emotion_adv: float = 0.5,
    ) -> VoiceProfile:
        path = self.voices_dir / name
        if path.exists():
            raise ValueError(f"Voice profile '{name}' already exists")

        path.mkdir(parents=True)

        try:
            wav_path = Path(reference_audio)

            ref_16k, _ = librosa.load(wav_path, sr=S3_SR)
            ref_16k = ref_16k[:ENC_COND_LEN]

            ref_24k, _ = librosa.load(wav_path, sr=S3GEN_SR)
            ref_24k = ref_24k[:DEC_COND_LEN]

            ve_embed = model.ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR)
            ve_embed = ve_embed.mean(axis=0, keepdims=True)

            s3_tokzr = model.s3gen.tokenizer
            plen = model.t3.hp.speech_cond_prompt_len
            prompt_tokens, _ = s3_tokzr.forward([ref_16k[:ENC_COND_LEN]], max_len=plen)
            prompt_tokens = np.atleast_2d(np.array(prompt_tokens, dtype=np.int64))

            np.save(path / "speaker_embedding.npy", ve_embed.astype(np.float32))
            np.save(path / "prompt_tokens.npy", prompt_tokens)

            import soundfile as sf
            sf.write(str(path / "ref_audio_24k.wav"), ref_24k, S3GEN_SR)

            meta = {
                "name": name,
                "description": description,
                "emotion_adv": emotion_adv,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "ref_sr": S3GEN_SR,
            }
            with open(path / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            return VoiceProfile(path)

        except Exception:
            shutil.rmtree(path, ignore_errors=True)
            raise

    def delete_profile(self, name: str) -> bool:
        path = self.voices_dir / name
        if path.exists() and (path / "meta.json").exists():
            shutil.rmtree(path)
            return True
        return False

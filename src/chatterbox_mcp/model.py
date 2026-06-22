"""Model loading and inference for Chatterbox MCP server.

Architecture (per ADR-0001):
  PyTorch: cond-enc (speaker embedding projection + Perceiver resampler)
  MLX:     T3 backbone (LLaMA 520M) + embeddings + speech head
  PyTorch: S3Gen decoder (speech tokens → mel → audio)
"""

import os
from pathlib import Path

import numpy as np
import torch
import mlx.core as mx
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from chatterbox.models.t3.t3 import T3, T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3gen import S3GEN_SR
from chatterbox.models.s3tokenizer import SPEECH_VOCAB_SIZE
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.t3_mlx import T3MLX, T3MLXConfig

REPO_ID = "ResembleAI/chatterbox"
CACHE_DIR = Path.home() / ".cache" / "chatterbox-mcp"
MLX_WEIGHTS_PATH = CACHE_DIR / "t3_mlx_weights.npz"

_BACKBONE_KEYS = {f"layers.{i}.{sub}" for i in range(30) for sub in [
    "attention_norm.weight",
    "ffn_norm.weight",
    "attention.q_proj.weight", "attention.k_proj.weight",
    "attention.v_proj.weight", "attention.o_proj.weight",
    "mlp.gate.weight", "mlp.up.weight", "mlp.down.weight",
]} | {"norm.weight", "speech_emb.weight", "speech_head.weight",
      "text_emb.weight", "text_head.weight",
      "speech_pos_emb.weight", "text_pos_emb.weight"}

_KEY_MAP = {}
for i in range(30):
    _KEY_MAP[f"tfmr.layers.{i}.self_attn.q_proj.weight"] = f"layers.{i}.attention.q_proj.weight"
    _KEY_MAP[f"tfmr.layers.{i}.self_attn.k_proj.weight"] = f"layers.{i}.attention.k_proj.weight"
    _KEY_MAP[f"tfmr.layers.{i}.self_attn.v_proj.weight"] = f"layers.{i}.attention.v_proj.weight"
    _KEY_MAP[f"tfmr.layers.{i}.self_attn.o_proj.weight"] = f"layers.{i}.attention.o_proj.weight"
    _KEY_MAP[f"tfmr.layers.{i}.input_layernorm.weight"] = f"layers.{i}.attention_norm.weight"
    _KEY_MAP[f"tfmr.layers.{i}.post_attention_layernorm.weight"] = f"layers.{i}.ffn_norm.weight"
    _KEY_MAP[f"tfmr.layers.{i}.mlp.gate_proj.weight"] = f"layers.{i}.mlp.gate.weight"
    _KEY_MAP[f"tfmr.layers.{i}.mlp.up_proj.weight"] = f"layers.{i}.mlp.up.weight"
    _KEY_MAP[f"tfmr.layers.{i}.mlp.down_proj.weight"] = f"layers.{i}.mlp.down.weight"

_KEY_MAP.update({
    "tfmr.norm.weight": "norm.weight",
    "speech_emb.weight": "speech_emb.weight",
    "speech_head.weight": "speech_head.weight",
    "text_emb.weight": "text_emb.weight",
    "text_head.weight": "text_head.weight",
    "speech_pos_emb.emb.weight": "speech_pos_emb.weight",
    "text_pos_emb.emb.weight": "text_pos_emb.weight",
})


class ChatterboxModel:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.t3: T3 | None = None
        self.t3_mlx: T3MLX | None = None
        self.s3gen: S3Gen | None = None
        self.ve: VoiceEncoder | None = None
        self.tokenizer: EnTokenizer | None = None
        self._snapshot_dir: Path | None = None

    def _resolve_snapshot(self) -> Path:
        local_path = hf_hub_download(repo_id=REPO_ID, filename="t3_cfg.safetensors")
        return Path(local_path).parent

    def _convert_mlx_weights(self, snapshot_dir: Path):
        ckpt_path = snapshot_dir / "t3_cfg.safetensors"
        print(f"Converting MLX weights from {ckpt_path} ...")
        sd = load_file(str(ckpt_path))
        out = {}
        for pt_key, pt_weight in sd.items():
            if pt_key in _KEY_MAP:
                mlx_key = _KEY_MAP[pt_key]
                out[mlx_key] = mx.array(pt_weight.numpy())
            elif pt_key.startswith("tfmr.layers.") and "rotary" not in pt_key and "embed_tokens" not in pt_key:
                pass
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        mx.savez(str(MLX_WEIGHTS_PATH), **out)
        size_mb = MLX_WEIGHTS_PATH.stat().st_size / 1024 / 1024
        print(f"Converted {len(out)} tensors ({size_mb:.0f} MB) -> {MLX_WEIGHTS_PATH}")

    def load(self):
        if self.t3 is not None:
            return

        snapshot_dir = self._resolve_snapshot()
        self._snapshot_dir = snapshot_dir
        print(f"Model snapshot: {snapshot_dir}")

        hp = T3Config.english_only()
        t3 = T3(hp).to(self.device).eval()
        t3_state = load_file(str(snapshot_dir / "t3_cfg.safetensors"))
        tfmr_sd = {k[5:]: v for k, v in t3_state.items()
                    if k.startswith("tfmr.") and "embed_tokens" not in k}
        t3.tfmr.load_state_dict(tfmr_sd, strict=False)
        t3.speech_emb.load_state_dict({"weight": t3_state["speech_emb.weight"]})
        t3.speech_head.load_state_dict({"weight": t3_state["speech_head.weight"]})
        t3.text_emb.load_state_dict({"weight": t3_state["text_emb.weight"]})
        t3.text_head.load_state_dict({"weight": t3_state["text_head.weight"]})
        t3.text_pos_emb.load_state_dict({"emb.weight": t3_state["text_pos_emb.emb.weight"]})
        t3.speech_pos_emb.load_state_dict({"emb.weight": t3_state["speech_pos_emb.emb.weight"]})
        cond_sd = {k[9:]: v for k, v in t3_state.items() if k.startswith("cond_enc.")}
        t3.cond_enc.load_state_dict(cond_sd, strict=False)
        self.t3 = t3
        print("PyTorch T3 (cond-enc + embeddings) loaded")

        if not MLX_WEIGHTS_PATH.exists():
            self._convert_mlx_weights(snapshot_dir)
        cfg = T3MLXConfig()
        mlx_model = T3MLX(cfg)
        w = mx.load(str(MLX_WEIGHTS_PATH))
        backbone = [(k, v) for k, v in w.items() if k in _BACKBONE_KEYS]
        mlx_model.load_weights(backbone)
        self.t3_mlx = mlx_model
        print("MLX T3 backbone loaded")

        s3gen = S3Gen(meanflow=False).to(self.device).eval()
        s3gen.load_state_dict(
            load_file(str(snapshot_dir / "s3gen.safetensors")), strict=False
        )
        self.s3gen = s3gen
        print("S3Gen decoder loaded")

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(str(snapshot_dir / "ve.safetensors")))
        ve.to(self.device).eval()
        self.ve = ve
        print("VoiceEncoder loaded")

        self.tokenizer = EnTokenizer(str(snapshot_dir / "tokenizer.json"))
        print("Tokenizer loaded")

    def prepare_conditioning(
        self,
        speaker_embedding: np.ndarray,
        prompt_tokens: np.ndarray,
        emotion_adv: float = 0.5,
    ) -> np.ndarray:
        spkr_emb = torch.from_numpy(speaker_embedding).to(self.device)
        prompt = torch.from_numpy(prompt_tokens).long().to(self.device)
        emotion = torch.tensor([[[emotion_adv]]], dtype=torch.float, device=self.device)
        t3_cond = T3Cond(
            speaker_emb=spkr_emb,
            cond_prompt_speech_tokens=prompt,
            cond_prompt_speech_emb=None,
            emotion_adv=emotion,
        )
        with torch.inference_mode():
            cond_emb = self.t3.prepare_conditioning(t3_cond)
        return cond_emb.cpu().numpy()

    def generate_speech_tokens(
        self,
        cond_emb: np.ndarray,
        text_tokens: np.ndarray,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        seed: int | None = None,
    ) -> list[int]:
        cond_mx = mx.array(cond_emb)
        text_mx = mx.array(text_tokens, dtype=mx.int32)
        tokens = self.t3_mlx.generate(
            cond_mx, text_mx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed or 42,
        )
        return list(map(int, tokens))

    def decode_audio(
        self,
        speech_tokens: list[int],
        ref_audio_path: str | Path | None = None,
    ) -> np.ndarray:
        tokens_pt = torch.tensor(speech_tokens, dtype=torch.long, device=self.device)
        tokens_pt = tokens_pt[tokens_pt < SPEECH_VOCAB_SIZE]
        tokens_pt = tokens_pt.unsqueeze(0)

        if ref_audio_path:
            ref_audio_path = Path(ref_audio_path)
            if ref_audio_path.exists():
                import librosa
                ref_wav, _ = librosa.load(ref_audio_path, sr=S3GEN_SR)
                ref_wav = torch.from_numpy(ref_wav).float().to(self.device)
                ref_wav = ref_wav.unsqueeze(0)
            else:
                ref_wav = torch.zeros(1, S3GEN_SR * 5, device=self.device)
        else:
            ref_wav = torch.zeros(1, S3GEN_SR * 5, device=self.device)

        with torch.inference_mode():
            wav, _ = self.s3gen.inference(
                speech_tokens=tokens_pt,
                ref_wav=ref_wav,
                ref_sr=S3GEN_SR,
            )
        return wav[0].cpu().numpy()

    def _normalize_text(self, text: str) -> tuple[str, torch.Tensor]:
        text = text[0].upper() + text[1:] if text and text[0].islower() else text
        text = " ".join(text.split())
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        sot = torch.tensor([[self.t3.hp.start_text_token]], device=self.device)
        eot = torch.tensor([[self.t3.hp.stop_text_token]], device=self.device)
        text_tokens = torch.cat([sot, text_tokens, eot], dim=1)
        return text, text_tokens

    def generate(
        self,
        text: str,
        speaker_embedding: np.ndarray,
        prompt_tokens: np.ndarray,
        ref_audio_path: str | Path | None = None,
        emotion_adv: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_tokens: int = 500,
        seed: int | None = None,
    ) -> tuple[np.ndarray, int]:
        if not text.strip():
            text = "You need to add some text for me to talk."
        text, text_tokens = self._normalize_text(text)

        cond_emb = self.prepare_conditioning(
            speaker_embedding, prompt_tokens, emotion_adv
        )

        speech_tokens = self.generate_speech_tokens(
            cond_emb,
            text_tokens.cpu().numpy(),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

        wav = self.decode_audio(speech_tokens, ref_audio_path)
        return wav, S3GEN_SR

    def generate_chunked(
        self,
        text: str,
        speaker_embedding: np.ndarray,
        prompt_tokens: np.ndarray,
        ref_audio_path: str | Path | None = None,
        emotion_adv: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_tokens: int = 500,
        chunk_tokens: int = 25,
        seed: int | None = None,
    ):
        """Yields (pcm_int16_bytes, chunk_index) as audio chunks are decoded.

        T3 generates all speech tokens first, then S3Gen decodes them in
        cumulative chunks. Each chunk extends the waveform, and the new
        portion is yielded to the caller for streaming.
        """
        speech_tokens = self.generate_tokens(
            text, speaker_embedding, prompt_tokens,
            emotion_adv, temperature, top_p, repetition_penalty,
            max_tokens, seed,
        )
        yield from self.decode_chunked(speech_tokens, ref_audio_path, chunk_tokens)

    def generate_tokens(
        self,
        text: str,
        speaker_embedding: np.ndarray,
        prompt_tokens: np.ndarray,
        emotion_adv: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_tokens: int = 500,
        seed: int | None = None,
    ) -> list[int]:
        """Text → speech tokens (MLX on main thread)."""
        text, text_tokens = self._normalize_text(text)
        cond_emb = self.prepare_conditioning(
            speaker_embedding, prompt_tokens, emotion_adv
        )
        return self.generate_speech_tokens(
            cond_emb,
            text_tokens.cpu().numpy(),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

    def decode_chunked(
        self,
        speech_tokens: list[int],
        ref_audio_path: str | Path | None = None,
        chunk_tokens: int = 25,
    ):
        """Yields (pcm_int16_bytes, chunk_index) from pre-generated tokens.

        PyTorch S3Gen decode runs on any thread — safe for background use.
        """
        if not speech_tokens:
            return
        prev_len = 0
        n = len(speech_tokens)
        chunk_idx = 0
        for chunk_end in range(chunk_tokens, n + chunk_tokens, chunk_tokens):
            chunk_end = min(chunk_end, n)
            wav = self.decode_audio(speech_tokens[:chunk_end], ref_audio_path)
            new_samples = len(wav) - prev_len
            if new_samples > 0:
                delta = wav[-new_samples:]
                pcm = (delta * 32767).astype(np.int16).tobytes()
                yield pcm, chunk_idx
                chunk_idx += 1
                prev_len = len(wav)

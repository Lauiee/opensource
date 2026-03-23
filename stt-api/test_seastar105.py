"""
Test seastar105/whisper-medium-komixv2 and jangmin/whisper-medium-ko-normalized-1273h
on the medical STT dataset (21 types).

Uses model.generate() directly with Whisper's sequential long-form decoding
to handle audio > 30 seconds.

Usage:
    python -X utf8 test_seastar105.py
    python -X utf8 test_seastar105.py --model seastar105
    python -X utf8 test_seastar105.py --model jangmin
    python -X utf8 test_seastar105.py --model both
    python -X utf8 test_seastar105.py --model seastar105 --prompt
"""

import argparse
import json
import os
import time
import unicodedata
from pathlib import Path

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torchaudio
import numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor,
    WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor,
)

# === Paths ===
DATA_DIR = Path(r"C:\Users\shwns\Desktop\data_set")
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "seastar105": "seastar105/whisper-medium-komixv2",
    "jangmin": "jangmin/whisper-medium-ko-normalized-1273h",
}

# === Baseline large-v3 CER results (from memory) ===
BASELINE_CER = {
    1: 11.1, 2: 41.0, 3: 24.0, 4: 46.0, 5: 22.0,
    6: 29.0, 7: 15.0, 8: 13.6, 9: 12.9, 10: 36.0,
    11: 59.0, 12: 8.3, 13: 3.2, 14: 10.3, 15: 0.0,
    16: 26.0, 17: 70.0, 18: 36.0, 19: 46.0, 20: 28.0,
    21: 32.0,
}


def load_reference(type_num: int) -> str:
    answer_path = DATA_DIR / f"answer{type_num}.txt"
    with open(answer_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return " ".join(item["content"] for item in data)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    if len(ref_norm) == 0:
        return 0.0 if len(hyp_norm) == 0 else 1.0
    dist = levenshtein_distance(ref_norm, hyp_norm)
    return dist / len(ref_norm)


def get_audio_duration(wav_path: str) -> float:
    info = torchaudio.info(wav_path)
    return info.num_frames / info.sample_rate


def load_audio(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio to 16kHz mono numpy array."""
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()


def load_model_and_processor(model_id: str):
    """Load model and processor, handling Flax models and missing tokenizers."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    is_flax = "seastar105" in model_id

    print(f"\n{'='*60}")
    print(f"Loading model: {model_id}")
    print(f"{'='*60}")

    if is_flax:
        from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model
        from huggingface_hub import hf_hub_download
        print("  Loading Flax weights (manual conversion)...")
        config = WhisperConfig.from_pretrained(model_id)
        model = WhisperForConditionalGeneration(config)
        flax_path = hf_hub_download(model_id, "flax_model.msgpack")
        model = load_flax_checkpoint_in_pytorch_model(model, flax_path)
        model = model.to(dtype=dtype).to(device)
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model.to(device)

    # Load processor (tokenizer + feature extractor)
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception:
        print(f"  Processor not found for {model_id}, using openai/whisper-medium")
        processor = AutoProcessor.from_pretrained("openai/whisper-medium")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count/1e6:.1f}M params on {device} ({dtype})")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory used: {mem:.2f} GB")

    # Load generation config from the model repo
    try:
        from transformers import GenerationConfig
        gen_config = GenerationConfig.from_pretrained(model_id)
        model.generation_config = gen_config
        print(f"  Generation config loaded from {model_id}")
    except Exception:
        print("  Using default generation config")

    model.config.forced_decoder_ids = None  # Let us control language/task

    return model, processor, device, dtype


def transcribe_long_audio(model, processor, audio_np, device, dtype,
                          language="ko", prompt_text=None):
    """
    Transcribe long audio using Whisper's built-in sequential long-form decoding.
    model.generate() handles chunking internally when input > 30s.
    """
    # Process audio features - DON'T truncate for long audio
    input_features = processor.feature_extractor(
        audio_np, sampling_rate=16000, return_tensors="pt",
        truncation=False, padding="longest",
        return_attention_mask=True,
    ).input_features.to(device, dtype=dtype)

    # Build generate kwargs
    gen_kwargs = {
        "input_features": input_features,
        "language": language,
        "task": "transcribe",
        "return_timestamps": True,   # needed for sequential long-form decoding
        "max_new_tokens": 440,
    }

    # Add prompt if specified
    if prompt_text:
        prompt_ids = processor.tokenizer.get_prompt_ids(prompt_text)
        gen_kwargs["prompt_ids"] = torch.tensor(prompt_ids, dtype=torch.long).to(device)

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    text = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text.strip()


def run_evaluation(model_key: str, use_prompt: bool = False):
    """Run full evaluation for a model on all 21 types."""
    model_id = MODELS[model_key]
    model, processor, device, dtype = load_model_and_processor(model_id)

    prompt_text = "진료 상담 녹음입니다." if use_prompt else None
    prompt_label = "with_prompt" if use_prompt else "no_prompt"

    results = []
    total_cer = 0.0

    print(f"\nEvaluating {model_key} ({prompt_label})...")
    print(f"{'Type':>5} | {'CER':>7} | {'Ref':>5} | {'Hyp':>5} | {'Ratio':>6} | {'Duration':>8} | {'Time':>6}")
    print("-" * 70)

    for type_num in range(1, 22):
        wav_path = str(DATA_DIR / f"type{type_num}" / f"type{type_num}.wav")
        if not os.path.exists(wav_path):
            print(f"  [SKIP] {wav_path} not found")
            continue

        reference = load_reference(type_num)
        duration = get_audio_duration(wav_path)
        audio_np = load_audio(wav_path)

        t0 = time.time()
        try:
            hypothesis = transcribe_long_audio(
                model, processor, audio_np, device, dtype,
                language="ko", prompt_text=prompt_text,
            )
        except Exception as e:
            print(f"  [ERROR] type{type_num}: {e}")
            hypothesis = ""
        elapsed = time.time() - t0

        cer = compute_cer(reference, hypothesis)
        ref_len = len(normalize_text(reference))
        hyp_len = len(normalize_text(hypothesis))
        ratio = hyp_len / ref_len if ref_len > 0 else 0.0
        truncated = ratio < 0.5 and ref_len > 50

        total_cer += cer

        results.append({
            "type": type_num,
            "cer": round(cer * 100, 1),
            "ref_len": ref_len,
            "hyp_len": hyp_len,
            "ratio": round(ratio, 2),
            "duration_s": round(duration, 1),
            "elapsed_s": round(elapsed, 1),
            "truncated": truncated,
            "reference": reference[:100],
            "hypothesis": hypothesis[:100],
        })

        flag = " *** TRUNCATED ***" if truncated else ""
        print(f"{type_num:>5} | {cer*100:>6.1f}% | {ref_len:>5} | {hyp_len:>5} | {ratio:>5.2f}x | {duration:>7.1f}s | {elapsed:>5.1f}s{flag}")

    avg_cer = total_cer / len(results) if results else 0.0
    print(f"\n{'AVERAGE CER':>20}: {avg_cer*100:.1f}%")
    print(f"{'Truncated types':>20}: {sum(1 for r in results if r['truncated'])}")

    output = {
        "model": model_id,
        "model_key": model_key,
        "prompt": prompt_text,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "avg_cer": round(avg_cer * 100, 1),
        "results": results,
    }

    filename = f"{model_key}_result_{prompt_label}.json"
    out_path = OUTPUT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output


def print_comparison(all_results: list):
    """Print comparison table across all models."""
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")

    header = f"{'Type':>5} | {'large-v3':>10}"
    for res in all_results:
        label = f"{res['model_key']}"
        if res['prompt']:
            label += "+prompt"
        header += f" | {label:>15}"
    header += f" | {'Best':>15}"
    print(header)
    print("-" * len(header))

    for type_num in range(1, 22):
        baseline = BASELINE_CER.get(type_num, -1)
        row = f"{type_num:>5} | {baseline:>9.1f}%"

        cers = {"large-v3": baseline}
        for res in all_results:
            label = res['model_key']
            if res['prompt']:
                label += "+prompt"
            type_result = next((r for r in res['results'] if r['type'] == type_num), None)
            if type_result:
                cer = type_result['cer']
                trunc = " T" if type_result['truncated'] else ""
                row += f" | {cer:>13.1f}%{trunc}"
                cers[label] = cer
            else:
                row += f" | {'N/A':>15}"

        if cers:
            best_name = min(cers, key=cers.get)
            row += f" | {best_name:>15}"
        print(row)

    avg_row = f"{'AVG':>5} | {sum(BASELINE_CER.values())/len(BASELINE_CER):>9.1f}%"
    for res in all_results:
        avg_row += f" | {res['avg_cer']:>13.1f}% "
    print("-" * len(header))
    print(avg_row)

    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline_avg_cer": round(sum(BASELINE_CER.values()) / len(BASELINE_CER), 1),
        "models": [],
    }
    for res in all_results:
        comparison["models"].append({
            "model": res["model"],
            "key": res["model_key"],
            "prompt": res["prompt"],
            "avg_cer": res["avg_cer"],
        })

    comp_path = OUTPUT_DIR / "model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\nComparison saved to {comp_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Korean Whisper models")
    parser.add_argument("--model", choices=["seastar105", "jangmin", "both"], default="both")
    parser.add_argument("--prompt", action="store_true", help="Also test with medical prompt")
    args = parser.parse_args()

    all_results = []

    models_to_test = []
    if args.model in ("seastar105", "both"):
        models_to_test.append("seastar105")
    if args.model in ("jangmin", "both"):
        models_to_test.append("jangmin")

    for model_key in models_to_test:
        result = run_evaluation(model_key, use_prompt=False)
        all_results.append(result)

        if args.prompt:
            result_prompt = run_evaluation(model_key, use_prompt=True)
            all_results.append(result_prompt)

    print_comparison(all_results)


if __name__ == "__main__":
    main()

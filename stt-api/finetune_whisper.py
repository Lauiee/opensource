"""Whisper Fine-tuning — PEFT 없이 직접 디코더만 학습.

PEFT 호환성 문제를 우회: 디코더 레이어만 unfreeze하여 학습.
RTX 3070 Ti (8GB VRAM) 대응.
"""
import json, sys, time
from pathlib import Path

DATA_DIR = Path(r"C:\Users\shwns\Desktop\data_set")
OUTPUT_DIR = Path(__file__).parent / "models" / "whisper-medical-ko"


def prepare_dataset():
    samples = []
    for t in range(1, 22):
        wav_path = DATA_DIR / f"type{t}" / f"type{t}.wav"
        ans_path = DATA_DIR / f"answer{t}.txt"
        if not wav_path.exists() or not ans_path.exists():
            continue
        try:
            raw = ans_path.read_text(encoding="utf-8").strip()
            be = raw.rfind("]")
            if be >= 0: raw = raw[:be + 1]
            data = json.loads(raw)
            text = " ".join(item.get("content", "") for item in data if item.get("content"))
        except Exception:
            text = ans_path.read_text(encoding="utf-8").strip()
        if text:
            samples.append({"audio": str(wav_path), "text": text[:500], "type": t})
    print(f"Dataset: {len(samples)} samples")
    return samples


def finetune():
    import torch
    import torchaudio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    samples = prepare_dataset()
    if not samples:
        return

    model_name = "openai/whisper-large-v3"
    print(f"Loading: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")

    # Freeze encoder, only train decoder attention
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze decoder attention layers only
    for name, param in model.named_parameters():
        if "decoder" in name and ("q_proj" in name or "v_proj" in name):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01
    )

    model.train()
    epochs = 3

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        epoch_loss = 0
        count = 0

        for i, sample in enumerate(samples):
            waveform, sr = torchaudio.load(sample["audio"])
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            waveform = waveform.squeeze().numpy()
            # 30 sec limit
            if len(waveform) > 30 * 16000:
                waveform = waveform[:30 * 16000]

            input_features = processor(
                waveform, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            labels = processor.tokenizer(
                sample["text"], return_tensors="pt",
                max_length=448, truncation=True,
            ).input_ids.to(device)

            try:
                outputs = model.model.forward(
                    input_features=input_features,
                    decoder_input_ids=labels[:, :-1],
                )
                # Manual loss
                logits = model.proj_out(outputs.last_hidden_state)
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels[:, 1:].reshape(-1))

                loss.backward()

                if (count + 1) % 4 == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                count += 1

                if count % 5 == 0:
                    print(f"  Step {count}: loss={loss.item():.4f} (Type{sample['type']})")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  OOM at Type{sample['type']}, skip")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

        # Final optimizer step
        optimizer.step()
        optimizer.zero_grad()

        if count > 0:
            print(f"  Avg loss: {epoch_loss/count:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved: {OUTPUT_DIR}")


if __name__ == "__main__":
    finetune()

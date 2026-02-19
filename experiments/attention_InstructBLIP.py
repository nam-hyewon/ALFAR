import argparse
import json
import os
import glob
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


# =========================
# Utilities
# =========================
def resolve_infoseek_image_path(image_folder: str, image_name: str) -> str | None:
    base_name = os.path.splitext(image_name)[0]
    possible_dirs = [image_folder] + [os.path.join(image_folder, f"{k:02d}") for k in range(1, 9)]
    for d in possible_dirs:
        search_pattern = os.path.join(d, f"{base_name}.*")
        matches = glob.glob(search_pattern)
        valid_files = [m for m in matches if os.path.isfile(m)]
        if valid_files:
            return valid_files[0]
    return None


def find_marker_token_positions(tokenizer, ids: list[int], marker: str) -> int:
    """Find first position where decoded prefix contains marker (incremental decode)."""
    pos = -1
    for k in range(len(ids)):
        text_so_far = tokenizer.decode(ids[:k+1], skip_special_tokens=False)
        if marker in text_so_far:
            pos = k
            break
    return pos


def safe_truncate_context(ctx: str, max_chars: int) -> str:
    if max_chars is None or max_chars <= 0:
        return ctx
    return ctx[:max_chars]


# =========================
# Main evaluation
# =========================
def eval_model(args):
    print(f"Loading Model: {args.model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for this script.")

    dtype = torch.float16 if args.fp16 else torch.bfloat16

    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",      # spreads weights if multiple GPUs; otherwise uses one
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

    # ---- Force eager attention (sdpa doesn't support output_attentions) ----
    # Global config on the wrapper (may or may not propagate)
    if hasattr(model, "config"):
        model.config.attn_implementation = "eager"

    # Vicuna/LLaMA language model config
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config.attn_implementation = "eager"

    # Internals: sometimes actual module is model.language_model.model
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "config"):
        model.language_model.model.config.attn_implementation = "eager"

    print("[Attn Impl] wrapper:", getattr(model.config, "attn_implementation", None))
    print("[Attn Impl] lm:", getattr(getattr(model, "language_model", None).config, "attn_implementation", None))

    processor = InstructBlipProcessor.from_pretrained(args.model_path)

    # IMPORTANT: make sure processor knows how many query tokens per image to expand placeholders
    # HF docs note this attribute should match model.config.num_query_tokens (often 32).
    if not hasattr(processor, "num_query_tokens") or processor.num_query_tokens is None:
        processor.num_query_tokens = int(getattr(model.config, "num_query_tokens", 32))

    tokenizer = processor.tokenizer  # vicuna tokenizer

    # image token index used in input_ids after processor expands placeholders
    image_token_index = getattr(model.config, "image_token_index", None)
    if image_token_index is None:
        # fallback: try "<image>" token id in tokenizer
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")

    print(f"[Config] num_query_tokens={processor.num_query_tokens}, image_token_index={image_token_index}")

    # ==========================================
    # Data path setting (same as yours)
    # ==========================================
    ROOT = Path(__file__).resolve().parents[0]
    if "experiments" in str(ROOT):
        ROOT = ROOT.parent
    DATA_DIR = ROOT / "data"

    try:
        question_file = DATA_DIR / "eval_data" / "mc" / "infoseek_mc.json"
        indices = np.load(DATA_DIR / "retrieval_result" / "infoseek_mc_indices_50_17k.npy", allow_pickle=True)
        indice_map = np.load(DATA_DIR / "wiki" / "wiki_map_17k.npy", allow_pickle=True).item()
        knowledge_base = np.load(DATA_DIR / "wiki" / "wiki_with_image.npy", allow_pickle=True).item()

        with open(question_file, "r") as f:
            questions = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Start Attention Analysis on {args.num_samples} valid samples...")

    valid_count = 0
    avg_layer_image_attn = None
    avg_layer_context_attn = None
    avg_layer_text_attn = None

    for i, line in tqdm(enumerate(questions), total=len(questions)):
        if i < args.start_index:
            continue
        if i >= len(indices):
            break

        # --------------------------
        # Load sample
        # --------------------------
        question = line["question"]
        choices = line["multiple_choices"]

        cur_indices = indices[i]
        know_index = indice_map[cur_indices[0]]
        context = knowledge_base[know_index]["wikipedia_summary"]
        context = safe_truncate_context(context, args.max_context_chars)

        image_path = resolve_infoseek_image_path(args.image_folder, line["image"])
        if image_path is None:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        # --------------------------
        # Build prompt (keep your style)
        # NOTE: InstructBLIP will prepend/expand <image> placeholders internally.
        # --------------------------
        choice_str = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in choices.items()])
        prompt = (
            "Answer the question using a single word based on the given context.\n"
            f"Context: {context}\n"
            f"Question: {question}{choice_str}\n"
            "Answer:"
        )

        # --------------------------
        # Prepare inputs via processor
        # --------------------------
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_text_tokens,
        )

        # move tensors to model's device (device_map="auto" -> choose first param device)
        # safest: send to same device as model language head
        model_device = next(model.parameters()).device
        for k in inputs:
            inputs[k] = inputs[k].to(model_device)

        input_ids = inputs["input_ids"][0].detach().cpu().tolist()

        # --------------------------
        # Forward with attentions (NO cache to save memory)
        # --------------------------
        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )

        # InstructBlipForConditionalGeneration returns language_model_outputs inside
        lm_out = getattr(outputs, "language_model_outputs", None)
        if lm_out is None or getattr(lm_out, "attentions", None) is None:
            print("\n[Warning] No language_model_outputs.attentions. Check transformers version / output_attentions.")
            return

        attentions = lm_out.attentions  # tuple: (layer) -> (B, heads, seq, seq)
        num_layers = len(attentions)

        # --------------------------
        # (1) Token index mapping: Image / Context / Text
        # --------------------------
        # Image indices = positions of expanded <image> placeholders
        image_indices = [j for j, tok in enumerate(input_ids) if tok == image_token_index]

        # Find context/question boundaries
        context_start_idx = find_marker_token_positions(tokenizer, input_ids, "Context:")
        query_start_idx = find_marker_token_positions(tokenizer, input_ids, "Question:")

        if context_start_idx == -1:
            context_start_idx = len(input_ids) // 3
        if query_start_idx == -1:
            query_start_idx = (len(input_ids) // 3) * 2

        context_indices = list(range(context_start_idx, max(context_start_idx, query_start_idx)))

        image_set = set(image_indices)
        context_set = set(context_indices)
        text_indices = [j for j in range(len(input_ids)) if (j not in image_set and j not in context_set)]

        print(f"\n[Token Check] Image: {len(image_indices)}개, Context: {len(context_indices)}개, Text: {len(text_indices)}개")

        # --------------------------
        # (2) Layer-wise attention aggregation (same logic)
        # --------------------------
        layer_image_attn = []
        layer_context_attn = []
        layer_text_attn = []

        for layer_idx in range(num_layers):
            layer_attn = attentions[layer_idx][0].float()   # (heads, seq, seq) after batch slice
            last_token_attn = layer_attn[:, -1, :]          # (heads, seq)
            avg_head_attn = last_token_attn.mean(dim=0).detach().cpu().numpy()  # (seq,)

            total_attn = avg_head_attn.sum()
            if total_attn <= 0:
                layer_image_attn.append(0.0)
                layer_context_attn.append(0.0)
                layer_text_attn.append(0.0)
                continue

            img_attn_ratio = avg_head_attn[image_indices].sum() / total_attn if image_indices else 0.0
            ctx_attn_ratio = avg_head_attn[context_indices].sum() / total_attn if context_indices else 0.0
            txt_attn_ratio = avg_head_attn[text_indices].sum() / total_attn if text_indices else 0.0

            layer_image_attn.append(img_attn_ratio)
            layer_context_attn.append(ctx_attn_ratio)
            layer_text_attn.append(txt_attn_ratio)

        if avg_layer_image_attn is None:
            avg_layer_image_attn = np.zeros(num_layers, dtype=np.float64)
            avg_layer_context_attn = np.zeros(num_layers, dtype=np.float64)
            avg_layer_text_attn = np.zeros(num_layers, dtype=np.float64)

        avg_layer_image_attn += np.array(layer_image_attn, dtype=np.float64)
        avg_layer_context_attn += np.array(layer_context_attn, dtype=np.float64)
        avg_layer_text_attn += np.array(layer_text_attn, dtype=np.float64)

        valid_count += 1
        if valid_count >= args.num_samples:
            break

    if valid_count == 0:
        print("No valid samples processed. Check image paths / dataset indices.")
        return

    # ==========================================
    # 4. Matplotlib visualization (averaged)
    # ==========================================
    avg_layer_image_attn /= valid_count
    avg_layer_context_attn /= valid_count
    avg_layer_text_attn /= valid_count

    plt.figure(figsize=(14, 6))
    x = np.arange(1, len(avg_layer_image_attn) + 1)
    width = 0.25

    plt.bar(x - width, avg_layer_image_attn, width, label="Image Tokens", color="#FF9999")
    plt.bar(x, avg_layer_context_attn, width, label="Context Tokens", color="#66B2FF")
    plt.bar(x + width, avg_layer_text_attn, width, label="Text Tokens", color="#99FF99")

    plt.xlabel("Layer Index")
    plt.ylabel("Average Attention Ratio")
    plt.title(f"Attention Distribution across {valid_count} samples: Image vs Context vs Text (InstructBLIP-Vicuna)")
    plt.xticks(x[::2])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(args.save_path)
    print(f"\nEvaluation finished! Processed {valid_count} samples.")
    print(f"Visualization saved to {args.save_path}. Please check this image!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="Salesforce/instructblip-vicuna-7b")
    parser.add_argument("--image-folder", type=str, default="../data/infoseek/images")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)

    # OOM control
    parser.add_argument("--max-context-chars", type=int, default=800)      # 줄이면 OOM 확률 확 내려감
    parser.add_argument("--max-text-tokens", type=int, default=1024)       # processor tokenizer max_length
    parser.add_argument("--fp16", action="store_true")                     # 필요하면 켜서 메모리 절약

    parser.add_argument("--save-path", type=str, default="attention_instructblip_result.png")

    args = parser.parse_args()
    eval_model(args)
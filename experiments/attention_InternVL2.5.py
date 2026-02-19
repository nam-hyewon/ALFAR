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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer


# =========================
# InternVL2.5 image loading
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # tie-breaker consistent with InternVL examples
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []

    # tile
    tiles_per_row = target_width // image_size
    for i in range(blocks):
        box = (
            (i % tiles_per_row) * image_size,
            (i // tiles_per_row) * image_size,
            ((i % tiles_per_row) + 1) * image_size,
            ((i // tiles_per_row) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))

    # optional thumbnail
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image_internvl(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])  # (num_tiles, 3, 448, 448)
    return pixel_values


# =========================
# Utilities
# =========================
def resolve_infoseek_image_path(image_folder: str, image_name: str) -> str | None:
    """
    Matches your original search logic:
      - check image_folder
      - then check image_folder/01 ... /08
      - match by base name with any extension
    """
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
    """
    Find first position where decoded prefix contains marker.
    Keeps your original "incremental decode" approach.
    """
    pos = -1
    for k in range(len(ids)):
        text_so_far = tokenizer.decode(ids[:k+1])
        if marker in text_so_far:
            pos = k
            break
    return pos


# =========================
# Main evaluation
# =========================
def eval_model(args):
    # ==========================================
    # 1. Loading Model & Tokenizer
    # ==========================================
    print(f"Loading Model: {args.model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for InternVL2.5 inference in this script.")

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,   # IMPORTANT for attentions
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )

    # InternVL requires this to map where to inject visual embeddings
    try:
        model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    except Exception:
        # Some variants may expose different property; still try to continue.
        pass

    # ==========================================
    # 2. Data path setting (same as yours)
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

    # ==========================================
    # 3. Inference & Attention Extraction
    # ==========================================
    valid_count = 0
    total_samples = args.num_samples

    avg_layer_image_attn = None
    avg_layer_context_attn = None
    avg_layer_text_attn = None

    # Token ids for InternVL image-token grouping
    img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    img_start_id = tokenizer.convert_tokens_to_ids("<img>")
    img_end_id = tokenizer.convert_tokens_to_ids("</img>")

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

        image_name = line["image"]
        image_path = resolve_infoseek_image_path(args.image_folder, image_name)
        if image_path is None:
            continue

        # --------------------------
        # Build prompt (InternVL: MUST include <image>)
        # --------------------------
        choice_str = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in choices.items()])
        prompt = (
            "Answer the question using a single word based on the given context.\n"
            f"Context: {context}\n"
            f"Question: {question}{choice_str}\n"
            "Answer:"
        )
        query = "<image>\n" + prompt

        # --------------------------
        # Prepare inputs
        # --------------------------
        try:
            pixel_values = load_image_internvl(image_path, input_size=args.image_size, max_num=args.max_tiles)
        except Exception:
            continue

        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)  # (num_tiles, 3, 448, 448)

        # pixel_values 준비 직후 (tokenizer(query) 호출 직전)에 추가
        num_tiles = pixel_values.size(0)

        # InternVL에서 타일 1장당 들어가는 비주얼 토큰 수 (보통 256)
        num_img_token = getattr(model, "num_image_token", 256)

        # <image> -> <img> <IMG_CONTEXT> ... </img> 로 확장 (공백 포함 권장)
        img_ctx = " ".join(["<IMG_CONTEXT>"] * (num_tiles * num_img_token))
        image_prompt = f"<img> {img_ctx} </img>"

        # query는 원래 "<image>\n" + prompt 였지?
        query_expanded = query.replace("<image>", image_prompt)

        enc = tokenizer(query_expanded, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # InternVL expects image_flags: (num_images_or_tiles, 1), all ones for valid tiles
        image_flags = torch.ones((pixel_values.size(0), 1), dtype=torch.long, device=device)

        print("IMG_CONTEXT id:", tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
        print("<img> id:", tokenizer.convert_tokens_to_ids("<img>"))
        print("</img> id:", tokenizer.convert_tokens_to_ids("</img>"))
        print("UNK id:", tokenizer.unk_token_id)

        # --------------------------
        # Forward with attentions
        # --------------------------
        with torch.inference_mode():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                output_attentions=True,
                return_dict=True,
            )

        attentions = outputs.attentions
        if attentions is None:
            print("\n[Warning] outputs.attentions is None. Try setting use_flash_attn=False or different dtype.")
            return

        num_layers = len(attentions)

        # --------------------------
        # (1) Token index mapping: Image / Context / Text
        # --------------------------
        ids = input_ids[0].detach().cpu().tolist()

        # Image token indices: <IMG_CONTEXT> + optional <img>, </img>
        image_indices = [j for j, t in enumerate(ids) if t in (img_ctx_id, img_start_id, img_end_id)]

        # Find "Context:" and "Question:" boundaries (incremental decode, like your code)
        context_start_idx = find_marker_token_positions(tokenizer, ids, "Context:")
        query_start_idx = find_marker_token_positions(tokenizer, ids, "Question:")

        # Fallbacks
        if context_start_idx == -1:
            context_start_idx = len(ids) // 3
        if query_start_idx == -1:
            query_start_idx = (len(ids) // 3) * 2

        # Context indices: from "Context:" to just before "Question:"
        context_indices = list(range(context_start_idx, max(context_start_idx, query_start_idx)))

        # Text indices: everything else not in image/context
        context_set = set(context_indices)
        image_set = set(image_indices)
        text_indices = [j for j in range(len(ids)) if (j not in image_set and j not in context_set)]

        print(f"\n[Token Check] Image: {len(image_indices)}개, Context: {len(context_indices)}개, Text: {len(text_indices)}개")

        # --------------------------
        # (2) Layer-wise attention aggregation (same logic)
        # --------------------------
        layer_image_attn = []
        layer_context_attn = []
        layer_text_attn = []

        for layer_idx in range(num_layers):
            # (B, heads, seq, seq) -> take batch 0
            layer_attn = attentions[layer_idx][0].float()

            # attention FROM last token TO all previous tokens: (heads, seq)
            last_token_attn = layer_attn[:, -1, :]

            # average over heads: (seq,)
            avg_head_attn = last_token_attn.mean(dim=0).detach().cpu().numpy()

            total_attn = avg_head_attn.sum()
            if total_attn <= 0:
                # avoid division by 0
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

        # Initialize accumulators
        if avg_layer_image_attn is None:
            avg_layer_image_attn = np.zeros(num_layers, dtype=np.float64)
            avg_layer_context_attn = np.zeros(num_layers, dtype=np.float64)
            avg_layer_text_attn = np.zeros(num_layers, dtype=np.float64)

        avg_layer_image_attn += np.array(layer_image_attn, dtype=np.float64)
        avg_layer_context_attn += np.array(layer_context_attn, dtype=np.float64)
        avg_layer_text_attn += np.array(layer_text_attn, dtype=np.float64)

        valid_count += 1
        if valid_count >= total_samples:
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

    # NOTE: keeping your style; colors are explicit as you used
    plt.bar(x - width, avg_layer_image_attn, width, label="Image Tokens", color="#FF9999")
    plt.bar(x, avg_layer_context_attn, width, label="Context Tokens", color="#66B2FF")
    plt.bar(x + width, avg_layer_text_attn, width, label="Text Tokens", color="#99FF99")

    plt.xlabel("Layer Index")
    plt.ylabel("Average Attention Ratio")
    plt.title(f"Attention Distribution across {valid_count} samples: Image vs Context vs Text")
    plt.xticks(x[::2])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    save_path = args.save_path
    plt.savefig(save_path)
    print(f"\nEvaluation finished! Processed {valid_count} samples.")
    print(f"Visualization saved to {save_path}. Please check this image!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model & io
    parser.add_argument("--model-path", type=str, default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--image-folder", type=str, default="../data/infoseek/images")
    parser.add_argument("--dataset", type=str, default="infoseek")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)

    # internvl image tiling
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-tiles", type=int, default=1)   ##############################

    # output
    parser.add_argument("--save-path", type=str, default="attention_internvl25_result.png")

    args = parser.parse_args()
    eval_model(args)
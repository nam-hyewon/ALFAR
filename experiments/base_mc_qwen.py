import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import glob

def eval_model(args):
    # ==========================================
    # 1. Loading Model & Processor 
    # ==========================================
    print(f"Loading Model: {args.model_path}")
    
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Baseline에서는 Attention이나 특수 토큰의 ID를 수동으로 추적할 필요가 없습니다.

    # ==========================================
    # 2. Data path setting
    # ==========================================
    ROOT = Path(__file__).resolve().parents[0] 
    if 'experiments' in str(ROOT): ROOT = ROOT.parent
    DATA_DIR = ROOT / 'data'
    
    try:
        question_file = DATA_DIR / 'eval_data' / 'mc' / 'infoseek_mc.json'
        indices = np.load(DATA_DIR / 'retrieval_result' / 'infoseek_mc_indices_50_17k.npy', allow_pickle=True)
        indice_map = np.load(DATA_DIR / 'wiki' / 'wiki_map_17k.npy', allow_pickle=True).item()
        knowledge_base = np.load(DATA_DIR / 'wiki' / 'wiki_with_image.npy', allow_pickle=True).item()
        
        with open(question_file, "r") as f:
            questions = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")

    print(f"Start Baseline (MRAG) evaluation on {len(questions)} items...")

    # ==========================================
    # 3. Baseline inference loop
    # ==========================================
    for i, line in tqdm(enumerate(questions), total=len(questions)):
        if i < args.start_index:
            continue
        idx = line['id']
        question = line["question"]
        choices = line['multiple_choices']
        
        # (1) Context & Image 준비
        if i >= len(indices): break
        
        cur_indices = indices[i]
        # Top-1 검색 결과 사용 (MRAG Baseline)
        know_index = indice_map[cur_indices[0]] 
        context = knowledge_base[know_index]['wikipedia_summary']

        image_name = line["image"]
        
        base_name = os.path.splitext(image_name)[0]
        
        # 탐색할 대상 폴더 경로들
        possible_dirs = [args.image_folder]
        for k in range(1, 9):
            possible_dirs.append(os.path.join(args.image_folder, f"{k:02d}"))
        
        image_path = None
        for d in possible_dirs:
            # 폴더 경로와 '파일명.*' 형태의 패턴 생성 (예: /path/to/images/01/123.*)
            search_pattern = os.path.join(d, f"{base_name}.*")
            
            # 패턴과 일치하는 모든 확장자 파일 검색
            matches = glob.glob(search_pattern)
            
            # 매칭된 파일 중 실제 파일인 것만 필터링 (디렉토리 제외)
            valid_files = [m for m in matches if os.path.isfile(m)]
            
            if valid_files:
                image_path = valid_files[0] # 첫 번째로 찾은 이미지 파일 사용
                break
        
        if image_path is None:
            continue

        # (2) 프롬프트 준비
        # Baseline은 ALFAR처럼 Parametric Prompt를 따로 만들지 않고, 
        # Context가 포함된 단일 프롬프트만 사용합니다.
        choice_str = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in choices.items()])
        prompt = f"Answer the question using a single word based on the given context.\nContext: {context}\nQuestion: {question}{choice_str}\nAnswer:"

        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]

        # (3) Processor 처리
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)

        # ==========================================
        # (4) Standard Generation (No ALFAR)
        # ==========================================
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=16, # 단답형
                do_sample=False    # Greedy Decoding (ALFAR 코드와 조건 통일)
            )
            
        # 프롬프트 부분을 제외하고 새로 생성된 토큰만 잘라냅니다.
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 텍스트 디코딩
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        ans_file.write(json.dumps({
            "data_id": idx,
            "prediction": output_text.strip(),
            "model_id": args.model_path,
            "image": line["image"]
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    print("Baseline Evaluation finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image-folder", type=str, default="../data/infoseek/images")
    parser.add_argument("--dataset", type=str, default="infoseek")
    parser.add_argument("--answers-file", type=str, default="./results/qwen_baseline_result.jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    
    args = parser.parse_args()
    eval_model(args)
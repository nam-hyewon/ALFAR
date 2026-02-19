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
import matplotlib.pyplot as plt 

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
        attn_implementation="eager", 
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

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

    print(f"Start Attention Analysis on {args.num_samples} valid samples...")

    # ==========================================
    # 3. Baseline inference & Attention Extraction
    # ==========================================
    valid_count = 0
    total_samples = args.num_samples
    
    # 누적 평균을 구하기 위한 변수
    avg_layer_image_attn = None
    avg_layer_context_attn = None
    avg_layer_query_attn = None

    for i, line in tqdm(enumerate(questions), total=len(questions)):
        if i < args.start_index:
            continue
            
        idx = line['id']
        question = line["question"]
        choices = line['multiple_choices']
        
        if i >= len(indices): break
        cur_indices = indices[i]
        know_index = indice_map[cur_indices[0]] 
        context = knowledge_base[know_index]['wikipedia_summary']

        # 이 부분이 예전 코드 그대로 남아있습니다!
        image_name = line["image"]
        base_name = os.path.splitext(image_name)[0]
        
        possible_dirs = [args.image_folder] + [os.path.join(args.image_folder, f"{k:02d}") for k in range(1, 9)]
        
        image_path = None
        for d in possible_dirs:
            search_pattern = os.path.join(d, f"{base_name}.*")
            matches = glob.glob(search_pattern)
            valid_files = [m for m in matches if os.path.isfile(m)]
            if valid_files:
                image_path = valid_files[0]
                break
        
        if image_path is None:
            continue

        choice_str = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in choices.items()])
        prompt = f"Answer the question using a single word based on the given context.\nContext: {context}\nQuestion: {question}{choice_str}\nAnswer:"

        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs, output_attentions=True)
            
        attentions = outputs.attentions 
        num_layers = len(attentions)
        
        # (1) 토큰 인덱스 매핑 (Image, Context, Text 분리) 
        input_ids = inputs.input_ids[0].cpu().numpy()
        input_ids_list = input_ids.tolist()
        
        context_start_idx = -1
        query_start_idx = -1
        
        # 누적 디코딩을 통해 "Context:"와 "Question:"이 등장하는 정확한 토큰 위치 탐색
        for idx_tok in range(len(input_ids_list)):
            text_so_far = processor.decode(input_ids_list[:idx_tok+1])
            
            if "Context:" in text_so_far and context_start_idx == -1:
                context_start_idx = idx_tok
            if "Question:" in text_so_far and query_start_idx == -1:
                query_start_idx = idx_tok
                break 
                
        # [수정 1] Image 인덱스 정확하게 모두 추출 (시작 토큰부터 끝 토큰까지의 범위)
        # Qwen-VL: 151652(<|vision_start|>), 151655(<|vision_end|>)
        # [수정 1] Image 인덱스 정확하게 모두 추출 (버그 픽스)
        # Qwen-VL 시리즈가 이미지에 사용하는 모든 특수 토큰 ID 
        # (vision_start, vision_end, image_pad 등)
        image_tokens_set = {151652, 151653, 151655, 151857, 151858, 151859}
        
        # input_ids 중에서 위 특수 토큰에 해당하는 모든 인덱스를 이미지로 할당 (수백 개가 잡혀야 정상!)
        image_indices = [i for i, tok in enumerate(input_ids_list) if tok in image_tokens_set]
        
        # Fallback (인덱스를 못 찾았을 경우 안전장치)
        if context_start_idx == -1: context_start_idx = len(input_ids_list) // 3
        if query_start_idx == -1: query_start_idx = (len(input_ids_list) // 3) * 2
        if not image_indices: image_indices = list(range(10, context_start_idx - 10))
            
        # Context 인덱스 ("Context:" 부터 "Question:" 직전까지)
        context_indices = list(range(context_start_idx, query_start_idx))
        
        # [수정 2] Text 인덱스: 전체 토큰 중 Image와 Context를 뺀 '나머지 전부'
        # 여기에는 시스템 프롬프트, "Answer the question...", Question 내용, Options, "Answer:" 가 모두 포함됩니다.
        text_indices = [i for i in range(len(input_ids)) if i not in image_indices and i not in context_indices]

        # 터미널에서 3등분 확인용 출력 (Query -> Text로 이름 변경)
        print(f"\n[Token Check] Image: {len(image_indices)}개, Context: {len(context_indices)}개, Text: {len(text_indices)}개")


        # (2) Layer별 Attention 집계
        layer_image_attn = []
        layer_context_attn = []
        layer_text_attn = [] # query 대신 text로 변경
        
        for layer_idx in range(num_layers):
            layer_attn = attentions[layer_idx][0].float() 
            last_token_attn = layer_attn[:, -1, :] 
            avg_head_attn = last_token_attn.mean(dim=0).cpu().numpy()
            
            total_attn = avg_head_attn.sum()
            
            img_attn_ratio = avg_head_attn[image_indices].sum() / total_attn
            ctx_attn_ratio = avg_head_attn[context_indices].sum() / total_attn
            txt_attn_ratio = avg_head_attn[text_indices].sum() / total_attn
            
            layer_image_attn.append(img_attn_ratio)
            layer_context_attn.append(ctx_attn_ratio)
            layer_text_attn.append(txt_attn_ratio)

        # 누적 평균 계산 초기화
        if avg_layer_image_attn is None:
            avg_layer_image_attn = np.zeros(num_layers)
            avg_layer_context_attn = np.zeros(num_layers)
            avg_layer_text_attn = np.zeros(num_layers) # 변수명 변경
            
        avg_layer_image_attn += np.array(layer_image_attn)
        avg_layer_context_attn += np.array(layer_context_attn)
        avg_layer_text_attn += np.array(layer_text_attn) # 변수명 변경

        valid_count += 1
        if valid_count >= total_samples:
            break

    # ==========================================
    # 4. Matplotlib 시각화 (평균치 산출 후)
    # ==========================================
    avg_layer_image_attn /= valid_count
    avg_layer_context_attn /= valid_count
    avg_layer_text_attn /= valid_count

    plt.figure(figsize=(14, 6))
    x = np.arange(1, num_layers + 1)
    width = 0.25 # 막대 두께
    
    # 막대 3개를 겹치지 않게 나란히 배치
    plt.bar(x - width, avg_layer_image_attn, width, label='Image Tokens', color='#FF9999')
    plt.bar(x, avg_layer_context_attn, width, label='Context Tokens', color='#66B2FF')
    plt.bar(x + width, avg_layer_text_attn, width, label='Text Tokens', color='#99FF99')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Average Attention Ratio')
    plt.title(f'Attention Distribution across {valid_count} samples: Image vs Context vs Text')
    plt.xticks(x[::2])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = "attention_qwen3vl_result.png"
    plt.savefig(save_path)
    print(f"\nEvaluation finished! Processed {valid_count} samples.")
    print(f"Visualization saved to {save_path}. Please check this image!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image-folder", type=str, default="../data/infoseek/images")
    parser.add_argument("--dataset", type=str, default="infoseek")
    parser.add_argument("--answers-file", type=str, default="./results/qwen_baseline_result.jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100) 
    
    args = parser.parse_args()
    eval_model(args)
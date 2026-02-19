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
        attn_implementation="eager" 
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Qwen Vision Special Tokens
    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"
    
    # 토크나이저에서 ID 확보
    if vision_start_token in processor.tokenizer.get_vocab():
        vision_start_id = processor.tokenizer.convert_tokens_to_ids(vision_start_token)
        vision_end_id = processor.tokenizer.convert_tokens_to_ids(vision_end_token)
    else:
        # Config에서 fallback
        vision_start_id = getattr(model.config, "vision_start_token_id", 151857)
        vision_end_id = getattr(model.config, "vision_end_token_id", 151858)

    print(f"Vision IDs -> Start: {vision_start_id}, End: {vision_end_id}")


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

    print(f"Start ALFAR evaluation on {len(questions)} items...")

    # ==========================================
    # 3. ALFAR inference loop
    # ==========================================
    # 인덱스 i는 별도로 관리 (tqdm loop와 동기화)
    for i, line in tqdm(enumerate(questions), total=len(questions)):
        if i < args.start_index:
            continue
        idx = line['id']
        question = line["question"]
        choices = line['multiple_choices']
        
        # (1) Context & Image 준비
        if i >= len(indices): break
        
        cur_indices = indices[i]
        know_index = indice_map[cur_indices[0]] 
        context = knowledge_base[know_index]['wikipedia_summary']

        # JSON에 적힌 파일명에서 확장자를 떼고 순수 ID만 추출 (ex: "oven_05050169")
        image_id = os.path.splitext(line["image"])[0] 
        
        import glob
        image_path = None
        
        # 1. 01~08 하위 폴더를 순회하며 검색
        for k in range(1, 9):
            sub_folder = f"{k:02d}"
            # 패턴 예시: data/infoseek/images/infoseek/05/oven_05050169.*
            # 이렇게 하면 .jpg, .JPEG, .png 등을 모두 한꺼번에 찾습니다.
            search_pattern = os.path.join(args.image_folder, sub_folder, f"{image_id}.*")
            matches = glob.glob(search_pattern)
            
            if matches:
                image_path = matches[0]
                break
        
        # 2. (혹시 모르니) 하위 폴더가 아닌 image_folder 바로 아래도 확인
        if image_path is None:
            fallback_pattern = os.path.join(args.image_folder, f"{image_id}.*")
            matches = glob.glob(fallback_pattern)
            if matches:
                image_path = matches[0]
        
        if image_path is None:
            continue

        # (2) 프롬프트 준비
        choice_str = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in choices.items()])
        
        # Qwen-Instruct 포맷
        prompt_ctx = f"Answer the question using a single word based on the given context.\nContext: {context}\nQuestion: {question}{choice_str}\nAnswer:"
        prompt_param = f"Answer the question using a single word.\nQuestion: {question}{choice_str}\nAnswer:"

        messages_ctx = [{"role": "user", "content": [{"type": "image", "image": image_path, "max_pixels": 1003520}, {"type": "text", "text": prompt_ctx}]}]
        messages_param = [{"role": "user", "content": [{"type": "image", "image": image_path, "max_pixels": 1003520}, {"type": "text", "text": prompt_param}]}]

        # (3) Processor 처리 (Qwen-VL 핵심: image_grid_thw)
        text_ctx = processor.apply_chat_template(messages_ctx, tokenize=False, add_generation_prompt=True)
        text_param = processor.apply_chat_template(messages_param, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(messages_ctx)
        
        inputs_ctx = processor(
            text=[text_ctx], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)
        
        inputs_param = processor(
            text=[text_param], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)

        # ==========================================
        # (4) ALFAR Manual Generation Loop
        # ==========================================
        generated_ids = []
        max_new_tokens = 16 # 단답형이므로 짧게 설정
        
        past_key_values_ctx = None
        past_key_values_param = None
        
        # 초기 입력 설정
        curr_inputs_ctx = inputs_ctx
        curr_inputs_param = inputs_param
        
        # Position ID 관리 변수
        cur_pos_ctx = inputs_ctx.input_ids.shape[1]
        cur_pos_param = inputs_param.input_ids.shape[1]

        # Vision Token 위치 찾기 (Contextual 기준)
        input_ids_list = inputs_ctx.input_ids[0].tolist()
        try:
            vis_start = input_ids_list.index(vision_start_id)
            # vis_end는 end token의 바로 다음 인덱스
            vis_end = input_ids_list.index(vision_end_id) + 1
        except ValueError:
            # 이미지가 없을 경우 처리
            vis_start, vis_end = 0, 0

        for step in range(max_new_tokens):
            with torch.inference_mode():
                # --- Forward Calls ---
                outputs_ctx = model(
                    **curr_inputs_ctx,
                    past_key_values=past_key_values_ctx,
                    use_cache=True,
                    output_attentions=True
                )
                
                outputs_param = model(
                    **curr_inputs_param,
                    past_key_values=past_key_values_param,
                    use_cache=True,
                    output_attentions=False 
                )

            # KV Cache 업데이트
            past_key_values_ctx = outputs_ctx.past_key_values
            past_key_values_param = outputs_param.past_key_values

            next_token_logits_ctx = outputs_ctx.logits[:, -1, :]
            next_token_logits_param = outputs_param.logits[:, -1, :]

            # --- ALFAR Attention Logic ---
            # tuple의 마지막 요소가 last layer attention
            last_layer_attn = outputs_ctx.attentions[-1] 
            
            # Attention Shape: [Batch, Head, Query_Len, Key_Len]
            # Key_Len은 past_key_values가 쌓일수록 계속 늘어남
            
            if step == 0:
                # Prefill: Query_Len = 전체 입력 길이. 마지막 토큰(질문 끝)의 Attention을 본다.
                # [-1] 인덱싱으로 마지막 쿼리 선택
                if vis_end > 0:
                    img_attn_score = last_layer_attn[:, :, -1, vis_start:vis_end].mean()
                    # vis_end 이후가 텍스트 컨텍스트 (질문 포함)
                    ctx_attn_score = last_layer_attn[:, :, -1, vis_end:].mean() + 1e-8
                else:
                    img_attn_score = torch.tensor(1.0)
                    ctx_attn_score = torch.tensor(1.0)
            else:
                # Decoding: Query_Len = 1.
                # Key에는 이전 모든 토큰이 포함되어 있음. 절대 위치 인덱스(vis_start:vis_end) 사용 가능
                if vis_end > 0:
                    img_attn_score = last_layer_attn[:, :, 0, vis_start:vis_end].mean()
                    ctx_attn_score = last_layer_attn[:, :, 0, vis_end:].mean() + 1e-8
                else:
                    img_attn_score = torch.tensor(1.0)
                    ctx_attn_score = torch.tensor(1.0)

            # Ratio 계산
            alpha = img_attn_score / ctx_attn_score
            
            # --- Logit Fusion ---
            # diffs = (1+1/alpha)*L_ctx - (1-alpha)*L_param
            
            fused_logits = (1 + 1.0/alpha) * next_token_logits_ctx - (1.0 - alpha) * next_token_logits_param
            
            # Greedy Decoding
            next_token = torch.argmax(fused_logits, dim=-1).unsqueeze(1)
            generated_ids.append(next_token.item())

            # Stop check
            if next_token.item() == model.generation_config.eos_token_id:
                break

            # --- Prepare Next Inputs ---
            # 1. Input IDs: 방금 생성한 토큰
            # 2. Attention Mask: 1 (새로운 토큰은 보임) -> 모델 내부에서 KV Cache 길이만큼 자동 확장됨
            # 3. Position IDs: 수동 증가 필수!
            # 4. 중요: image_grid_thw, pixel_values는 Step 0 이후에는 안 줘도 됨 (이미 KV Cache에 임베딩됨)
            
            curr_inputs_ctx = {
                "input_ids": next_token,
                "attention_mask": torch.ones((1, 1), device=model.device, dtype=torch.long),
                "position_ids": torch.tensor([[cur_pos_ctx]], device=model.device, dtype=torch.long)
            }
            
            curr_inputs_param = {
                "input_ids": next_token,
                "attention_mask": torch.ones((1, 1), device=model.device, dtype=torch.long),
                "position_ids": torch.tensor([[cur_pos_param]], device=model.device, dtype=torch.long)
            }
            
            cur_pos_ctx += 1
            cur_pos_param += 1

        # 결과 저장
        output_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        ans_file.write(json.dumps({
            "data_id": idx,
            "prediction": output_text.strip(),
            "model_id": args.model_path,
            "image": line["image"]
        }) + "\n")
        ans_file.flush()
        torch.cuda.empty_cache()

    ans_file.close()
    print("ALFAR Evaluation finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image-folder", type=str, default="../data/infoseek/images")
    parser.add_argument("--dataset", type=str, default="infoseek")
    parser.add_argument("--answers-file", type=str, default="./results/qwen_alfar_result.jsonl")
    parser.add_argument("--start-index", type=int, default=0)
    
    args = parser.parse_args()
    eval_model(args)
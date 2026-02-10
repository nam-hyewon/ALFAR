import argparse
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Qwen2.5-VL 관련 라이브러리 (최신 transformers 필요)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def eval_model(args):
    # 1. 모델 로딩 (Qwen2.5-VL)
    print(f"Loading Qwen2.5-VL model from: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16, # 실제 실험에서는 다시 bfloat16으로 바꿔주기
        device_map="cpu"  # 가능한 GPU에 자동 할당 -> 실제 실험에서는 다시 auto로 바꿔주기
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 2. 데이터 경로 설정 (기존 로직 유지)
    # 현재 파일 위치 기준으로 data 폴더 찾기
    ROOT = Path(__file__).resolve().parents[0] # experiments 폴더 안에 있다고 가정
    if 'experiments' in str(ROOT):             # 만약 experiments 폴더 안이라면 한 단계 위로
        ROOT = ROOT.parent
        
    DATA_DIR = ROOT / 'data'
    
    # Infoseek 데이터셋 경로 설정
    if args.dataset == 'infoseek':
        question_file = DATA_DIR / 'eval_data' / 'mc' / 'infoseek_debug.json' # 실제 실험에서는 다시 infoseek_mc.json으로 변경해주기
        # 검색(Retrieval) 결과 파일들 (없으면 에러나니 경로 확인 필수)
        indices = np.load(DATA_DIR / 'retrieval_result' / 'infoseek_mc_indices_50_17k.npy', allow_pickle=True)
        indice_map = np.load(DATA_DIR / 'wiki' / 'wiki_map_17k.npy', allow_pickle=True).item()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} support is under construction.")

    # 위키피디아 텍스트 DB 로드
    knowledge_base = np.load(DATA_DIR / 'wiki' / 'wiki_with_image.npy', allow_pickle=True).item()
    
    with open(question_file, "r") as f:
        questions = json.load(f)

    # 결과 저장 파일 준비
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    print(f"Start evaluation on {len(questions)} items...")

    # 3. 추론 루프 (Inference Loop)
    # 디버깅용으로 5개만 먼저 돌려보려면: for idx, line in enumerate(tqdm(questions[:5])):
    i = 0
    for line in tqdm(questions):
        idx = line['id']
        question = line["question"]
        choices = line['multiple_choices']
        
        # (1) Context(배경지식) 가져오기
        # 기존 코드의 로직: retrieval_result에서 가져온 인덱스로 위키피디아 요약문 찾기
        indice = indices[i]
        know_index = indice_map[indice[0]] # Top-1 검색 결과 사용
        context = knowledge_base[know_index]['wikipedia_summary']
        i += 1

        # (2) 프롬프트 구성 (Qwen 스타일)
        # 질문 + 배경지식 + 보기(Options)
        choice_str = "\nOptions:\n"
        for c_name, c_content in choices.items():
            choice_str += f"{c_name}: {c_content}\n"
        
        final_prompt = f"Answer the question using a single word based on the given context.\nContext: {context}\nQuestion: {question}{choice_str}\nAnswer:"

        # (3) 이미지 로드
        image_id = line["image"].split('.')[0]
        # jpg, JPEG, png 등 확장자 처리
        possible_exts = ['.jpg', '.JPEG', '.png']
        image_path = None
        for ext in possible_exts:
            temp_path = os.path.join(args.image_folder, image_id + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                break
        
        if image_path is None:
            # 이미지가 없으면 건너뛰거나 에러 로깅
            print(f"[Warning] Image not found: {image_id}")
            continue

        # (4) Qwen2.5-VL 입력 포맷 만들기
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": final_prompt},
                ],
            }
        ]

        # (5) 전처리 및 생성
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # 생성 (Standard Generation)
        # *주의: ALFAR의 cd_beta 등은 여기서 지원하지 않음. 기본 성능 측정용.
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        # (6) 결과 디코딩
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # (7) 저장
        ans_file.write(json.dumps({
            "data_id": idx,
            "prediction": output_text.strip(),
            "model_id": args.model_path,
            "image": line["image"]
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    print("Evaluation finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 기본값 설정 (Qwen 최신 모델)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image-folder", type=str, default="../data/infoseek/images")
    parser.add_argument("--dataset", type=str, default="infoseek")
    parser.add_argument("--answers-file", type=str, default="./results/qwen_infoseek_result.jsonl")
    args = parser.parse_args()

    eval_model(args)
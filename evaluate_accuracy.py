import json
import re

# 1. 파일 경로 설정
pred_file = "./results/qwen_base_run3.jsonl"
gt_file = "./data/eval_data/mc/infoseek_mc.json" 

def calculate_accuracy():
    # 2. 실제 정답(Ground Truth) 불러오기
    print("원본 데이터에서 정답을 불러오는 중...")
    with open(gt_file, "r", encoding='utf-8') as f:
        gt_data = json.load(f)

    # 데이터 ID를 Key로, 정답(A, B, C, D 등)을 Value로 하는 사전 만들기
    gt_dict = {}
    for item in gt_data:
        gt_dict[item['id']] = str(item.get('multiple_choices_answer', '')).strip().upper()

    # 3. 모델 예측 결과 불러와서 채점하기
    correct_count = 0
    total_count = 0

    print("모델의 답변을 채점하는 중...")
    with open(pred_file, "r", encoding='utf-8') as f:
        for line in f:
            pred_data = json.loads(line)
            data_id = pred_data["data_id"]
            raw_pred = pred_data["prediction"]
            
            # 느낌표(!)나 특수문자 다 떼어내고, 맨 앞의 알파벳 딱 한 글자만 추출 (예: "C!!!!" -> "C")
            clean_pred = re.sub(r'[^a-zA-Z]', '', raw_pred)
            if len(clean_pred) > 0:
                clean_pred = clean_pred[0].upper()
            else:
                clean_pred = "" # 모델이 이상한 말만 했을 경우

            # 정답 비교!
            if data_id in gt_dict:
                gt_answer = gt_dict[data_id]

                if clean_pred == gt_answer:
                    correct_count += 1
                total_count += 1

    # 4. 최종 결과 출력
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("-" * 30)
        print(f"🎯 총 채점 문항: {total_count}개")
        print(f"✅ 정답 수: {correct_count}개")
        print(f"🏆 최종 정답률(Accuracy): {accuracy:.2f}%")
        print("-" * 30)
    else:
        print("채점할 데이터가 매칭되지 않았습니다. 파일 경로를 다시 확인해 주세요!")

if __name__ == "__main__":
    calculate_accuracy()
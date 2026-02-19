from huggingface_hub import login, hf_hub_download
import os

# 1. 인증 
login()

repo_id = "ychenNLP/oven"
save_path = "/scratch/skaodjs/infoseek" 
os.makedirs(save_path, exist_ok=True)

# 2. 01부터 08까지 shard 파일 다운로드
for i in range(1, 9):
    file_name = f"shard{i:02d}.tar"
    print(f"\n📦 {file_name} 다운로드 시작...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            repo_type="dataset",
            local_dir=save_path,
            resume_download=True  # 네트워크가 끊겨도 이어서 다운로드
        )
        print(f"✅ {file_name} 완료!")
    except Exception as e:
        print(f"❌ {file_name} 에러 발생: {e}")

print("\n✨ 모든 파일 다운로드 프로세스가 종료되었습니다.")

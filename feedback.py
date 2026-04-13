import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

with open("swing_data.json", "r") as f:
    data = json.load(f)

prompt = f"""
당신은 전문 야구 타격 코치입니다.
아래는 타자의 스윙 분석 데이터입니다.

[스윙 데이터]
- 총 분석 프레임: {data['total_frames']}
- 힙 최대 회전각도: {data['hip']['max']}도
- 힙 평균 회전각도: {data['hip']['avg']}도
- 어깨 최대 회전각도: {data['shoulder']['max']}도
- Kinetic Chain Gap 최대값: {data['kinetic_chain']['max_gap']}도
- Kinetic Chain Gap 최솟값: {data['kinetic_chain']['min_gap']}도
- 평균 Gap: {data['kinetic_chain']['avg_gap']}도
- 머리 평균 움직임: {data['head']['avg_movement']} (0.02 이하 안정)
- 불안정 프레임 수: {data['head']['unstable_frames']}
- 뒷팔꿈치 평균 거리: {data['elbow']['avg_distance']}
- 뒷무릎 평균 각도: {data['knee']['avg_angle']}도

[분석 기준]
- Kinetic Chain Gap 60도 이상: 좋은 상하체 분리
- Gap 0도 이하: 어깨가 먼저 열리는 문제 (파워 손실)
- 머리 움직임 0.02 이하: 안정적인 헤드
- 팔꿈치 거리 0.15 이하: 몸에 붙어서 나오는 이상적인 폼

위 데이터를 바탕으로 아래 형식으로 피드백을 작성해주세요.

1. 전체 평가 (2-3문장)
2. 잘하고 있는 점
3. 개선이 필요한 점
4. 구체적인 훈련 방법 (1-2가지)

코치가 선수에게 직접 말하는 톤으로, 전문적이지만 이해하기 쉽게 작성해주세요.
"""

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)

print("=" * 50)
print("AI 타격 피드백")
print("=" * 50)
print(message.content[0].text)
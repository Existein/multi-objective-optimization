import boto3
import time
import csv
import base64
import re

# Lambda 클라이언트 생성
lambda_client = boto3.client('lambda', region_name='us-east-1')

# 테스트할 함수명 및 Alias 리스트
function_name = "PostProcessing"
memory_aliases = [f"{m}MB" for m in range(128, 3009, 64)]

# 각 설정당 몇 번씩 호출할지(평균 내기 위함)
num_invocations = 5

# 입력 이벤트 (필요하다면 run_id나 source_key 등 삽입 가능)
# test_event = {
#     "run_id": "profiling-run-001",
#     "test_mode": True
# }

results = []

duration_pattern = re.compile(r'Duration:\s+([\d\.]+)\s+ms')

# 이전 평균 실행 시간 초기화
previous_avg_time = None
early_stop_threshold = 5.0  # 향상률(%)

for alias in memory_aliases:
    function_arn = f"arn:aws:lambda:us-east-1:891376968462:function:{function_name}:{alias}"
    print(f"Testing {function_arn} ...")
    durations = []
    for i in range(num_invocations):
        response = lambda_client.invoke(
            FunctionName=function_arn,
            # Payload=str(test_event).encode('utf-8'),
            LogType='Tail'  # 로그 결과를 포함하도록 함
        )

        # LogResult를 base64 디코딩
        log_result = base64.b64decode(response['LogResult']).decode('utf-8')
        # 로그에서 Duration 추출
        match = duration_pattern.search(log_result)
        if match:
            duration_ms = float(match.group(1))  # ms 단위
            duration_sec = duration_ms / 1000.0
            durations.append(duration_sec)
        else:
            # 매칭 실패 시 기본값(또는 오류 처리)
            durations.append(float('inf'))

    avg_time = sum(durations) / len(durations)
    results.append((alias, avg_time))
    print(f"{alias} average time based on logResult Duration: {avg_time:.4f} sec")

    # Early Stopping 조건 체크
    if previous_avg_time is not None:
        improvement = (previous_avg_time - avg_time) / previous_avg_time * 100
        print(f"Improvement from previous memory setting: {improvement:.2f}%")
        if improvement < early_stop_threshold:
            print(f"Early stopping at {alias}: Improvement {improvement:.2f}% is below {early_stop_threshold}% threshold.")
            break

    previous_avg_time = avg_time

# 결과 정렬(옵션)
results.sort(key=lambda x: int(x[0].replace("MB", "")))

# CSV로 저장 (옵션)
with open("profiling_results_logresult.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["MemoryAlias", "AvgTimeSec"])
    for r in results:
        writer.writerow([r[0], r[1]])

print("Profiling completed. Results saved to profiling_results_logresult.csv")

# 결과 화면 출력
print("Memory Settings vs Avg Time from LogResult:")
for alias, avg_time in results:
    print(f"{alias}: {avg_time:.4f} sec")

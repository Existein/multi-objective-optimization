import boto3
import csv
import time
import base64
import re

# Lambda 클라이언트
lambda_client = boto3.client('lambda', region_name='us-east-1')

# 테스트할 함수명과 메모리 설정
functions = {
    "DataIngest": [128, 192, 256, 320, 384],
    "DataTransform": [128, 192, 256, 320],
    "ModelInference": [128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960],
    "PostProcessing": [128, 192, 256, 320, 384, 448, 512]
}

output_file = "workflow_test_results.csv"

# Duration 추출 정규식 (REPORT 로그 사용)
duration_pattern = re.compile(r'Duration:\s+([\d\.]+)\s+ms')

def invoke_and_get_duration(function_arn):
    """
    해당 Lambda 함수를 호출(LogType='Tail')하고, LogResult에서 Duration(ms)을 추출하여 초 단위로 반환.
    """
    response = lambda_client.invoke(
        FunctionName=function_arn,
        InvocationType='RequestResponse',
        LogType='Tail'
    )
    log_result = base64.b64decode(response['LogResult']).decode('utf-8')
    match = duration_pattern.search(log_result)
    if match:
        duration_ms = float(match.group(1))
        return duration_ms / 1000.0
    else:
        return float('inf')

# CSV 파일 헤더 작성
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Key", "DataIngest Time", "DataTransform Time", "ModelInference Time", "PostProcessing Time", "Total Time"])

# 워크플로우 테스트 실행
for mem1 in functions["DataIngest"]:
    for mem2 in functions["DataTransform"]:
        for mem3 in functions["ModelInference"]:
            for mem4 in functions["PostProcessing"]:
                # 메모리 설정 Key 생성
                key = f"F1: {mem1}MB, F2: {mem2}MB, F3: {mem3}MB, F4: {mem4}MB"

                # 각 함수별 실행 시간 기록용 리스트
                exec_times = []

                # 4개 함수 순서대로 Alias 호출
                for func_name, mem_setting in zip(functions.keys(), [mem1, mem2, mem3, mem4]):
                    alias_name = f"{mem_setting}MB"  # 예: "128MB", "192MB"
                    func_arn = f"arn:aws:lambda:us-east-1:891376968462:function:{func_name}:{alias_name}"

                    # 함수 호출 및 실행 시간 측정(LogResult 기반)
                    duration_sec = invoke_and_get_duration(func_arn)
                    exec_times.append(duration_sec)

                # 총 실행 시간
                total_time = sum(exec_times)

                # CSV에 결과 추가
                with open(output_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([key] + exec_times + [total_time])

print(f"Workflow tests completed. Results saved to {output_file}.")

import boto3
import base64
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import *
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import sys

# Lambda 함수 이름 (이미 alias가 메모리별로 존재한다고 가정)
function_name = "DataTransform"  
# Duration 추출 정규식
duration_pattern = re.compile(r'Duration:\s+([\d\.]+)\s+ms')

# 메모리 설정 시 Lambda 호출을 통한 실행 시간 측정 함수
def get_exec_time(memory):
    # memory를 정수 처리 (혹은 소수점 방지)
    mem_int = int(memory)
    alias = f"{mem_int}MB"
    lambda_client = boto3.client('lambda', region_name='us-east-1')

    # Lambda 함수 호출 (alias 사용)
    response = lambda_client.invoke(
        FunctionName=f"arn:aws:lambda:us-east-1:891376968462:function:{function_name}:{alias}",
        InvocationType='RequestResponse',
        LogType='Tail'
    )
    log_result = base64.b64decode(response['LogResult']).decode('utf-8')
    match = duration_pattern.search(log_result)
    if match:
        duration_ms = float(match.group(1))
        print(f"Fuction duration time (ms):", {duration_ms})
        return duration_ms
    else:
        # 로그 파싱 실패 시 매우 큰 값 반환 (실행 실패 가정)
        return 99999999.0
    
def get_cost(duration, memory):
    # Lambda 비용 계산
    cost = 0.0000002 + (memory / 1024) * 0.000016667 * (duration / 1000)
    return cost

def black_box_function(m):
    exec_time = get_exec_time(m)
    cost = get_cost(exec_time, m)
    
    # 비용이 낮을 수록 좋으므로, target = 1/cost 사용
    return 1.0 / cost
    
    # 아래는 이전 로직
    # 실행 시간이 작을수록 좋으므로, target = 1/exec_time 사용
    # if exec_time == 0:
    #     exec_time = 0.000001
    # return 1.0 / exec_time

def run_bayesian_optimization():
    pbounds = { 'm': (128, 3008) }
    acq=acquisition.ExpectedImprovement(xi=0.001)
    
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        acquisition_function=acq,
        verbose=2,
        random_state=46,
        allow_duplicate_points=True
    )

    # init point 선정
    init_points = [128, 1216, 3008]
    counter = 0
    
    for init_m in init_points:
        counter += 1
        print(f"Next point to probe is: {init_m}")
        target = black_box_function(init_m)
        optimizer.register(params={'m': init_m}, target=target)
        print(f"Samples so far: {counter}")

    # 메인 최적화 루프
    total_samples = 20

    while counter < total_samples:
        # 다음 탐색 지점 추천
        next_point = optimizer.suggest()

        # 64MB 단위로 조정
        next_point['m'] = int(round(next_point["m"] / 64.0) * 64)
        next_point['m'] = min(max(next_point["m"], 128), 3008)  # 범위 제한
        print(f"Next point to probe is: {next_point}")

        # 실행 및 결과 등록
        target = black_box_function(**next_point)
        print("Found the cost to be:", 1/target)
        next_point['m'] = next_point['m']+random()
        next_point['m'] = min(max(next_point["m"], 128), 3008)  # 범위 제한
        optimizer.register(params=next_point, target=target)

        # 샘플 수 업데이트
        counter += 1
        print(f"Samples so far: {counter}")

    # 최적값 출력
    best = optimizer.max
    print("BEST RESULT:")
    print("Memory:", best["params"]["m"], "MB")
    print("Cost:", 1.0 / best["target"], "USD")

    # 결과 플롯
    results = optimizer.res
    memory_values = [r["params"]["m"] for r in results]
    targets = [r["target"] for r in results]
    costs = [1.0 / t for t in targets]

    df = pd.DataFrame({
        "sample": range(1, len(results) + 1),
        "memory": memory_values,
        "cost": costs,
    })

    plt.figure(figsize=(8, 4), dpi=300)
    sns.scatterplot(data=df, x="sample", y="cost", hue="memory", palette="viridis", edgecolor="black")
    plt.yscale("log")
    plt.xlabel("Sample")
    plt.ylabel("Cost (dollar, log scale)")
    plt.title("Bayesian Optimization of Memory Setting for AWS Lambda")
    plt.tight_layout()
    plt.savefig("bo_result_64mb.png")
    plt.close()

if __name__ == "__main__":
    # 터미널 아웃풋을 result.txt로 리다이렉트
    with open("result.txt", "w") as f:
        sys.stdout = f
        run_bayesian_optimization()
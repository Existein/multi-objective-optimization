import requests
import boto3
import json
import time

# AWS client 생성
logs_client = boto3.client('logs', region_name='us-east-1')
lambda_client = boto3.client('lambda', region_name='us-east-1')


# Lambda Function URL
LAMBDA_FUNCTION_URL = "https://i4wqbvsnsygb3zeejtlpjx66bi0pyqar.lambda-url.us-east-1.on.aws/"

# memory configuration list
config_list = [128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 
               1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048, 2112, 2176, 2240, 
               2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880, 2944, 3008]

# Lambda 함수 메모리 크기 변경
def update_lambda_memory(function_name, memory_size):
    try:
        response = lambda_client.update_function_configuration(
            FunctionName=function_name,
            MemorySize=memory_size
        )
        print(f"Updated Lambda memory size to {memory_size}MB for function {function_name}.")
        return True
    except Exception as e:
        print(f"Failed to update memory size for function {function_name}: {str(e)}")
        return False
    
# Lambda Function URL 호출 (로그는 나중에 처리)
def invoke_lambda(memory_size, iteration):
    # HTTP POST 요청을 Lambda Function URL로 전송
    response = requests.post(
        LAMBDA_FUNCTION_URL,
        json={'memory_size': memory_size}  # 필요한 인자를 body에 넣음
    )
    
    if response.status_code == 200:
        print(f"Lambda function with {memory_size}MB executed successfully (Iteration: {iteration}).")
        return True
    else:
        print(f"Error invoking Lambda function with memory {memory_size}MB (Iteration: {iteration}): {response.text}")
        return False

# 로그에서 실행 시간 추출 (모든 Lambda 함수 호출이 끝난 후)
def get_all_execution_times(log_group):
    # 로그 스트림에서 실행 시간(Duration)을 추출
    logs = logs_client.filter_log_events(
        logGroupName=log_group,
        filterPattern="REPORT"
    )
    
    execution_times = []
    
    # 모든 로그 이벤트에서 실행 시간 추출
    for event in logs['events']:
        message = event['message']
        if 'REPORT' in message:
            duration_str = message.split('Billed Duration: ')[1].split(' ms')[0]
            execution_time = float(duration_str)
            execution_times.append(execution_time)
    
    return execution_times

# 벤치마크 실행 (각 메모리 설정에서 10회 반복, 로그는 나중에 처리)
def run_benchmark():
    log_group = "/aws/lambda/io_intensive_benchmark"  # 로그 그룹 설정
    
    # Lambda 함수 실행만 진행
    for memory in config_list:
        update_lambda_memory("io_intensive_benchmark", memory)  # 메모리 크기 변경 
        print(f"Running Lambda function with {memory}MB of memory...")
        
        for i in range(1, 11):  # 1부터 10까지 반복 실행
            invoke_lambda(memory, i)  # 몇 번째 실행인지 전달
    
    print("Benchmark completed.")

if __name__ == "__main__":
    run_benchmark()

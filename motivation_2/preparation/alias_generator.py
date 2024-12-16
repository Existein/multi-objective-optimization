import boto3
import time

lambda_client = boto3.client('lambda', region_name='us-east-1')

function_arns = [
    # "arn:aws:lambda:us-east-1:891376968462:function:DataIngest",
    # "arn:aws:lambda:us-east-1:891376968462:function:DataTransform",
    "arn:aws:lambda:us-east-1:891376968462:function:ModelInference",
    # "arn:aws:lambda:us-east-1:891376968462:function:PostProcessing"
]

# 128MB부터 3008MB까지 64MB 단위로 증가
# 128, 192, 256, ... 3008
memory_sizes = range(128, 3009, 64)

for func_arn in function_arns:
    print(f"Processing function: {func_arn}")
    for mem_size in memory_sizes:
        # 함수 메모리 설정 변경
        lambda_client.update_function_configuration(
            FunctionName=func_arn,
            MemorySize=mem_size
        )

        # 업데이트 완료 대기
        waiter = lambda_client.get_waiter('function_updated')
        waiter.wait(FunctionName=func_arn)

        # 새로운 버전을 퍼블리시
        publish_resp = lambda_client.publish_version(FunctionName=func_arn)
        new_version = publish_resp['Version']

        # 메모리 크기를 별도 Alias로 연결
        alias_name = f"{mem_size}MB"
        lambda_client.create_alias(
            FunctionName=func_arn,
            Name=alias_name,
            FunctionVersion=new_version,
            Description=f"Alias for memory size {mem_size}MB"
        )

        print(f"Created alias '{alias_name}' for {func_arn} version {new_version}")

        # 작은 대기(선택사항): 많은 버전 퍼블리시 시 람다 API에 부담 덜기 위함
        time.sleep(0.5)

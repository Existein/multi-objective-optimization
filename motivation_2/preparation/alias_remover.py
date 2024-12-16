import boto3

# Boto3 Lambda 클라이언트 생성
lambda_client = boto3.client('lambda', region_name='us-east-1')

# Lambda 함수 ARNs 리스트
function_arns = [
    "arn:aws:lambda:us-east-1:891376968462:function:ModelInference",
]

# Alias 및 관련 버전 삭제 로직
for func_arn in function_arns:
    print(f"Processing function: {func_arn}")

    # 함수에 연결된 모든 alias 가져오기
    aliases = lambda_client.list_aliases(FunctionName=func_arn)['Aliases']

    for alias in aliases:
        alias_name = alias['Name']
        print(f"Deleting alias: {alias_name} for {func_arn}")

        # Alias 삭제
        lambda_client.delete_alias(FunctionName=func_arn, Name=alias_name)
        print(f"Deleted alias: {alias_name}")

    # 함수에 연결된 모든 버전 가져오기
    versions = lambda_client.list_versions_by_function(FunctionName=func_arn)['Versions']

    for version in versions:
        version_number = version['Version']

        # $LATEST는 삭제하지 않음
        if version_number == "$LATEST":
            continue

        print(f"Deleting version: {version_number} for {func_arn}")

        # 버전 삭제
        lambda_client.delete_function(FunctionName=f"{func_arn}:{version_number}")
        print(f"Deleted version: {version_number}")

print("All aliases and versions deleted successfully.")

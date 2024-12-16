import json
import os
import boto3
from datetime import datetime

s3 = boto3.client('s3')
dynamodb = boto3.client('dynamodb')

def lambda_handler(event, context):
    bucket = os.environ['DATA_BUCKET']
    stage3_key = os.environ['STAGE3_KEY']
    result_table = os.environ['RESULT_TABLE']

    # event에서 run_id 추출
    run_id = event.get('run_id', 'default-run-id')

    # F3 결과 읽기
    response = s3.get_object(Bucket=bucket, Key=stage3_key)
    data = json.load(response['Body'])

    count = len(data)
    timestamp = datetime.now().isoformat()

    # DynamoDB에 결과 기록
    dynamodb.put_item(
        TableName=result_table,
        Item={
            'run_id': {'S': run_id},
            'result_count': {'N': str(count)},
            'timestamp': {'S': timestamp}
        }
    )

    return {
        'statusCode': 200,
        'message': 'Post processing completed',
        'inserted_count': count,
        'run_id': run_id,
        'timestamp': timestamp
    }

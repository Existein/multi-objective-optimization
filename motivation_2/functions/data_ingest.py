import json
import os
import boto3
import io
from datetime import datetime

s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucket = os.environ['DATA_BUCKET']
    source_key = os.environ['SOURCE_KEY']
    stage1_key = os.environ['STAGE1_KEY']

    # event에서 run_id 추출
    run_id = event.get('run_id', 'default-run-id')

    # S3에서 raw 데이터 가져오기
    response = s3.get_object(Bucket=bucket, Key=source_key)
    raw_data = json.load(response['Body'])

    # null value 제거
    cleaned_data = [item for item in raw_data if item.get('value') is not None]

    # 정제된 데이터 S3 저장
    out_bytes = io.BytesIO(json.dumps(cleaned_data).encode('utf-8'))
    s3.put_object(Bucket=bucket, Key=stage1_key, Body=out_bytes)

    # run_id 포함하여 반환
    return {
        'statusCode': 200,
        'message': 'Data ingest completed',
        'run_id': run_id,
        'next_input': {
            'stage_key': stage1_key,
            'run_id': run_id
        }
    }

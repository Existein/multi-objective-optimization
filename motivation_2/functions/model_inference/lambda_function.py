import json
import os
import io
import boto3
import pickle
from datetime import datetime

s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucket = os.environ['DATA_BUCKET']
    stage2_key = os.environ['STAGE2_KEY']
    stage3_key = os.environ['STAGE3_KEY']
    model_key = os.environ['MODEL_KEY']
    vec_key = os.environ['VEC_KEY']

    # event에서 run_id 추출
    run_id = event.get('run_id', 'default-run-id')

    # F2 결과 읽기
    response = s3.get_object(Bucket=bucket, Key=stage2_key)
    data = json.load(response['Body'])

    # 모델, 벡터라이저 S3에서 로드
    model_obj = s3.get_object(Bucket=bucket, Key=model_key)
    model = pickle.loads(model_obj['Body'].read())

    vec_obj = s3.get_object(Bucket=bucket, Key=vec_key)
    vectorizer = pickle.loads(vec_obj['Body'].read())

    # 'value'를 벡터라이저에 통과시킨 뒤 모델.predict
    texts = [item['value'] for item in data]
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    # label 추가
    results = []
    for item, pred in zip(data, preds):
        new_item = dict(item)
        new_item['label'] = pred  # pred가 'spam' 또는 'ham'
        results.append(new_item)

    # 결과 저장
    out_bytes = io.BytesIO(json.dumps(results).encode('utf-8'))
    s3.put_object(Bucket=bucket, Key=stage3_key, Body=out_bytes)

    # run_id 포함하여 반환
    return {
        'statusCode': 200,
        'message': 'Model inference completed',
        'run_id': run_id,
        'next_input': {
            'stage_key': stage3_key,
            'run_id': run_id
        }
    }

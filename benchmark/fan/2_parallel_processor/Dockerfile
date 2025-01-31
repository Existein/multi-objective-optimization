FROM public.ecr.aws/lambda/python:3.9

# 필요한 패키지 설치
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# 함수 코드 복사
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Lambda 핸들러 설정
CMD ["lambda_function.lambda_handler"]
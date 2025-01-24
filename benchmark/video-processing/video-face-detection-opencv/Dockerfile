FROM public.ecr.aws/lambda/python:3.9

# OpenCV 의존성 설치
RUN yum update -y && \
    yum install -y mesa-libGL && \
    yum clean all

# 필요한 파일들 복사
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT} 
COPY haarcascade_frontalface_default.xml ${LAMBDA_TASK_ROOT}

# Python 패키지 설치
RUN pip install --platform manylinux2014_x86_64 --target "${LAMBDA_TASK_ROOT}" --implementation cp --python-version 3.9 --only-binary=:all: --upgrade -r requirements.txt

# Lambda 핸들러 설정
CMD ["lambda_function.lambda_handler"]


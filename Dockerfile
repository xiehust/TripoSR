ARG BASE_IMAGE
ARG PIP_INDEX

# Use the BASE_IMAGE argument
FROM ${BASE_IMAGE}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*


# Make PIP_INDEX available after FROM
ARG PIP_INDEX
# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training


# 设置工作目录
WORKDIR /app

# 复制整个tsr目录到容器内的/app

COPY tsr /app/tsr
COPY server.py /app
COPY serve /app

# 安装新的依赖
COPY requirements.txt /app
RUN pip install -r requirements.txt  

RUN chmod +x /app/serve

# 让端口8080在容器外可用
EXPOSE 8080

# 定义环境变量
ENV PATH="/app:${PATH}"

# 运行serve
ENTRYPOINT []
CMD ["serve"]



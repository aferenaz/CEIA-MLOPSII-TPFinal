FROM python:3.11-slim
WORKDIR /app
COPY dockerfiles/streaming/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY grpc_service/protos ./protos
COPY streaming ./streaming
RUN python -m grpc_tools.protoc -I protos \
      --python_out=streaming --grpc_python_out=streaming protos/forecast.proto \
 && sed -i 's/^import forecast_pb2/from streaming import forecast_pb2/' streaming/forecast_pb2_grpc.py

FROM python:3.11-slim
WORKDIR /app
COPY dockerfiles/grpc/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/gout_common ./gout_common
COPY grpc_service ./grpc_service
RUN python -m grpc_tools.protoc -I grpc_service/protos \
      --python_out=grpc_service --grpc_python_out=grpc_service \
      grpc_service/protos/forecast.proto \
 && sed -i 's/^import forecast_pb2/from grpc_service import forecast_pb2/' grpc_service/forecast_pb2_grpc.py
EXPOSE 50051
CMD ["python", "-m", "grpc_service.server"]

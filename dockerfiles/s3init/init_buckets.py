
"""Crea los buckets de MinIO y sube los CSV de ./data a data/raw/.
Reemplaza al cliente mc (las imágenes minio/mc dejaron de estar disponibles en Docker Hub).
Reintenta hasta que MinIO responda, así que también funciona como chequeo de disponibilidad."""
import os
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3', endpoint_url=os.environ['S3_ENDPOINT'],
                  aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
                  aws_secret_access_key=os.environ['MINIO_SECRET_ACCESS_KEY'],
                  region_name='us-east-1')

for intento in range(60):
    try:
        s3.list_buckets()
        break
    except Exception as e:
        print(f'Esperando a MinIO ({intento + 1}/60): {type(e).__name__}', flush=True)
        time.sleep(2)
else:
    raise SystemExit('MinIO no respondió')

for bucket in (os.environ['MLFLOW_BUCKET_NAME'], os.environ['DATA_REPO_BUCKET_NAME']):
    try:
        s3.create_bucket(Bucket=bucket)
        print('Bucket creado:', bucket)
    except ClientError as e:
        if e.response['Error']['Code'] not in ('BucketAlreadyOwnedByYou', 'BucketAlreadyExists'):
            raise
        print('Bucket existente:', bucket)

csvs = sorted(Path('/seed').glob('*.csv'))
for f in csvs:
    s3.upload_file(str(f), os.environ['DATA_REPO_BUCKET_NAME'], f'raw/{f.name}')
    print('Subido:', f.name)
if not csvs:
    print('Sin CSV en ./data')

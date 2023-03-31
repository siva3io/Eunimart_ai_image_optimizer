import os
import json
import uuid
import boto3
import logging
import datetime
from config import Config
from botocore.client import Config as boto_client_config
# from google.cloud import storage
from boto3.session import Session
from urllib.parse import urlparse

s3 = boto3.client("s3",aws_access_key_id=os.environ['AWS_ACCESS_KEY'] ,aws_secret_access_key=os.environ['AWS_SECRET_KEY'],region_name=os.environ['AWS_REGION'],config=boto_client_config(signature_version='s3v4'))

def catch_exceptions(func):
    def wrapped_function(*args, **kargs):
        try:
            return func(*args, **kargs)
        except Exception as e:
            l = logging.getLogger(func.__name__)
            l.error(e, exc_info=True)
            return None                
    return wrapped_function

# def download_from_google_storage(bucket_name,source_blob_name,destination_file_name):
#     """Downloads a blob from the bucket."""
    
#     destination_dir = '/'.join(destination_file_name.split('/')[:-1])
#     os.makedirs(destination_dir, exist_ok=True)
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)


def download_from_s3(path,file_name):
    s3.download_file(Config.AWS_MODELS,path,file_name)

def get_signed_url(key):
    url = s3.generate_presigned_url(
    ClientMethod='get_object',
        Params={
            'Bucket': os.environ['AWS_BUCKET'],
            'Key': key
        }
    )
    return url

def upload_to_s3(file_data,request_data, file_name):
    date_obj_year = datetime.date.today().year
    date_obj_month = datetime.date.today().month
    date_obj_date = datetime.date.today().day

    key = request_data["account_id"]+'/images/'+str(date_obj_year)+"/"+str(date_obj_month)+"/"+str(date_obj_date)+"/"+str(request_data["channel_id"])+"/"+str(request_data["sku_id"])+"/"+file_name
    s3_response = s3.put_object(Bucket=Config.BOTO3_BUCKET, Key=key, Body=file_data)
    signed_url = get_signed_url(key)
    return signed_url

import os
import boto3

class S3:
    def __init__(self):
        self.__session = boto3._get_default_session()
    
    def list_files_from_s3(self, bucket, prefix=""):
        objects = []
        next_token = ""
        list_parameters = {
            "Bucket": bucket,
            "Prefix": prefix
        }

        client = self.__session.client("s3")

        while next_token is not None:
            if next_token:
                list_parameters.update({'ContinuationToken': next_token})

            results = client.list_objects_v2(**list_parameters)
            objects.extend(results["Contents"])
            next_token = results.get("NextContinuationToken")

        return objects

    def download_files(self, bucket):
        client = self.__session.resource("s3")

        for file_object in self.list_files_from_s3(bucket):
            parent_dir = os.path.dirname(file_object["Key"])

            os.makedirs(parent_dir, exist_ok=True)

            try:
                print(f"[INFO] Donwloading file {file_object['Key']}.")
                client.meta.client.download_file(bucket, file_object["Key"], file_object["Key"])
            except IsADirectoryError:
                os.makedirs(file_object["Key"], exist_ok=True)
            except NotADirectoryError:
                pass
    
    def upload_files(self, bucket, local_folder, previous_folder=""):
        client = self.__session.resource("s3")

        for entry in os.scandir(local_folder):
            folder_name = os.path.basename(local_folder)

            if previous_folder:
                folder_name = f"{previous_folder}/{folder_name}" 

            if entry.is_dir():
                self.upload_files(bucket, entry.path, folder_name)
            else:
                destination = entry.path.split(folder_name)[-1].replace("\\", "/")[1:]
                print(f"[INFO] Uploading file {entry.path} to s3://{bucket}/{folder_name}/{destination}")
                client.meta.client.upload_file(entry.path, bucket, f"{folder_name}/{destination}")

    def create_bucket(self, bucket):
        client = self.__session.client("s3")
        try:
            client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={
                    "LocationConstraint": self.__session.region_name,
                }
            )
            print(f"[INFO] Bucket {bucket} created.")
        except client.exceptions.BucketAlreadyOwnedByYou:
            print(f"[INFO] Bucket {bucket} already exists.")
        
        return bucket
    
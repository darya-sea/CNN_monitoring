import os
import boto3

class S3:
    def __init__(self):
        self.__session = boto3._get_default_session()
    
    def list_files_from_s3(self, bucket_name, prefix=""):
        objects = []
        next_token = ""
        list_parameters = {
            "Bucket": bucket_name,
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

    def download_files(self, bucket_name):
        client = self.__session.resource("s3")

        for file_object in self.list_files_from_s3(bucket_name):
            parent_dir = os.path.dirname(file_object["Key"])

            os.makedirs(parent_dir, exist_ok=True)

            try:
                print(f"[INFO] Donwloading file {file_object['Key']}.")
                client.meta.client.download_file(bucket_name, file_object["Key"], file_object["Key"])
            except IsADirectoryError:
                os.makedirs(file_object["Key"], exist_ok=True)
            except NotADirectoryError:
                pass
    
    def upload_files(self, bucket_name, local_folder, previous_folder=""):
        client = self.__session.resource("s3")

        for entry in os.scandir(local_folder):
            folder_name = os.path.basename(local_folder)

            if previous_folder:
                folder_name = f"{previous_folder}/{folder_name}" 

            if entry.is_dir():
                self.upload_files(bucket_name, entry.path, folder_name)
            else:
                destination = entry.path.split(folder_name)[-1].replace("\\", "/")[1:]
                print(f"[INFO] Uploading file {entry.path} to s3://{bucket_name}/{folder_name}/{destination}")
                client.meta.client.upload_file(entry.path, bucket_name, f"{folder_name}/{destination}")

    def create_bucket(self, bucket_name):
        client = self.__session.client("s3")
        try:
            client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    "LocationConstraint": self.__session.region_name,
                }
            )
            print(f"[INFO] Bucket {bucket_name} created.")
        except client.exceptions.BucketAlreadyOwnedByYou:
            print(f"[INFO] Bucket {bucket_name} already exists.")
        
        return bucket_name

    def delete_bucket(self, bucket_name):
        resource = self.__session.resource('s3')
        client = self.__session.client('s3')

        bucket = resource.Bucket(bucket_name)

        try:
            bucket.objects.all().delete()
            bucket.delete()
        except client.exceptions.NoSuchBucket as err:
            pass

        print(f"[INFO] Bucket {bucket_name} deleted.")
        return bucket
    
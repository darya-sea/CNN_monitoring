import sys
import config
import logging
import warnings
import os
import time
import subprocess

from pprint import pprint
from aws.s3 import S3
from aws.ec2 import EC2
from aws.ssm import SSM

warnings.filterwarnings('ignore')

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


def prepare():
    from prepare.data import PrepareData
    PrepareData(config.CNN_FOLDER, config.DATA_FOLDER).prepare_images()

def predict(image_path: str):
    from train.prediction import Prediction
    from visualization.visualization import Visualization

    predict = Prediction()
    visualization = Visualization()

    models_path = os.path.join(config.DATA_FOLDER, "output/models")
    model_file = predict.get_best_model(models_path)
    plant_types_file = os.path.join(config.DATA_FOLDER, "output/models/train_data_types.json")

    if model_file:
        plant_types = predict.load_classes(plant_types_file)
        resutls = predict.predict(image_path, model_file)
        if resutls and plant_types:
            visualization.show_predicted_images(resutls, plant_types)
            #pprint(resutls)

def train():
    from train.train import Train
    from visualization.visualization import Visualization

    training = Train(config.DATA_FOLDER)
    visualization = Visualization()
    
    history_path = os.path.join(config.DATA_FOLDER, "output")

    train_generator = training.get_data_generator("train")
    validation_generator = training.get_data_generator("validation")

    if train_generator and validation_generator:
        history = training.train(
            train_generator,
            validation_generator,
            config.TRAINING_EPOCHS
        )
        visualization.save_traning_plot(history, history_path)
        visualization.save_history(history, history_path)

def sync_s3(local_folder: str = None):
    s3 = S3()
    s3.create_bucket(config.S3_BUCKET)

    if local_folder:
        if os.path.exists(local_folder):
            s3.upload_files(config.S3_BUCKET, local_folder)
        else:
            print(f"[ERROR] Local folder {local_folder} not found.")
    else:
        s3.download_files(config.S3_BUCKET)

def clean_up():
    ec2 = EC2()
    s3 = S3()
    ssm = SSM()

    ec2.cancel_spot_fleet_request()
    #s3.delete_bucket(config.S3_BUCKET)
    #ec2.delete_volume()
    ec2.delete_launch_templat()
    ec2.delete_instance_profile()
    ssm.delete_logs()

def request_spot():
    ec2 = EC2()
    ssm = SSM()

    ec2.create_volume()
    ec2.create_instance_profile()
    ec2.create_launch_template()
    ec2.create_spot_fleet_role()
    ec2.request_spot_fleet(config.EC2_INSTANCE_TYPES, config.EC2_MAX_PRICE)

    while True:
        if (spot_request := ec2.get_active_spot_fleet_request()):
            if spot_request["ActivityStatus"] == "fulfilled":
                for instance in ec2.get_spot_fleet_instances(spot_request["SpotFleetRequestId"]):
                    time.sleep(3)

                    device_name = ec2.attach_volume(instance["InstanceId"])[2]
                    output = ssm.execute_command(
                        instance["InstanceId"],
                        [
                            f"mount {device_name} /mnt",
                            f"mkfs.ext4 {device_name}",
                            f"mount {device_name} /mnt",
                            "yum install opencv-python -y",
                            "pip3 install virtualenv"
                        ]
                    )
                    if output["StandardErrorContent"]:
                        print(output["StandardErrorContent"])
                    if output["StandardOutputContent"]:
                        print(output["StandardOutputContent"])

                    output = ssm.execute_command(
                        instance["InstanceId"],
                        [
                            "cd /mnt",
                            "git clone https://github.com/darya-sea/CNN_monitoring.git",
                            "cd /mnt/CNN_monitoring",
                            "touch /var/log/train.log",
                            "sh /mnt/CNN_monitoring/application/scripts/install.sh > /var/log/train.log 2>&1",
                            f"AWS_DEFAULT_REGION={ec2.get_session().region_name} sh /mnt/CNN_monitoring/application/scripts/send_logs.sh &",
                            f"aws s3 sync s3://{config.S3_BUCKET}/DATA /mnt/CNN_monitoring/DATA >> /var/log/train.log 2>&1",
                            "sh /mnt/CNN_monitoring/application/scripts/train.sh >> /var/log/train.log 2>&1"
                        ]
                    )
                    if output["StandardErrorContent"]:
                        print(output["StandardErrorContent"])
                    if output["StandardOutputContent"]:
                        print(output["StandardOutputContent"])
                break
            else:
                print("[INFO] Waiting for request to be fulfilled.")
        time.sleep(3)

    # ec2.cancel_spot_fleet_request()
    # ec2.delete_volume()

def get_spot_logs():
   ssm = SSM()
   while True:
    for log in ssm.get_logs():
        message = log["message"].strip()
        print(message)
    time.sleep(5)

def send_spot_logs(log_path: str):
    ssm = SSM()

    if not os.path.exists(log_path):
        print(f"[ERROR] Log file {log_path} not found.")
        return

    process = subprocess.Popen(
        ["tail", "-f", log_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    while True:
        ssm.send_log(process.stdout.readline())

def train_history():
    from visualization.visualization import Visualization

    visualization = Visualization()
    visualization.show_traning_plot(
        os.path.join(
            config.DATA_FOLDER,
            "output/model_history.json"
        )
    )

def help(script_name: str):
    print(
    f"""
        usage: {script_name} <prepare|train|predict|sync|request_spot|get_spot_logs|clean_up|train_history|send_spot_logs>

        preapre example: 
          python {script_name} prepare
        train example: 
          python {script_name} train
        predict example: 
          python {script_name} predict "../CNN/heřmánkovec nevonný/2022_09_21 hermankovec/00257C.tif"
        sync example: 
          python {script_name} sync
          python {script_name} sync CNN
        request_spot example: 
          python {script_name} request_spot
        get_spot_logs example: 
          python {script_name} spot_logs
        clean_up example: 
          python {script_name} clean_up
        train_history example: 
          python {script_name} train_history
        send_spot_logs example: 
          python {script_name} tail_logs
    """
    )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        match sys.argv[1]:
            case "help":
                help(sys.argv[0])
            case "prepare":
                prepare()
            case "train":
                train()
            case "predict":
                if len(sys.argv) > 2:
                    predict(sys.argv[2])
                else:
                    print(f"usage: {sys.argv[0]} predict <image_path>")
            case "sync":
                sync_s3(sys.argv[2] if len(sys.argv) > 2 else None)
            case "request_spot":
                request_spot()
            case "get_spot_logs":
                get_spot_logs()
            case "send_spot_logs":
                send_spot_logs("/var/log/train.log")
            case "clean_up":
                clean_up()
            case "train_history":
                train_history()
    else:
        help(sys.argv[0])

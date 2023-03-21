import sys
import config
import logging
import warnings
import os
import time

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

def predict(image_path):
    from train.prediction import Prediction
    from visualization.visualization import Visualization

    predict = Prediction()
    visualization = Visualization()

    models_path = os.path.join(config.DATA_FOLDER, "output/models")
    model_file = predict.get_best_model(models_path)
    plant_types_file = os.path.join(config.DATA_FOLDER, "output/models/train_plant_types.json")

    if model_file:
        plant_types = predict.load_classes(plant_types_file)
        resutls = predict.predict(image_path, model_file)
        if resutls:
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
        visualization.plot_accuracy(history, history_path)
        visualization.save_history(history, history_path)

def sync_s3(local_folder=None):
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

    ec2.cancel_spot_fleet_request()
    #s3.delete_bucket(config.S3_BUCKET)
    # ec2.delete_volume()

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
                            f"mount {device_name} /mnt > /var/log/train.log 2>&1",
                            f"mkfs.ext4 {device_name} >> /var/log/train.log 2>&1",
                            f"mount {device_name} /mnt >> /var/log/train.log 2>&1",
                            "cd /mnt >> /var/log/train.log >> /var/log/train.log 2>&1",
                            "git clone https://github.com/darya-sea/CNN_monitoring.git >> /var/log/train.log 2>&1",
                            "cd /mnt/CNN_monitoring >> /var/log/train.log 2>&1",
                            f"aws s3 sync s3://{config.S3_BUCKET}/DATA DATA >> /var/log/train.log 2>&1",
                            "yum install opencv-python -y >> /var/log/train.log 2>&1",
                            "pip3 install virtualenv >> /var/log/train.log 2>&1",
                            "git checkout crop >> /var/log/train.log 2>&1",
                            "sh /mnt/CNN_monitoring/application/install.sh >> /var/log/train.log 2>&1",
                            "sh /mnt/CNN_monitoring/application/train.sh >> /var/log/train.log 2>&1"
                        ]
                    )
                    output = ssm.execute_command(
                        instance["InstanceId"], 
                        ["cat /var/log/train.log"]
                    )
                    print(output["StandardOutputContent"])
                break
            else:
                print("[INFO] Waiting for request to be fulfilled.")
        time.sleep(3)

    # ec2.cancel_spot_fleet_request()
    # ec2.delete_volume()

def spot_status():
    ec2 = EC2()
    ssm = SSM()

    if (spot_request := ec2.get_active_spot_fleet_request()):
        if spot_request["ActivityStatus"] == "fulfilled":
            for instance in ec2.get_spot_fleet_instances(spot_request["SpotFleetRequestId"]):
                output = ssm.execute_command(
                    instance["InstanceId"], 
                    ["tail -n 20 /var/log/train.log"]
                )
                print(output["StandardOutputContent"].replace("--output truncated--", ""))

def show_history():
    from visualization.visualization import Visualization

    visualization = Visualization()
    visualization.show_from_json(
        os.path.join(config.DATA_FOLDER, "output/model_history.json")
    )

def help(script_name):
    print(
    f"""
        usage: {script_name} <prepare|train|predict|sync|request_spot|spot_status|clean_up>

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
        spot_status example: 
          python {script_name} spot_status
        clean_up example: 
          python {script_name} clean_up
        show_history example: 
          python {script_name} show_history
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
            case "spot_status":
                spot_status()
            case "clean_up":
                clean_up()
            case "show_history":
                show_history()
                
    else:
        help(sys.argv[0])

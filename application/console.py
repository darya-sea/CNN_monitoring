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
    classes_path = os.path.join(config.DATA_FOLDER, "validation_classes.json")

    model_file = predict.get_best_model(models_path)

    if model_file:
        classes = predict.load_classes(classes_path)
        resutls = predict.predict(image_path, classes, model_file)
        if resutls:
            visualization.show_predicted_images(resutls)
            pprint(resutls)

def train():
    from train.train import Train
    from visualization.visualization import Visualization

    training = Train(config.DATA_FOLDER)
    visualization = Visualization()
    
    classes_path = os.path.join(config.DATA_FOLDER, "validation_classes.json")
    history_path = os.path.join(config.DATA_FOLDER, "output")

    train_generator, validation_generator = training.get_train_generator()
    if train_generator and validation_generator:
        training.save_classes(validation_generator, classes_path)
        history = training.train(train_generator, validation_generator, config.TRAINING_EPOCHS)
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

def prepare_spot():
    ec2 = EC2()
    ec2.create_volume()
    ec2.create_instance_profile()
    ec2.create_launch_template()
    ec2.create_spot_fleet_role()

def request_spot():
    ec2 = EC2()
    ssm = SSM()

    ec2.create_volume()
    ec2.request_spot_fleet()

    while True:
        if (spot_request := ec2.get_active_spot_fleet_request()):
            if spot_request["ActivityStatus"] == "fulfilled":
                for instance in ec2.get_spot_fleet_instances(spot_request["SpotFleetRequestId"]):
                    time.sleep(3)
                    device_name = ec2.attach_volume(instance["InstanceId"])[2]

                    if (command_id := ssm.get_command_id()):
                        for output in ssm.get_command_invocation(command_id, instance["InstanceId"]):
                            print(output["StandardOutputContent"])
                            print(output["StandardErrorContent"])
                    else:
                        output = ssm.execute_command(
                            instance["InstanceId"],
                            [
                                f"mount {device_name} /mnt",
                                f"mkfs.ext4 {device_name}",
                                f"mount {device_name} /mnt"
                                "cd /mnt",
                                "git clone https://github.com/darya-sea/CNN_monitoring.git",
                                "cd /mnt/CNN_monitoring",
                                "pip3 install virtualenv",
                                "yum install python3-opencv",
                                "sh /mnt/CNN_monitoring/application/install.sh",
                                f"aws s3 sync s3://{config.S3_BUCKET} DATA",
                                "sh /mnt/CNN_monitoring/application/train.sh"
                            ]
                        )
                        with open(".commandid", "w") as _file:
                            _file.write(output["CommandId"])
    
                        print(output["StandardOutputContent"])
                        print(output["StandardErrorContent"])
                break
            else:
                print("[INFO] Waiting for request to be fulfilled.")
        time.sleep(3)

    # ec2.cancel_spot_fleet_request()
    # ec2.delete_volume()

def help(script_name):
    print(
    f"""
        usage: {script_name} <prepare|train|predict|sync>

        preapre example: 
          python {script_name} prepare
        train example: 
          python {script_name} train
        predict example: 
          python {script_name} predict "../CNN/heřmánkovec nevonný/2022_09_21 hermankovec/00257C.tif"
        sync example: 
          python {script_name} sync
          python {script_name} sync CNN
        prepare_spot example: 
          python {script_name} prepare_spot
        request_spot example: 
          python {script_name} request_spot
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
            case "prepare_spot":
                prepare_spot()
            case "request_spot":
                request_spot()
                
    else:
        help(sys.argv[0])

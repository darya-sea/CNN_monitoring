import boto3
import json
import time


class EC2:
    def __init__(self):
        self.__launch_template = "CNNTrainInstanceTemplate"
        self.__instance_profile = "CNNTrainInstanceProfile"
        self.__instance_role = "CNNTrainInstanceRole"
        self.__spot_fleet_role = "CNNTrainSpotFleetRole"
        self.__volume_name = "CNNTrainInstanceVolume"
        self.__volume_size = 30
        self.__volume_device = "/dev/xvdf"
        self.__ami_id = "ami-0e8ac16acd5e85cc4"

        self.__instance_role_policies = [
            "AmazonEC2FullAccess",
            "AmazonS3FullAccess",
            "AmazonSSMManagedInstanceCore",
            "CloudWatchLogsFullAccess"
        ]

        self.__session = boto3._get_default_session()

    def create_spot_fleet_role(self):
        client = self.__session.client("iam")
        try:
            client.create_role(
                RoleName=self.__spot_fleet_role,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Sid": "",
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "spotfleet.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole"
                            }
                        ]
                    }
                )
            )
            print(f"[INFO] Role {self.__spot_fleet_role} created.")
        except client.exceptions.EntityAlreadyExistsException:
            print(f"[INFO] Role {self.__spot_fleet_role} already exist.")

        client.attach_role_policy(
            RoleName=self.__spot_fleet_role,
            PolicyArn=f"arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
        )

    def create_instance_profile(self):
        client = self.__session.client("iam")

        try:
            client.create_role(
                RoleName=self.__instance_role,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "ec2.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole"
                            }
                        ]
                    }
                )
            )
            print(f"[INFO] Role {self.__instance_role} created.")
        except client.exceptions.EntityAlreadyExistsException:
            print(f"[INFO] Role {self.__instance_role} already exist.")

        for policy in self.__instance_role_policies:
            client.attach_role_policy(
                RoleName=self.__instance_role,
                PolicyArn=f"arn:aws:iam::aws:policy/{policy}"
            )
        try:
            client.create_instance_profile(
                InstanceProfileName=self.__instance_profile
            )
            print(f"[INFO] InstanceProfile {self.__instance_profile} created.")
        except client.exceptions.EntityAlreadyExistsException:
            print(f"[INFO] InstanceProfile {self.__instance_profile} already exist.")

        try:
            client.add_role_to_instance_profile(
                InstanceProfileName=self.__instance_profile,
                RoleName=self.__instance_role
            )
            print(f"[INFO] Role {self.__instance_role} added to InstanceProfile {self.__instance_profile}.")
            return self.__instance_profile
        except client.exceptions.LimitExceededException:
            print(f"[INFO] Role {self.__instance_role} already added to InstanceProfile {self.__instance_profile}.")
        return self.__instance_profile
    
    def delete_instance_profile(self):
        client = self.__session.client("iam")

        try:
            client.remove_role_from_instance_profile(
                InstanceProfileName=self.__instance_profile,
                RoleName=self.__instance_role
            )
        except client.exceptions.NoSuchEntityException:
            pass

        try:
            client.delete_instance_profile(InstanceProfileName=self.__instance_profile)
        except client.exceptions.NoSuchEntityException:
            pass

        for policy in self.__instance_role_policies:
            try:
                client.detach_role_policy(
                    RoleName=self.__instance_role,
                    PolicyArn=f"arn:aws:iam::aws:policy/{policy}"
                )
            except client.exceptions.NoSuchEntityException:
                pass

        try:
            client.delete_role(RoleName=self.__instance_role)
        except client.exceptions.NoSuchEntityException:
            pass

        print(f"[INFO] Role {self.__instance_role} removed from InstanceProfile {self.__instance_profile}.")
        print(f"[INFO] InstanceProfile {self.__instance_profile} deleted.")
        print(f"[INFO] Role {self.__instance_role} deleted.")

    def create_launch_template(self):
        client = self.__session.client("ec2")
        try:
            client.create_launch_template(
                LaunchTemplateName=self.__launch_template,
                LaunchTemplateData={
                    "ImageId": self.__ami_id,
                    "SecurityGroups": ["default"],
                    "Placement": {
                        "AvailabilityZone": f"{self.__session.region_name}a"
                    },
                    "IamInstanceProfile": {
                        "Name": self.__instance_profile,
                    }
                }
            )
            print(f"[INFO] LaunchTemplate {self.__launch_template} created.")
        except client.exceptions.ClientError as err:
            if "InvalidLaunchTemplateName.AlreadyExistsException" in err.args[0]:
                print(f"[INFO] LaunchTemplate {self.__launch_template} already exists.")

        return self.__launch_template
    
    def delete_launch_templat(self):
        client = self.__session.client("ec2")

        try:
            client.delete_launch_template(LaunchTemplateName=self.__launch_template)
        except client.exceptions.ClientError:
            pass

        print(f"[INFO] LaunchTemplate {self.__launch_template} deleted.")

    def create_volume(self):
        client = self.__session.client("ec2")
        response = client.describe_volumes(
            Filters=[
                {
                    "Name": "tag:Name",
                    "Values": [
                        self.__volume_name
                    ]
                },
            ]
        )

        if response["Volumes"]:
            print(f"[INFO] Volume {self.__volume_name } already exists.")
        else:
            response = client.create_volume(
                AvailabilityZone=f"{self.__session.region_name}a",
                Size=self.__volume_size,
                VolumeType="standard",
                TagSpecifications=[
                    {
                        "ResourceType": "volume",
                        "Tags": [
                            {
                                "Key": "Name",
                                "Value": self.__volume_name
                            }
                        ]
                    }
                ],
            )
            print(f"[INFO] Volume {self.__volume_name} created.")
        return self.__volume_name
    
    def attach_volume(self, instance_id):
        client = self.__session.client("ec2")
        response = client.describe_volumes(
            Filters=[
                {
                    "Name": "tag:Name",
                    "Values": [
                        self.__volume_name
                    ]
                }
            ]
        )

        for volume in response["Volumes"]:
            try:
                response = client.attach_volume(
                    Device=self.__volume_device,
                    InstanceId=instance_id,
                    VolumeId=volume["VolumeId"],
                )
                time.sleep(5)
            except client.exceptions.ClientError as err:
                if "is already attached to an instance" in err.args[0]:
                    print(
                        f"[INFO] Volume {volume['VolumeId']} attached to {instance_id}.",
                        f"Device {self.__volume_device}."
                    )
                else:
                    print(f"[ERROR] {err.args[0]}")

        return instance_id, volume["VolumeId"], self.__volume_device

    def delete_volume(self):
        client = self.__session.client("ec2")
        response = client.describe_volumes(
            Filters=[
                {
                    "Name": "tag:Name",
                    "Values": [
                        self.__volume_name
                    ]
                },
            ]
        )
        for volume in response.get("Volumes", []):
            try:
                client.delete_volume(VolumeId=volume["VolumeId"])
                print(f"[INFO] Volume {self.__volume_name} deleted.")
            except client.exceptions.ClientError as err:
                print(f"[ERROR] {err.args[0]}")
    
    def get_account_id(self):
        client = self.__session.client("sts")
        return client.get_caller_identity()["Account"]

    def get_active_spot_fleet_request(self):
        iam_fleet_role = f"arn:aws:iam::{self.get_account_id()}:role/{self.__spot_fleet_role}"
        client = self.__session.client("ec2")
        
        for spot_request in client.describe_spot_fleet_requests()["SpotFleetRequestConfigs"]:
            if spot_request["SpotFleetRequestState"] == "active":
                if spot_request["SpotFleetRequestConfig"]["IamFleetRole"] == iam_fleet_role:
                    return spot_request

    def get_spot_fleet_instances(self, spot_flee_request_id):
        client = self.__session.client("ec2")
        return client.describe_spot_fleet_instances(
            SpotFleetRequestId=spot_flee_request_id
        )["ActiveInstances"]


    def cancel_spot_fleet_request(self):
        client = self.__session.client("ec2")

        if (spot_request := self.get_active_spot_fleet_request()):
            fleet_instances = self.get_spot_fleet_instances(spot_request['SpotFleetRequestId'])

            print(f"[INFO] Canceling request {spot_request['SpotFleetRequestId']}.")
            client.cancel_spot_fleet_requests(
                SpotFleetRequestIds=[spot_request['SpotFleetRequestId']],
                TerminateInstances=True
            )

            print(f"[INFO] Terminating instances {[instance['InstanceId'] for instance in fleet_instances]}.")

            for instance in fleet_instances:
                instance_statuses = client.describe_instance_status(InstanceIds=[instance["InstanceId"]])
                if instance_statuses["InstanceStatuses"]:
                    for status in instance_statuses["InstanceStatuses"]:
                        if status["InstanceState"]["Name"] == "terminated":
                            return spot_request
                else:
                    return spot_request
        else:
            print(f"[INFO] Not found active spot fleet requests.")
        return spot_request

    def request_spot_fleet(self, instance_types, max_price):
        client = self.__session.client("ec2")
        
        if (spot_request := self.get_active_spot_fleet_request()):
            print(
                f"[INFO] Found active spot fleet request {spot_request['SpotFleetRequestId']}.", 
                f"ActivityStatus is {spot_request['ActivityStatus']}."
            )
            return spot_request

        response = client.request_spot_fleet(
            SpotFleetRequestConfig={
                "IamFleetRole": f"arn:aws:iam::{self.get_account_id()}:role/{self.__spot_fleet_role}",
                "AllocationStrategy": "priceCapacityOptimized",
                "TargetCapacity": 1,
                "SpotPrice": str(max_price),
                "TerminateInstancesWithExpiration": True,
                "LaunchSpecifications": [],
                "Type": "maintain",
                "OnDemandTargetCapacity": 1,
                "LaunchTemplateConfigs": [
                    {
                        "LaunchTemplateSpecification": {
                            "LaunchTemplateName": self.__launch_template,
                            "Version": "$Latest"
                        },
                        "Overrides": instance_types
                    }
                ]
            }
        )
        print(f"[INFO] Created request with id {response['SpotFleetRequestId']}.")
        return response

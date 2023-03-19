import os
import boto3
import botocore

class SSM:
    def __init__(self):
        self.__session = boto3._get_default_session()
    
    def execute_command(self, instance_id, commands):
        client = self.__session.client("ssm")
        response = client.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": commands}
        )
        command_id = response["Command"]["CommandId"]

        waiter = client.get_waiter('command_executed')
        try:
            waiter.wait(CommandId=command_id, InstanceId=instance_id)
        except botocore.exceptions.WaiterError:
            pass

        output = client.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        return output

    def get_command_id(self):
        client = self.__session.client("ssm")

        if os.path.exists(".commandid"):
            with open(".commandid", "r") as _file:
                instance_id, command_id = _file.read().split(":")
        
                command_invocations = client.list_command_invocations(CommandId=command_id, InstanceId=instance_id)
                
                if command_invocations["CommandInvocations"]:
                    return command_id, instance_id

    def get_command_invocation(self, command_id, instance_id):
        client = self.__session.client("ssm")
        waiter = client.get_waiter('command_executed')

        command_invocations = client.list_command_invocations(CommandId=command_id, InstanceId=instance_id)

        if command_invocations["CommandInvocations"]:
            while True:
                try:
                    waiter.wait(CommandId=command_id, InstanceId=instance_id)
                except botocore.exceptions.WaiterError:
                    pass

                output = client.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
                if output["Status"] in ("Failed", "Success"):
                    break
                else:
                    yield output
        return output
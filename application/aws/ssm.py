import os
import boto3

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
        output = client.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        return output["StandardOutputContent"]


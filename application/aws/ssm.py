import time
import boto3
import botocore


class SSM:
    """Manage AWS SSM."""

    def __init__(self):  # noqa
        self.__session = boto3._get_default_session()
        self.__log_group_name = "CNNTrainLogs"
        self.__log_stream_name = "CNNTrainLogsStream"

    def execute_command(self, instance_id: str, commands: list) -> dict:
        """Execute SSM command on target instance.

        Args:
            instance_id (str): target instance id.
            commands (list): commands list to execute.

        Returns:
            dict: results.
        """
        client = self.__session.client("ssm")
        response = client.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": commands}
        )
        command_id = response["Command"]["CommandId"]

        waiter = client.get_waiter("command_executed")
        try:
            waiter.wait(CommandId=command_id, InstanceId=instance_id)
        except botocore.exceptions.WaiterError as err:
            pass

        output = client.get_command_invocation(
            CommandId=command_id, InstanceId=instance_id)
        return output

    def get_command_invocation(self, command_id: str, instance_id: str) -> dict:
        """Get exucted command results.

        Args:
            command_id (str): id of executed command task.
            instance_id (str): target instance id.

        Returns:
            dict: results.
        """
        client = self.__session.client("ssm")
        command_invocations = client.list_command_invocations(
            CommandId=command_id, InstanceId=instance_id)

        if command_invocations["CommandInvocations"]:
            while True:
                output = client.get_command_invocation(
                    CommandId=command_id, InstanceId=instance_id)
                if output["Status"] in ("Failed", "Success"):
                    break
                else:
                    yield output
        return output

    def list_commands(self) -> list:
        """List of all executed ssm commands.

        Returns:
            list: results.
        """
        client = self.__session.client("ssm")
        return client.list_commands()["Commands"]

    def send_log(self, log_message: str):
        """Send log to CloudWatch stream.

        Args:
            log_message (str): message to send.
        """
        client = self.__session.client("logs")

        try:
            response = client.describe_log_streams(
                logGroupName=self.__log_group_name,
                logStreamNamePrefix=self.__log_stream_name
            )
        except client.exceptions.ResourceNotFoundException:
            client.create_log_group(logGroupName=self.__log_group_name)
            response = client.create_log_stream(
                logGroupName=self.__log_group_name,
                logStreamName=self.__log_stream_name
            )

        log_event = {
            "logGroupName": self.__log_group_name,
            "logStreamName": self.__log_stream_name,
            "logEvents": [
                {
                    "timestamp": int(round(time.time() * 1000)),
                    "message": log_message.decode()
                }
            ]
        }
        if (sequence_token := response["logStreams"][0].get("uploadSequenceToken")):
            log_event.update({"sequenceToken": sequence_token})

        response = client.put_log_events(**log_event)

    def get_logs(self):
        """Get CloudWatch logs.

        Returns:
            list: results.
        """
        client = self.__session.client("logs")

        try:
            response = client.get_log_events(
                logGroupName=self.__log_group_name,
                logStreamName=self.__log_stream_name,
                limit=100
            )
            return response["events"]
        except client.exceptions.ResourceNotFoundException:
            return []

    def delete_logs(self):
        """Delete CloudWatch logs."""
        client = self.__session.client("logs")

        try:
            client.delete_log_group(logGroupName=self.__log_group_name)
        except client.exceptions.ResourceNotFoundException:
            pass

        print(f"[INFO] Log group {self.__log_group_name} deleted.")

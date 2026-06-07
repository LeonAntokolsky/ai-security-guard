import os
import json
import boto3
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Global AWS Configurations
BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')


@tool
def fetch_latest_security_logs() -> str:
    """
    Useful when you need to read the latest visual logs from the S3 security bucket.
    This tool retrieves the JSON timeline containing frame IDs, timestamps, and detected objects.
    """
    try:
        s3 = boto3.client('s3', region_name=AWS_REGION)

        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix='activity_logs/')
        if 'Contents' not in response:
            return "Error: No security log files found in the target S3 bucket."

        files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        latest_file_key = files[0]['Key']

        s3_object = s3.get_object(Bucket=BUCKET_NAME, Key=latest_file_key)
        log_content = s3_object['Body'].read().decode('utf-8')

        return log_content

    except Exception as e:
        return f"Failed to retrieve data from AWS infrastructure: {str(e)}"


class SecurityAIAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

        self.tools = [fetch_latest_security_logs]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an elite AI Security Guard & Concierge system monitoring an automated camera feed.\n"
                "Your objective is to inspect structured event timelines retrieved via your tools to answer user inquiries.\n"
                "Always specify precise timestamps (in seconds) or frame details when explaining an incident.\n"
                "If an object or person was not logged in the metrics data, clearly state that it was not observed."
            )),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent_runtime = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=agent_runtime, tools=self.tools, verbose=True)

    def ask(self, user_query: str) -> str:
        result = self.executor.invoke({"input": user_query})
        return result["output"]
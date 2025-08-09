from agents import(
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig
)
from dotenv import load_dotenv
import os
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()

api_key=os.getenv("GOOGLE_API_KEY")
base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
model="gemini-2.5-flash"

external_client=AsyncOpenAI(
    api_key=api_key,
    base_url=base_url
)

model=OpenAIChatCompletionsModel(
    model=model,
    openai_client=external_client
)

config=RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent=Agent(
    name="Career Planner Agent",
    instructions="""
      When a user provides a list of their skills, you must:
    1.  Analyze the provided skills to understand the user's current capabilities and potential.
    2.  Suggest relevant career paths and industries where these skills are valuable.
    3.  Identify specific additional skills the user should acquire to enhance their profile or transition into desired roles.
    4.  Recommend potential job roles that align with their current and suggested skill sets.
    5.  Offer actionable advice for professional development, continuous learning, and strategic networking to help them achieve their future goals.
    6. must jugde current stage for their user prompt
    give user simple short answer in 80 words in easy english or roman urdu.
    """
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hi How I can help you?").send()

@cl.on_message
async def on_message(message: cl.Message):
    history=cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    msg=cl.Message(content="")
    await msg.send()

    result=Runner.run_streamed(
        agent,
        input=history,
        run_config=config
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
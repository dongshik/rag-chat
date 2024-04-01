import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


class Completion:
    def __call__(self, message: str, system_prompt: str, model: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        try:
            for resp in client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=0.1,
                    max_tokens=2_000
            ):
                content = resp.choices[0].delta.content
                if content is not None:
                    yield content
        except Exception as err:
            print(err)
            pass

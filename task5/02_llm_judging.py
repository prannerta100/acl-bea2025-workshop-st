import os
import requests
import json
import warnings
import httpx
import asyncio
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HAWKEYE_CLIENT_BASE_URL = os.environ["HAWKEYE_CLIENT_BASE_URL"]
path = "/chat/completions"

df = pd.read_csv("./data/mrbench_v3_devset_train_data.csv").dropna(subset=["response"]).sample(frac=1, random_state=42)

example_json = {
    "conversation_id": "258053434",
    "conversation_history": "Tutor: You earn one point for your good beginning. \n Tutor: That was a good try. \n Tutor: What is the value of 3^3? \n Student: 9",
    "tutor_responses": {
        "Sonnet": {
            "response": "Let's double-check that: 3^3 means 3 multiplied by itself three times, so it's 3 x 3 x 3, which equals 27.",
            "annotation": {
                "Actionability": "No",
                "Mistake_Identification": "Yes",
                "Mistake_Location": "Yes",
                "Providing_Guidance": "Yes"
            }
        },
        "Novice": {
            "response": "That was a good try. An exponent is a number or letter written above and to the right of a mathematical expression called the base. It indicates that the base is to be raised to a certain power. 3 is the base and 3 is the exponent or power. So, you have to multiply 3 for 3 times. Good work so far.",
            "annotation": {
                "Actionability": "Yes",
                "Mistake_Identification": "Yes",
                "Mistake_Location": "Yes",
                "Providing_Guidance": "Yes"
            }
        },
        "Llama31405B": {
            "response": "Here is the revised response:\\n\\n\"Almost, but 3^3 actually means 3 multiplied by itself 3 times, so it would be 3 x 3 x 3 = 27, not 9.\"",
            "annotation": {
                "Actionability": "No",
                "Mistake_Identification": "Yes",
                "Mistake_Location": "Yes",
                "Providing_Guidance": "Yes"
            }
        },
        "GPT4": {
            "response": "That's a good effort, but actually, 3^3 means 3 multiplied by itself three times, which equals 27.",
            "annotation": {
                "Actionability": "No",
                "Mistake_Identification": "Yes",
                "Mistake_Location": "Yes",
                "Providing_Guidance": "Yes"
            }
        },
        "Gemini": {
            "response": "Remember, 3 to the power of 3 means 3 multiplied by itself three times: 3 x 3 x 3.",
            "annotation": {
                "Actionability": "Yes",
                "Mistake_Identification": "Yes",
                "Mistake_Location": "Yes",
                "Providing_Guidance": "Yes"
            }
        },
        "Mistral": {
            "response": "\"Remember, 3^3 means 3 multiplied by itself 3 times, which is 3 * 3 * 3.\"",
            "annotation": {
                "Actionability": "Yes",
                "Mistake_Identification": "Yes",
                "Mistake_Location": "Yes",
                "Providing_Guidance": "Yes"
            }
        },
        "Expert": {
            "response": "Great try! You solved the problem 3x3. Can you recall what an exponent does to a number?",
            "annotation": {
                "Actionability": "Yes",
                "Mistake_Identification": "Yes",
                "Mistake_Location": "Yes",
                "Providing_Guidance": "Yes"
            }
        },
        "Llama318B": {
            "response": "You're on the right track, but let's see if we can simplify it further, 3^3 actually equals 3 multiplied by itself 3 times, which gives us 3*3*3 = 27.",
            "annotation": {
                "Actionability": "No",
                "Mistake_Identification": "Yes",
                "Mistake_Location": "Yes",
                "Providing_Guidance": "Yes"
            }
        },
        "Phi3": {
            "response": "Actually, the value of 3^3 is 27, not 9. Great effort though!",
            "annotation": {
                "Actionability": "No",
                "Mistake_Identification": "Yes",
                "Mistake_Location": "Yes",
                "Providing_Guidance": "No"
            }
        }
    }
}


async def make_open_ai_request(path, data):
    auth_server_url = HAWKEYE_CLIENT_BASE_URL  + "/oauthprovider/oauth2/token"
    token_req_payload = {'grant_type': 'client_credentials'}
    token_response = requests.post(auth_server_url,
                                    data=token_req_payload,
                                    verify=False,
                                    allow_redirects=False,
                                    auth=(os.environ["HAWKEYE_CLIENT_ID"], os.environ["HAWKEYE_CLIENT_SECRET"]))
    tokens = json.loads(token_response.text)
    access_token = tokens["access_token"]
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    url = HAWKEYE_CLIENT_BASE_URL + path
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None),
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
        ) as client:
            try:
                response = await client.post(url, json=data, headers=headers)
                # Check if the response status code is not successful (e.g., not in the 200 range)
                if not response.is_success:
                    # Log or handle the failure
                    # Raise an exception to propagate the failure
                    response.raise_for_status()
                # If the response is successful, return the response data
                return response
            except httpx.HTTPStatusError as exc:
                # Handle HTTP status errors (e.g., 4xx or 5xx responses)
                raise exc
            except httpx.RequestError as exc:
                # Handle request-related errors (e.g., connection errors)
                raise exc
    except Exception as exc:
        raise exc

        


predictions = []
for i, row in df.iterrows():
    request_data = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "system",
            "content": "You are an expert who detects which LLM or human type has generated a certain response to a math tutoring related conversation. You are supposed to answer only in a json format with the following format `{\"reasoning\": \"<your reasoning\", \"answer\": \"<one of the 9 options>\"}`. The 9 options allowed in the \"answer\" field are: \"Sonnet\", \"Llama318B\", \"Llama31405B\", \"GPT4\", \"Mistral\", \"Expert\", \"Gemini\", \"Phi3\", \"Novice\". All these options are roughly equally likely to occur and you should not be biased towards a certain answer, so please focus on the nuances inside a certain model's response or the response of an expert or a novice. An example conversation is given to you here, and you should carefully investigate the quirks of each of the 7 LLMs and the novice and the expert:" + "\n\n\n" + json.dumps(example_json)
        },
        {"role": "user", "content": "Conversation:\n" + row["conversation_history"] + "\n\nResponse:\n" + row["response"]},
    ],
    # "temperature": 0.000001,
    }
    llm_response = asyncio.run(make_open_ai_request(path, request_data)).json()["choices"][0]["message"]["content"]
    try:
        predictions.append(json.loads(llm_response)["answer"])
    except:
        predictions.append(None)

        
print("True Labels: ", df["label"])
print("Predicted Labels: ", predictions)
print("Accuracy: ", np.mean(np.array(predictions) == df["label"]))
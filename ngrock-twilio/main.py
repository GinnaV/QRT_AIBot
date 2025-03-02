import datetime
import os
import json
import base64
import asyncio
import requests
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = "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"
PORT = int(os.getenv('PORT', 5050))
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate."
    " You can also fetch real-time cryptocurrency prices and past trends. "
    "If a user asks about Bitcoin, Ethereum, or any major crypto, provide the latest price and a summary of past trends."
)
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]
SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

CRYPTO_IDS = [
    "bitcoin", "ethereum", "dogecoin", "ripple", "cardano", "solana", "polkadot",
    "litecoin", "chainlink", "stellar", "monero", "tron", "avalanche-2", "uniswap", "algorand"
]
crypto_names_map = {name: name for name in CRYPTO_IDS}
crypto_prices = {}


def get_crypto_prices():
    """Fetches real-time prices for predefined crypto IDs."""
    ids_string = ",".join(CRYPTO_IDS)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_string}&vs_currencies=usd"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error fetching prices:", e)
        return {}

crypto_prices = get_crypto_prices()


def get_crypto_history(crypto_id, days=3):
    """Fetches historical prices for a given cryptocurrency."""
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            datetime.datetime.fromtimestamp(item[0] / 1000, tz=datetime.timezone.utc).strftime('%Y-%m-%d'): item[1]
            for item in data["prices"]
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical data for {crypto_id}: {e}")
        return {}


@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    response.say("Please wait while we connect your call to the A. I. voice assistant, powered by Twilio and the Open-A.I. Realtime API")
    response.pause(length=1)
    response.say("O.K. you can start talking!")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from OpenAI and send audio back to Twilio."""
    try:
        async for openai_message in openai_ws:
            response = json.loads(openai_message)
            if response['type'] in LOG_EVENT_TYPES:
                print(f"Received event: {response['type']}", response)

            # Inject Crypto Data if Mentioned
            user_text = response.get('content', '')  # Extract user text if available
            crypto_name = extract_crypto_name(user_text)

            if crypto_name:
                crypto_price = crypto_prices.get(crypto_name, {}).get("usd", "Unknown")
                history_data = get_crypto_history(crypto_name, days=3)

                crypto_response = f"The current price of {crypto_name.capitalize()} is ${crypto_price} USD."
                if history_data:
                    crypto_response += " Here is the last 3 days of price history:\n"
                    for date, price in history_data.items():
                        crypto_response += f"- {date}: ${price:.2f}\n"

                # Send crypto data as an assistant response
                crypto_event = {
                    "type": "response.audio.delta",
                    "delta": base64.b64encode(crypto_response.encode()).decode("utf-8"),
                }
                await websocket.send_json(crypto_event)

    except Exception as e:
        print(f"Error in send_to_twilio: {e}")

        def extract_crypto_name(query):
            """Extracts cryptocurrency name from the user query."""
            query = query.lower()
            for name in crypto_names_map.keys():
                    if name in query:
                        return name
        return None



        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can imagine. How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    # await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

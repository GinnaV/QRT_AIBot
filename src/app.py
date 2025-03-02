from flask import Flask, request
import flask
import json
from flask_cors import CORS, cross_origin
import prompt_3 as p3

app = Flask(__name__)
CORS(app)

@app.route("/hello")
@cross_origin(supports_credentials=True)
def hello():
    return "Hello, World!"

@app.route('/users', methods=["GET", "POST"])
@cross_origin(supports_credentials=True)
def users():
    if request.method == "POST":
        received_data = request.get_json()
        print(f"received data: {received_data}")
        newMessage = p3.ask_crypto_ai(received_data.get("data"))
        print(newMessage)
        return_data = {
            "status": "success",
            "message": f"{newMessage}"
        }
        return flask.Response(response=json.dumps(return_data), status=201)
    # print("users endpoint reached...")
    # if request.method == "GET":
    #     with open("users.json", "r") as f:
    #         data = json.load(f)
    #         data.append({
    #             "username": "user4",
    #             "pets": ["hamster"]
    #         })

    #         return flask.jsonify(data)
    # if request.method == "POST":
    #     received_data = request.get_json()
    #     print(f"received data: {received_data}")
    #     message = received_data['data']
    #     return_data = {
    #         "status": "success",
    #         "message": f"received: {message}"
    #     }
    #     return flask.Response(response=json.dumps(return_data), status=201)

if __name__ == "__main__":
    app.run("localhost", 6969)
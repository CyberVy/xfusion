from flask import Flask,request
from multiprocessing import Process

def run_server(address="127.0.0.1",port=5000,):

    app = Flask(__name__)

    @app.route("/")
    def hello():
        return "Hello, Flask!"

    @app.route("/textToImage")
    def text_to_image():
        prompt = request.args.get("prompt")
        return prompt

    def run_flask():
        app.run(address,port,use_reloader=False)
    flask_process = Process(target=run_flask)
    flask_process.start()

    print(f"Flask is running on {address}:{port}.")

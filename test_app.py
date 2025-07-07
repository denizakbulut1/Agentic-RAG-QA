from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    print("--- Request received ---")
    return "<h1>Hello, Flask is working!</h1>"

if __name__ == '__main__':
    print("--- Starting Flask Server ---")
    app.run(debug=True, port=5001) # Using a different port just in case
from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("Starting OCEAN Personality Analyzer Frontend...")
    print("Frontend URL: http://127.0.0.1:5000")
    print("Make sure your API server is running on http://127.0.0.1:8000")
    
    app.run(host='127.0.0.1', port=5000, debug=True)

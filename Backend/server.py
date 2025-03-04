from flask import Flask
from flask_cors import CORS
from routes import api_routes  # Import the Blueprint from routes.py

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Register routes from routes.py
app.register_blueprint(api_routes, url_prefix='/api')  # Now all routes will be under /api

if __name__ == '__main__':
    app.run(debug=True)
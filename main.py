from flask import Flask
from api.auth import auth_bp
from api.predict import predict_bp

app = Flask(__name__)


app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(predict_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
# StrokePrediction

## Project Description
StrokePrediction is a backend service designed to predict the likelihood of a stroke based on user-provided health data. It leverages machine learning models and provides a RESTful API for integration with other applications.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StrokeBackend.git
   cd StrokeBackend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

2. Access Docs:  `http://127.0.0.1:8000/docs` 

3. Access the API at `http://127.0.0.1:8000/`.

4. Use tools like Postman or cURL to send requests to the API.

## Contributing
Feel free to fork the repository and submit pull requests for improvements or bug fixes.

## License
This project is licensed under the MIT License.
# Machine Learning Model Deployment using FastAPI, Scikit-Learn, and Pickle

## Overview

This project demonstrates how to train a machine learning model using **Scikit-Learn**, serialize it using **Pickle**, and deploy it as a **REST API using FastAPI**.

The API accepts input data and returns predictions in real time.

---

## Tech Stack

* Python
* Scikit-Learn
* FastAPI
* Pickle
* Uvicorn

---

## Project Structure

```
ml-fastapi-api/
│
├── train_model.py        # Train and save the ML model
├── model.pkl             # Serialized trained model
├── main.py               # FastAPI application
├── requirements.txt      # Dependencies
└── README.md
```

---

## Step 1: Train the Model

Run the training script to generate the model file.

```
python train_model.py
```

This creates:

```
model.pkl
```

---

## Step 2: Run the FastAPI Server

Start the API using:

```
uvicorn main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

---

## API Documentation

FastAPI automatically generates interactive documentation.

Open in browser:

```
http://127.0.0.1:8000/docs
```

You can test the API directly from the browser.

---

## Example API Request

Endpoint:

```
POST /predict
```

Example Input

```
[5.1, 3.5, 1.4, 0.2]
```

Example Response

```
{
  "prediction": [0]
}
```

---

## Install Dependencies

```
pip install -r requirements.txt
```

Example `requirements.txt`

```
fastapi
uvicorn
scikit-learn
numpy
pandas
```

---

## Use Cases

* Deploy machine learning models as APIs
* Real-time prediction systems
* Backend integration for ML services

---

## Author

Machine Learning API project built with **FastAPI and Scikit-Learn**.

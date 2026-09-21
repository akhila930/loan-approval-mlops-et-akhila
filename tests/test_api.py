"""
Tests for the FastAPI loan approval prediction service.
"""

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Loan Approval Prediction API"
    assert data["status"] == "running"


def test_health_endpoint():
    """Test the health check endpoint."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_model_info_endpoint():
    """Test the model information endpoint."""

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "model" in data
    assert "model_file" in data
    assert "preprocessor_file" in data
    assert "features" in data

    assert len(data["features"]) == 12


def test_prediction_endpoint():
    """Test a valid loan prediction request."""

    payload = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": 1,
        "Education": "Graduate",
        "Employment_Status": "Employed",
        "Applicant_Income": 50000,
        "Coapplicant_Income": 20000,
        "Loan_Amount": 200,
        "Loan_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
        "Age": 30,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "loan_status" in data
    assert "approval_probability" in data
    assert "rejection_probability" in data
    assert "model" in data

    assert data["prediction"] in [0, 1]

    assert data["loan_status"] in [
        "Approved",
        "Rejected",
    ]

    assert 0.0 <= data["approval_probability"] <= 1.0

    assert 0.0 <= data["rejection_probability"] <= 1.0

    assert (
        abs(
            data["approval_probability"]
            + data["rejection_probability"]
            - 1.0
        )
        < 0.0001
    )


def test_invalid_credit_history():
    """Test validation of an invalid Credit_History value."""

    payload = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": 1,
        "Education": "Graduate",
        "Employment_Status": "Employed",
        "Applicant_Income": 50000,
        "Coapplicant_Income": 20000,
        "Loan_Amount": 200,
        "Loan_Term": 360,
        "Credit_History": 2,
        "Property_Area": "Urban",
        "Age": 30,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_negative_income():
    """Test validation of negative applicant income."""

    payload = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": 1,
        "Education": "Graduate",
        "Employment_Status": "Employed",
        "Applicant_Income": -50000,
        "Coapplicant_Income": 20000,
        "Loan_Amount": 200,
        "Loan_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
        "Age": 30,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_metrics_endpoint():
    """Test the Prometheus metrics endpoint."""

    response = client.get("/metrics")

    assert response.status_code == 200

    assert "http" in response.text.lower()
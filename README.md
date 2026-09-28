# Loan Approval MLOps - ET - Akhila

MLOps End Term Project: Loan Approval Prediction System.

This repository follows the architecture and repository structure specified in the MLOps End Term Project statement:
Data Source -> DVC -> Data Validation -> Feature Engineering -> Model Training -> MLflow -> Pytest -> Docker -> GitHub Actions -> Kubernetes -> FastAPI -> Prometheus/Grafana.

Project status: Initial project skeleton.

A synthetic loan approval dataset was used for academic and MLOps demonstration purposes.

My project is a Loan Approval Prediction System built using an MLOps approach. It takes applicant information, validates and transforms the data, trains and tracks machine-learning models, exposes the selected model through a FastAPI REST API, containerizes it using Docker, deploys it using Kubernetes, and monitors the API using Prometheus and Grafana. DVC is used for data and pipeline reproducibility, Pytest for testing, MLflow for experiment tracking, and GitHub Actions for CI.


# Architecture

             Dataset
                │
                ▼
              DVC
                │
                ▼
        Data Validation
                │
                ▼
       Feature Engineering
                │
                ▼
         Model Training
                │
                ▼
             MLflow
                │
                ▼
             Pytest
                │
                ▼
             Docker
                │
                ▼
        GitHub Actions CI/CD
                │
                ▼
           Kubernetes
                │
                ▼
            FastAPI
                │
                ▼
          Prometheus
                │
                ▼
             Grafana

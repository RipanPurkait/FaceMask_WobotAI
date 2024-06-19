# Use the official Python image from the Docker Hub
FROM python:3.10
ADD main.py .
RUN pip install ultralytics argparse
CMD ["python", "main.py"]

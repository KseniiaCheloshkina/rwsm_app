FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt --default-timeout=100
CMD ["python3.7", "app.py"]


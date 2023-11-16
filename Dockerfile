FROM python:3.9.18-slim
WORKDIR /app
COPY main.py /app
COPY constants.py /app
ADD models /app/models
COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --compile -r requirements.txt
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000" ]

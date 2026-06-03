FROM prefecthq/prefect:3-latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flow.py .
COPY prefect.yaml .

ENV PYTHONIOENCODING=utf-8
ENV PREFECT_API_URL=http://prefect-server:4200/api

CMD ["python", "flow.py"]
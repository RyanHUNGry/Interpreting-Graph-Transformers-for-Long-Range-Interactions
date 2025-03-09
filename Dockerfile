FROM python:3.11-slim

WORKDIR /usr/local/intepreting-graph-transformers-for-long-range-interactions

# Copy in the source code
COPY . .

RUN pip install --upgrade pip
RUN pip install torch
RUN pip install torch_geometric
RUN pip install scipy numpy
RUN pip install networkx
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install captum
RUN pip install torchmetrics

CMD ["sh", "-c", "python run.py"]

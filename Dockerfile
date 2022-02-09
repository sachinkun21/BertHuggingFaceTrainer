FROM huggingface/transformers-pytorch-cpu:4.9.1
copy ./ /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "fastAPI:app", "--host", "0.0.0.0", "--port 8000"]

# BertHuggingFaceTrainer

- To run Inference in your virtual environment:
  - Clone the repo: `git clone https://github.com/sachinkun21/BertHuggingFaceTrainer.git`
  - Install requirements: 'pip install -r requirements.txt'
  - pull the data and model file: dvc pull dvcfile.dvc
  - uvicorn fastAPI:app --host 0.0.0.0 --port 8000 --reload

- Via Docker:
  - docker build -t inference:pytorch .
  - docker run --rm -p 8000:8000 --name inference_container inference:pytorch (if you get environment error `Click will abort further execution because Python was configured to use ASCII as encoding` )
  - excecute this : ` docker run -p 8000:8000 --rm --name inference-torch inference:pytorch export LC_ALL=C.UTF-8  && export LANG=C.UTF-8`

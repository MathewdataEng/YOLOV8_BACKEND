# Quickstart

1. Clone the Label Studio Machine Learning Backend git repository:

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
```

2. Set up the environment:

```bash
cd label-studio-ml-backend/

pip install -U -e .

```

3. Clone this git repository:

```bash
git clone https://github.com/MathewdataEng/YOLOV8_BACKEND.git

```

4. Modify your label studio credential in model.py:
```python
LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'Your URL to label studio')
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN", 'Your access token')
```

4. Run without Docker:
```bash
label-studio-ml start YOLOV8_BACKEND -p 9091

```

FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/models
WORKDIR /app
COPY requirements-api.txt requirements.txt ./
# CPU-only torch keeps the image small; the model runs fine on CPU for the small backbone
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install transformers numpy scipy pillow scikit-learn \
 && pip install fastapi "uvicorn[standard]" python-multipart
COPY fsad ./fsad
# Download the backbone at build time so the container starts offline
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-with-registers-small')"
EXPOSE 8000
CMD ["uvicorn", "fsad.api:app", "--host", "0.0.0.0", "--port", "8000"]

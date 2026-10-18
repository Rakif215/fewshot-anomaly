FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR $HOME/app
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user transformers numpy scipy pillow scikit-learn fastapi "uvicorn[standard]" python-multipart
COPY --chown=user fsad ./fsad
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-with-registers-small')"
EXPOSE 8000
CMD ["uvicorn", "fsad.api:app", "--host", "0.0.0.0", "--port", "8000"]

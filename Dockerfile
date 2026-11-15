# DeepQ Learning Lab as one container: build the web UI, then run the Python lab
# server on CPU. Used by the Hugging Face Space (see docs/deploy_huggingface.md),
# and works with plain Docker too:
#   docker build -t deepq-lab . && docker run -p 7860:7860 deepq-lab

# ---- 1. Web UI -------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- 2. Lab server ---------------------------------------------------------
FROM python:3.12-slim

# A compiler, CMake and zlib build multi-agent-ale-py (two-player Pong) from source.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face runs Docker Spaces as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

# The CPU build of PyTorch is far smaller than the default CUDA one.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml README.md ./
COPY src ./src
# Editable install: the server finds trained_models/ and web/dist/ next to src/.
RUN pip install --no-cache-dir -e ".[env,lab]" websockets \
 && python -c "from pettingzoo.atari import pong_v3; import multi_agent_ale_py; print('two-player Pong OK')"

COPY trained_models ./trained_models
COPY --from=web /web/dist ./web/dist
RUN mkdir -p experiments && chown -R user:user /app
USER user

# Public mode: listen on every interface and accept the Space's own host name.
ENV DQN_LAB_PUBLIC=1 \
    DQN_LAB_HOST=0.0.0.0 \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2
EXPOSE 7860
CMD ["python", "-m", "dqn_project.pong_server"]

# Run DeepQ Learning Lab with Docker

Docker is the easiest way to run the lab: no Python, Node or emulator setup, the
same steps on Windows, macOS and Linux. The image builds the web UI, installs
PyTorch, the Atari emulator and two-player Pong, and ships the trained models.

## 1. Install Docker

- **Windows / macOS:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  and start it (on Windows it runs on WSL2).
- **Linux:** install Docker Engine and the Compose plugin
  (for example `sudo apt install docker.io docker-compose-v2` on Ubuntu).

## 2. Get the code

```bash
git clone https://github.com/bhagathchan1/DQN.git
cd DQN
```

## 3. Start the lab

```bash
docker compose up --build
```

The first build takes about 5–10 minutes (it compiles the two-player Pong
emulator); later starts take seconds. When the log shows
`Uvicorn running on http://0.0.0.0:7860`, open **<http://localhost:8765>**.

Stop it with `Ctrl+C`, or run `docker compose down` from another terminal.
Run it in the background with `docker compose up -d`.

## Options

| Want | Do |
|---|---|
| A different port | `LAB_PORT=9000 docker compose up` then open http://localhost:9000 |
| An NVIDIA GPU for Pong learning | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build` (needs the NVIDIA driver and Container Toolkit; Docker Desktop on Windows supports it through WSL2) |
| Start fresh (delete saved Pong sessions and Breakout scores) | `docker compose down -v` |
| Update to the latest code | `git pull` then `docker compose up --build` |
| Without Compose | `docker build -t deepq-lab .` then `docker run -p 8765:7860 deepq-lab` |

## What is saved

Pong sessions (each learned model version) and Breakout scores live in the
`lab-data` Docker volume, so they survive restarts and rebuilds. The trained
models are part of the image.

## Good to know

- **CPU is enough to play.** The network makes 15 decisions a second. On CPU,
  learning between Pong matches takes about a minute (measured: 49 s on a laptop);
  a GPU makes it a few seconds.
- **Resources:** the image needs about 3.5 GB of disk; the running container
  used about 550 MB of memory in testing, so 1–2 GB is plenty.
- **One game at a time.** Pong and Breakout share one emulator slot.
- **Sharing on a network or online:** the container runs the server in public
  mode, so it answers any host name but only accepts requests from its own pages.
  Put it behind HTTPS if you expose it to the internet; see
  [`deploy_huggingface.md`](deploy_huggingface.md) for hosted options.

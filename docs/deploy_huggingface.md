# Putting DeepQ Learning Lab online (Hugging Face Spaces)

The lab needs a running Python server (the Atari emulator, PyTorch and live
WebSockets), so static hosts such as GitHub Pages cannot run the games. A
**Hugging Face Docker Space** can: 2 CPU cores and 16 GB of RAM, which is plenty
for playing against the trained models.

> **Cost (checked November 2026):** Hugging Face now requires a PRO subscription
> to run Docker (and Gradio) Spaces on its basic CPU hardware; only static Spaces
> are free. Creating the Space on a free account fails with "402 Payment Required".
> The same `Dockerfile` also runs on any Docker host (Railway, a cloud VM).

## What gets deployed

- `Dockerfile` builds the React UI, installs the CPU build of PyTorch, the Atari
  emulator and two-player Pong, and starts `dqn_project.pong_server` on port 7860.
- The server runs in **public mode** (`DQN_LAB_PUBLIC=1`): it answers any host
  name, but only its own pages may use the API and WebSockets (the browser's
  `Origin` must match the host). Locally the server stays localhost-only.
- `scripts/deploy_hf_space.py` uploads only what the image needs: `src/`,
  `trained_models/`, the web sources and the packaging files (about 240 MB).
  `experiments/`, `results/` and notebooks stay on your computer. The Space gets
  its own README, which holds its settings (`sdk: docker`, `app_port: 7860`).

## First deployment

1. Create a free account at <https://huggingface.co/join>.
2. Create an access token with **Write** permission at
   <https://huggingface.co/settings/tokens>.
3. In the DQN folder, log in from WSL (paste the token when asked; it is stored
   in your WSL home folder, never in this repository):

   ```bash
   wsl --cd "$PWD" ~/.venvs/dqn-lab/bin/hf auth login
   ```

4. Upload, replacing `<username>` with your Hugging Face user name:

   ```bash
   wsl --cd "$PWD" ~/.venvs/dqn-lab/bin/python scripts/deploy_hf_space.py <username>/deepq-learning-lab
   ```

5. Open `https://huggingface.co/spaces/<username>/deepq-learning-lab`. The first
   build takes about 10 minutes (it compiles the two-player Pong emulator); the
   **Logs** tab shows progress. The app is also reachable directly at
   `https://<username>-deepq-learning-lab.hf.space`.

Use `--dry-run` to list the files without uploading, and `--private` on the first
run to create a Space only you can see. Outside WSL, install the uploader with
`pip install -e ".[deploy]"`.

## Updating

Run step 4 again. Only changed files are uploaded, and the Space rebuilds itself.

## Running the same image elsewhere

```bash
docker build -t deepq-lab .
docker run -p 7860:7860 deepq-lab
```

## Limits of a basic CPU Space

- **One game at a time.** The server runs a single game; a second visitor sees
  the first visitor's match. Fine for demos and classes.
- **Sleeps when idle.** After about 48 hours without visitors the Space sleeps
  and takes a minute or two to wake on the next visit.
- **CPU only.** Playing is unaffected; learning between Pong matches is slower
  than on a GPU.
- **Nothing learned online is kept.** The Space's disk is reset on every restart
  or redeploy, so sessions and Breakout scores start fresh; the shipped trained
  models always come back.
- **Input lag.** The game picture is streamed from the server, so players far
  from it notice some delay.

# Docker

This folder contains everything needed to build and run the project Docker image.

## Prerequisites
- Docker with `docker compose` v2
- Repository root available as build context

## Build the image
From the repository root:

```bash
docker build -f docker/Dockerfile -t multiagent-rlrm .
```

## Quick run
Drop into a shell with all dependencies installed:

```bash
docker run --rm -it multiagent-rlrm bash
```

Sanity check:

```bash
docker run --rm -i multiagent-rlrm python - <<'PY'
from multiagent_rlrm import __version__
from multiagent_rlrm.environments.frozen_lake.ma_frozen_lake import MultiAgentFrozenLake

print("Multi-Agent RLRM version:", __version__)
env = MultiAgentFrozenLake(width=4, height=4, holes=[(1, 1)])
print("Environment:", env.metadata)
PY
```

## Develop with Docker Compose
The `docker-compose.yml` mounts the repository to `/workspace` for interactive work:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm rlrm bash
```

Run a script or module inside the mounted container:

```bash
docker compose -f docker/docker-compose.yml run --rm rlrm \
  python -m multiagent_rlrm.environments.frozen_lake.frozen_lake_main
```

## Notes
- `SDL_VIDEODRIVER=dummy` and `MPLCONFIGDIR=/tmp/matplotlib` avoid needing a display server.
- For GPU/CUDA, swap the base image in `Dockerfile` for a compatible NVIDIA variant and add the proper runtime in Compose.
- The image installs dependencies from `requirements.txt` and then installs the package editable (`pip install -e .`) to ease iteration.

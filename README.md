# 🦎 axolotl-forge

Plug-and-play [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) fine-tuning image. Give it a model link and a
dataset link; an AI of your choice reads 5 random sectors of the data, the plugin list and the machine's GPUs and writes
a balanced config; it trains, merges, and uploads to the Hugging Face org you pick. Everything is purple. No wandb.

| | |
|---|---|
| base | `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` · Python 3.12 via **uv** |
| training | axolotl **0.21.1** · torch **2.14.1** (cu130) · transformers 5.20 · peft 0.21 · trl 1.14 |
| multi-GPU | DeepSpeed ZeRO 1/2/3 (+offload), FSDP2, context parallel (ringmaster), expert parallel, Ray |
| plugins | Cut Cross Entropy, Liger, MoE kernels (ScatterMoE/SonicMoE), DenseMixer, Spectrum, MixLoRA, Grokfast, KD, LM-Eval, Diffusion, Diffusion-decision, FLA sm120 shim, LLMCompressor, SwanLab, NeMo Gym, Hatchery |
| extras | optimizers pack (GaLore, APOLLO, LOMO, optimi, CAME, SCAO, Q-GaLore, schedule-free), FLA kernels, HF hub kernels (flash-attn 2/3) |
| box | ssh, zsh + starship, zellij, tmux, nvtop, lsd, git-lfs, TensorBoard |
| off | wandb (`WANDB_MODE=disabled`), axolotl telemetry, HF telemetry |

## Build

The image is ~30 GB and needs no GPU to build. Two ways:

**GitHub Actions (free, nothing to install).** Push this folder to a GitHub repo; the workflow in
[.github/workflows/build-image.yml](.github/workflows/build-image.yml) runs on every push to `main` (or from the Actions
tab → *build image* → *Run workflow*) and pushes `ghcr.io/kibafyi/axolotl-forge:latest`. It takes roughly
1–1.5 h the first time; later changes to `forge/` reuse the cached layers. To pull it on RunPod/Vast either make the
package public (GitHub → your profile → Packages → axolotl-forge → Package settings) or add ghcr.io registry
credentials. Tick *also push to Docker Hub* (with `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repo secrets) to push there too.

**Any Linux box with Docker and ~80 GB free disk:**

```bash
PUSH=1 ./build.sh you/axolotl-forge
```

On Windows (Docker Desktop):

```powershell
.\build.ps1 -Image you/axolotl-forge -Push
```

Hopper/Blackwell extras (Flash-Attention 4 and DeepEP for NVLink expert parallel) are opt-in:
`WITH_FA4=1 WITH_DEEPEP=1 DEEPEP_ARCH=9.0 ./build.sh you/axolotl-forge` (`DEEPEP_ARCH=10.0` for B200).

## Use it

### The setup screen

Log in (ssh, `docker run -it`, `docker exec -it ... zsh`) and the wolfgon pops up in the middle of the terminal with one
question under it at a time:

1. Hugging Face token (checked live; your account + orgs are fetched for the upload pick-lists)
2. AI key: Anthropic `sk-ant-…` or OpenRouter `sk-or-…` (checked live)
3. model link · 4. dataset link (typos / gated repos are caught right away)
5. Claude Sonnet 5.5 or Claude Opus 5.5
6. model name
7. where the LoRA goes → `<you or org>/<name>-lora`
8. where the merged model goes → `<you or org>/<name>`
9. Discord webhook (optional)

A summary lets you change any answer (`1-9`), add notes for Claude (`n`) or flip private/public (`p`). Then: the model and
dataset download, Claude reads 5 random sectors + the plugins + your GPUs and writes the config, and you look through it
(`Enter` starts training · `e` edit in nano · `d` dataset samples · `r` tell Claude what to change · `b` rules-engine
baseline). Training starts. If it crashes, the end of the log goes back to Claude, which fixes the config and training
retries. After 3 failed attempts (`FORGE_TRAIN_ATTEMPTS`) Discord gets a ❌ ping and you see the error. On success the LoRA
is merged, both repos are uploaded, and Discord gets **training done → uploaded → all finished**. Then you're back on the
start screen, answers kept (Enter through them or change what you need).

Type **`cmd-exit`** at any prompt to leave; `forge setup` brings it back. It runs inside tmux session `forge`, so a dropped
ssh connection never kills training; just log in again to re-attach. `FORGE_SETUP_SCREEN=0` turns the pop-up off.
Keys and the webhook are kept in memory only; the other answers are remembered in `/workspace/forge/setup.json`.

| first screen | summary |
|---|---|
| ![setup screen](docs/setup-1-token.svg) | ![summary](docs/setup-summary.svg) |

The art lives in [forge/assets/wolfgon.txt](forge/assets/wolfgon.txt) and is printed byte-for-byte: every line gets the same
left padding to center it, and characters are only coloured (`@%#*+` hot pink, `-=` deep purple, `:.` purple).

```bash
docker run --gpus all -it --rm -v $PWD/workspace:/workspace you/axolotl-forge
```

**Hands-off** – set `FORGE_MODEL` + `FORGE_DATASET` (e.g. in a RunPod/Vast template, see [.env.example](.env.example)).
The container starts the whole pipeline on boot inside tmux session `forge`, mirrors the log to `docker logs`, and you
can watch it with `forge attach` (or `forge logs -f`). Nothing asks questions; uploads go to `FORGE_HUB_NAMESPACE`
(defaults to your own account).

```bash
docker run --gpus all -d -p 2222:22 -v /data/ws:/workspace \
  -e FORGE_MODEL=Qwen/Qwen3-8B \
  -e FORGE_DATASET=https://huggingface.co/datasets/mlabonne/FineTome-100k \
  -e ANTHROPIC_API_KEY=... -e HF_TOKEN=... -e FORGE_HUB_NAMESPACE=my-org \
  -e PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
  you/axolotl-forge
```

### Links it understands (models and datasets)

`org/name` · `https://huggingface.co/[datasets/]org/name[/tree|blob|resolve/<rev>/<path>]` · `hf://...` ·
any `https://` file (resumable; `.jsonl .json .parquet .csv .arrow`, `.gz/.zst`, `.zip .tar.gz .tar.zst`) ·
public Google Drive file links · `s3:// gs:// az:// oci://` · local paths. Folders with `train` and `test` files use
only the train files.

### The AI config writer

| provider | key | default model |
|---|---|---|
| Claude | `ANTHROPIC_API_KEY` | `claude-opus-5-5` (official SDK, streaming, adaptive thinking, refusal fallback on) |
| OpenAI | `OPENAI_API_KEY` | `gpt-6.1-terra` |
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-opus-5.5` |
| custom | `FORGE_LLM_BASE_URL` (+`FORGE_LLM_API_KEY`) | first model the endpoint lists; vLLM, Ollama, LM Studio, Together, Groq, DeepSeek… `FORGE_LLM_API_STYLE=anthropic` for Anthropic-compatible endpoints |
| none | – | built-in rules engine |

Override with `FORGE_LLM_MODEL`; if a default model is ever retired, forge switches to the newest one the provider lists.

What the AI gets: the GPUs (count, VRAM, arch, NVLink), CPU/RAM/disk, the model's architecture/size/MoE/context/chat
template, **5 random sectors** of the dataset (the file is cut into 5 equal slices and one record is read at a random
spot in each; fast on 100 GB files), a 256-record profile (fields, detected format, real token-length p50/p90/p99 with
the model's tokenizer), every installed plugin with its exact YAML and when to use it, and a **baseline config from
forge's rules engine** computed from a VRAM model of the machine.

What happens to the answer: guardrails lock paths, strip wandb/hub keys, drop plugins that aren't installed (and their
keys), keep CCE and Liger's fused CE exclusive, fix attention/precision for the GPU generation, remove multi-GPU keys
on one GPU, then **axolotl's own validator** checks it. If axolotl rejects it, the error goes back to the AI to repair
(`FORGE_MAX_REPAIRS`, default 2); if it still fails you get the validated baseline. In guided mode you can accept, edit,
regenerate with extra instructions, or switch to the baseline.

### After training

LoRA/QLoRA runs are merged (`FORGE_MERGE`), then uploaded with a model card (loss, steps, hardware, config):
merged model → `namespace/name`, adapter → `namespace/name-lora` (`FORGE_UPLOAD_WHAT` to change). Full fine-tunes upload
the model. A training crash goes to the AI for a fixed config and retries up to `FORGE_TRAIN_ATTEMPTS` (3) times.
`FORGE_DISCORD_WEBHOOK` pings Discord on training done / uploaded / all finished / failed. `FORGE_ON_FINISH_CMD` runs at
the very end, e.g. `runpodctl stop pod $RUNPOD_POD_ID`. Re-running the same config resumes from the last checkpoint.

## Commands

| | |
|---|---|
| `forge` / `forge setup` | the setup screen |
| `forge run -m MODEL -d DATASET [-g "goal"] [--yes]` | full pipeline, no screen (templates / scripts) |
| `forge config -m MODEL -d DATASET` | only write a config |
| `forge train CONFIG.yml` | train + merge + upload an existing config |
| `forge upload CONFIG.yml --namespace ask` | upload a finished run |
| `forge sample DATASET -n 5 [--json]` | show random sectors + profile (and what the AI sees) |
| `forge orgs` | accounts/orgs your HF token can push to |
| `forge models --provider openrouter` | list a provider's models |
| `forge plugins` | plugin status in this image |
| `forge doctor` / `forge hw` | versions, keys, GPUs |
| `forge logs -f` / `forge attach` | follow the background run |
| `purple <cmd>` | run anything with purple output |

Files live in `/workspace/forge/{configs,datasets,models,outputs,logs,runs}`; HF cache in `/workspace/.cache`.
Mount `/workspace` to keep them.

## Purple

The shell sets the terminal's default text colour with OSC 10, colours what you type (`zle_highlight`), the prompt
(starship), `ls`/`grep`/`less`/`git`, tmux panes and zellij. Training output runs through a PTY that rewrites every ANSI
colour to a purple shade (errors stay hot pink so they still stand out), and the same goes for `docker logs`.
Change the shade with `FORGE_PURPLE=#C77DFF`; `FORGE_PURPLE_STRICT=0` keeps tools' own colours.

## Notes

- **wandb** – axolotl 0.21 imports `wandb` at module level, so the package stays installed but is disabled
  everywhere and never configured. Uninstalling it would break `axolotl train`.
- **Not included** – MoRA (needs a PEFT fork that conflicts with axolotl's pinned `peft==0.21.2`), vLLM / mamba-ssm /
  auto-gptq / fbgemm-gpu extras (axolotl marks them as conflicting), and the cloud-launcher integrations
  (Modal/Baseten/Nebius/CoreWeave), which launch jobs elsewhere rather than on this box.
- **LLMCompressor** – axolotl marks it as conflicting, but 0.15 fits the 0.21.1 pins, so it's installed against the
  pinned core. If that ever stops resolving, the build skips it and notes it in `/opt/forge/build-notes.txt`.
- amd64 only (that's where the CUDA training stack lives).

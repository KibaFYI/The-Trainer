# syntax=docker/dockerfile:1.7
# ───────────────────────────────────────────────────────────────────────────────
#   axolotl-forge · plug-and-play Axolotl fine-tuning image (purple edition)
#
#   CUDA 13.0 · Python 3.12 · torch 2.14.1 · axolotl 0.21.1 · uv
#   every Axolotl plugin whose deps can co-exist, AI config generation,
#   auto-upload to the Hugging Face Hub, ssh + zsh + starship + zellij + tmux.
#   wandb is switched off everywhere (axolotl hard-imports it, so the package stays).
# ───────────────────────────────────────────────────────────────────────────────
ARG CUDA_VERSION=13.0.3
ARG UBUNTU_VERSION=24.04
ARG UV_VERSION=0.13.7

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.14.1
ARG TORCH_BACKEND=cu130
ARG AXOLOTL_VERSION=0.21.1
# deepspeed  -> ZeRO 1/2/3 multi-GPU          ringmaster -> context parallel (multi-GPU long context)
# fla        -> linear-attention kernels       optimizers -> galore/apollo/lomo/optimi/came/scao/q-galore
# ray        -> ray train launcher             kernels-curated -> HF hub kernels (flash-attn, scattermoe...)
ARG AXOLOTL_EXTRAS=deepspeed,optimizers,ray,fla,ringmaster,kernels-curated
ARG CCE_REF=v0.2.0
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0+PTX"
ARG MAX_JOBS=8
# optional extras (1 = on)
ARG WITH_LLMCOMPRESSOR=1
ARG WITH_FA4=0
ARG WITH_DEEPEP=0
ARG DEEPEP_REF=v1.2.1
ARG DEEPEP_ARCH="9.0"
ARG ZELLIJ_VERSION=v0.45.1
ARG STARSHIP_VERSION=v1.27.0

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TERM=xterm-256color \
    COLORTERM=truecolor \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/forge/venv \
    PATH=/opt/forge/venv/bin:/opt/forge/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS}

# ── system packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git git-lfs build-essential ninja-build pkg-config cmake \
        libaio-dev libibverbs-dev librdmacm-dev rdma-core ibverbs-providers ibverbs-utils infiniband-diags \
        openssh-server zsh tmux sudo vim nano less htop nvtop lsd jq unzip zstd pigz rsync \
        iproute2 iputils-ping lsof procps psmisc tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --skip-repo \
    && chsh -s /bin/zsh root

# ── starship prompt + zellij ──────────────────────────────────────────────────
RUN curl -fsSL "https://github.com/starship/starship/releases/download/${STARSHIP_VERSION}/starship-x86_64-unknown-linux-musl.tar.gz" | tar -xz -C /usr/local/bin \
 && curl -fsSL "https://github.com/zellij-org/zellij/releases/download/${ZELLIJ_VERSION}/zellij-x86_64-unknown-linux-musl.tar.gz" | tar -xz -C /usr/local/bin \
 && chmod +x /usr/local/bin/starship /usr/local/bin/zellij

# ── uv + python venv ───────────────────────────────────────────────────────────
COPY --from=uv /uv /uvx /usr/local/bin/
RUN uv python install ${PYTHON_VERSION} \
 && uv venv --seed --python ${PYTHON_VERSION} /opt/forge/venv

# ── torch (CUDA 13) + cutlass DSL / nvshmem (SonicMoE, DeepEP) ─────────────────
RUN --mount=type=cache,target=/root/.cache/uv \
    set -e; \
    NV_BACKEND="${TORCH_BACKEND%?}"; \
    uv pip install packaging setuptools wheel psutil ninja py-cpuinfo; \
    UV_TORCH_BACKEND=${TORCH_BACKEND} uv pip install \
        "torch==${TORCH_VERSION}" torchvision apache-tvm-ffi \
        "nvidia-cutlass-dsl[${NV_BACKEND}]" "nvidia-nvshmem-${NV_BACKEND}"; \
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ── axolotl + multi-GPU extras + cut cross entropy ────────────────────────────
RUN --mount=type=cache,target=/root/.cache/uv \
    set -e; \
    echo "torch==${TORCH_VERSION}" > /tmp/torch-pin.txt; \
    uv pip install --no-build-isolation-package deepspeed -c /tmp/torch-pin.txt \
        "axolotl[${AXOLOTL_EXTRAS}]==${AXOLOTL_VERSION}" \
        "huggingface_hub>=1.5.0" "httpx<1" kernels-data; \
    uv pip install -c /tmp/torch-pin.txt \
        "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@${CCE_REF}"; \
    mkdir -p /opt/forge; \
    uv pip freeze | grep -iE '^(torch|torchvision|transformers|peft|accelerate|datasets|trl|tokenizers|axolotl|axolotl-ringmaster|deepspeed|triton|liger-kernel|bitsandbytes|huggingface-hub|flash-linear-attention|fla-core|torchao|xformers)==' \
        > /opt/forge/constraints.txt; \
    echo "core stack pinned:"; cat /opt/forge/constraints.txt

# ── extra plugin deps (held to the core pins so nothing drifts) ───────────────
#   densemixer -> DenseMixerPlugin     swanlab -> SwanLabPlugin
#   tinker + math-verify -> HatcheryPlugin     orjson -> KDPlugin
#   anthropic/openai/rich -> forge's AI config generator + purple UI
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -c /opt/forge/constraints.txt \
        densemixer swanlab tinker math-verify orjson \
        anthropic openai rich pyarrow pyyaml requests zstandard tensorboard

# llm_compressor plugin: axolotl marks it as conflicting, but 0.15 fits the 0.21.1 pins.
# It is tried against the pinned core; if it ever stops resolving the build keeps going.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${WITH_LLMCOMPRESSOR}" = "1" ]; then \
        uv pip install -c /opt/forge/constraints.txt "llmcompressor>=0.10.0" \
        || echo "llmcompressor: does not co-install with the pinned core stack - skipped" | tee -a /opt/forge/build-notes.txt; \
    fi

# Flash Attention 4 + quack kernels (Hopper / Blackwell). Off by default: FA2/FA3 come from HF hub kernels.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${WITH_FA4}" = "1" ]; then \
        uv pip install -c /opt/forge/constraints.txt \
            "flash-attn-4[cu13]==4.0.0b35" "quack-kernels[cu13]==0.6.5"; \
    fi

# DeepEP for the expert_parallel `deep_ep` backend (NVLink H100/B200). The `torch` backend needs nothing.
RUN if [ "${WITH_DEEPEP}" = "1" ]; then \
        set -e; \
        git clone --depth 1 --branch "${DEEPEP_REF}" https://github.com/deepseek-ai/DeepEP.git /tmp/DeepEP; \
        cd /tmp/DeepEP; \
        TORCH_CUDA_ARCH_LIST="${DEEPEP_ARCH}" MAX_JOBS=${MAX_JOBS} \
        CFLAGS=-I/usr/local/cuda/include/cccl CXXFLAGS=-I/usr/local/cuda/include/cccl \
        NVCC_PREPEND_FLAGS=-I/usr/local/cuda/include/cccl \
            python setup.py bdist_wheel -d /tmp/wheels; \
        uv pip install --no-deps /tmp/wheels/deep_ep-*.whl; \
        cd / && rm -rf /tmp/DeepEP /tmp/wheels; \
        python -c "import deep_ep; print(deep_ep.Buffer)"; \
    fi

# ── deepspeed ZeRO configs (multi-GPU) ────────────────────────────────────────
RUN mkdir -p /opt/forge/deepspeed_configs && cd /opt/forge/deepspeed_configs \
 && for f in zero1 zero1_torch_compile zero2 zero2_torch_compile zero3 zero3_bf16 \
             zero3_bf16_cpuoffload_params zero3_bf16_cpuoffload_all; do \
        curl -fsSL "https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/v${AXOLOTL_VERSION}/deepspeed_configs/${f}.json" -o "${f}.json"; \
    done

# ── forge (CLI + AI config generator + uploader) and the purple shell ─────────
ENV PYTHONPATH=/opt/forge \
    FORGE_HOME=/workspace/forge \
    FORGE_PURPLE="#B388FF" \
    HF_HOME=/workspace/.cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    TOKENIZERS_PARALLELISM=false \
    WANDB_MODE=disabled \
    WANDB_DISABLED=true \
    WANDB_SILENT=true \
    AXOLOTL_DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

COPY forge /opt/forge/forge
COPY rootfs/ /
# strip Windows line endings (in case the repo was checked out with CRLF), then self-test the plugins
RUN for f in /usr/local/bin/forge /usr/local/bin/forge-entrypoint /usr/local/bin/purple /etc/forge/purple.sh \
             /root/.zshrc /root/.bashrc /root/.tmux.conf /root/.gitconfig /root/.config/starship.toml \
             /root/.config/zellij/config.kdl /etc/ssh/sshd_config.d/50-forge.conf; do sed -i 's/\r$//' "$f"; done \
 && chmod +x /usr/local/bin/forge /usr/local/bin/forge-entrypoint /usr/local/bin/purple \
 && python -m forge plugins --build-check --write /opt/forge/plugins.json \
 && echo "axolotl-forge build: axolotl ${AXOLOTL_VERSION} / torch ${TORCH_VERSION} / ${TORCH_BACKEND} / py${PYTHON_VERSION}" >> /opt/forge/build-notes.txt

WORKDIR /workspace
EXPOSE 22 6006
ENTRYPOINT ["/usr/local/bin/forge-entrypoint"]
CMD []

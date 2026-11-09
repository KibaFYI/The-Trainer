# ── axolotl-forge bash (zsh is the default shell; this covers `docker exec ... bash`)
[ -z "$PS1" ] && return
[ -r /etc/forge/env.sh ] && . /etc/forge/env.sh
. /etc/forge/purple.sh

export PATH="/opt/forge/venv/bin:/opt/forge/bin:$PATH"
HISTFILE="${FORGE_BASH_HISTFILE:-/workspace/.bash_history}"
HISTSIZE=50000
HISTFILESIZE=50000
shopt -s histappend checkwinsize

eval "$(starship init bash)"

# typed text + output stay purple
_forge_purple_ps1() { PS1="${PS1}\[${FORGE_SGR}\]"; forge_set_term_fg; }
PROMPT_COMMAND="${PROMPT_COMMAND:+${PROMPT_COMMAND};}_forge_purple_ps1"
PS0="${FORGE_SGR}"

alias ls='lsd'
alias ll='lsd -lah'
alias la='lsd -A'
alias gpu='purple nvidia-smi'
alias fl='forge logs -f'
alias fa='forge attach'
alias setup='forge setup'

# first-boot setup screen (see ~/.zshrc)
if [ -z "$TMUX" ] && [ -z "$ZELLIJ" ] && [ "${FORGE_SETUP_SCREEN:-1}" != "0" ] && [ ! -e /tmp/forge-setup-dismissed ]; then
    forge setup
fi

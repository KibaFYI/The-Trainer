# ── axolotl-forge zsh ─────────────────────────────────────────── purple all the way
[[ -r /etc/forge/env.sh ]] && source /etc/forge/env.sh
source /etc/forge/purple.sh

export EDITOR="${EDITOR:-nano}"
export PATH="/opt/forge/venv/bin:/opt/forge/bin:$PATH"

HISTFILE="${FORGE_HISTFILE:-/workspace/.zsh_history}"
HISTSIZE=50000
SAVEHIST=50000
setopt share_history inc_append_history hist_ignore_dups hist_reduce_blanks
setopt autocd interactive_comments no_beep

autoload -Uz compinit && compinit -u
zstyle ':completion:*' menu select
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"

# what you type is purple
zle_highlight=(default:fg=${FORGE_PURPLE} region:bg=#4A148C isearch:fg=#EA80FC,bold paste:fg=#D1B3FF suffix:fg=#9575CD)

# command output starts purple; OSC 10 keeps it purple after programs reset colours
autoload -Uz add-zsh-hook
_forge_preexec() { print -n -- "$FORGE_SGR" }
_forge_precmd()  { forge_set_term_fg }
add-zsh-hook preexec _forge_preexec
add-zsh-hook precmd  _forge_precmd

eval "$(starship init zsh)"

# aliases
alias ls='lsd'
alias ll='lsd -lah'
alias la='lsd -A'
alias lt='lsd --tree --depth 2'
alias gpu='purple nvidia-smi'
alias gpuw='watch -c -n 1 nvidia-smi'
alias fl='forge logs -f'
alias fa='forge attach'
alias ws='cd /workspace'
alias ff='cd $FORGE_HOME'

alias setup='forge setup'

# first-boot setup screen: the wolfgon pops up when you log in (in tmux, so training
# survives a dropped ssh). cmd-exit leaves it for this boot; `forge setup` brings it back.
if [[ -o interactive && -z "$TMUX" && -z "$ZELLIJ" && "${FORGE_SETUP_SCREEN:-1}" != "0" \
      && ! -e /tmp/forge-setup-dismissed ]]; then
    export FORGE_BANNER_SHOWN=1
    forge setup
fi

if [[ -o interactive && -z "$FORGE_BANNER_SHOWN" ]]; then
    export FORGE_BANNER_SHOWN=1
    forge banner 2>/dev/null
fi

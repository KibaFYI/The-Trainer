# axolotl-forge: make the whole terminal purple. Sourced by bash and zsh.
#   FORGE_PURPLE       main text colour   (default #B388FF)
#   FORGE_PURPLE_DEEP  accents            (default #7C4DFF)

: "${FORGE_PURPLE:=#B388FF}"
: "${FORGE_PURPLE_DEEP:=#7C4DFF}"
export FORGE_PURPLE FORGE_PURPLE_DEEP

_forge_hex="${FORGE_PURPLE#\#}"
_forge_r=$((16#${_forge_hex:0:2}))
_forge_g=$((16#${_forge_hex:2:2}))
_forge_b=$((16#${_forge_hex:4:2}))
FORGE_SGR="$(printf '\033[38;2;%d;%d;%dm' "$_forge_r" "$_forge_g" "$_forge_b")"
export FORGE_SGR
unset _forge_hex _forge_r _forge_g _forge_b

# OSC 10 sets the terminal's *default* foreground, so even text that resets
# colours (\e[0m) stays purple. Supported by xterm.js (RunPod/Vast web
# terminals), Windows Terminal, iTerm2, VS Code, kitty, alacritty, foot...
forge_set_term_fg() {
    if [ -t 1 ]; then
        printf '\033]10;%s\007' "$FORGE_PURPLE"
        printf '%s' "$FORGE_SGR"
    fi
}

# purple-ish palette for ls / lsd / grep / less / man / gcc
export LS_COLORS="rs=0:di=1;38;2;124;77;255:ln=38;2;234;128;252:mh=38;2;209;179;255:pi=38;2;149;117;205:so=38;2;149;117;205:do=38;2;149;117;205:bd=38;2;149;117;205:cd=38;2;149;117;205:or=1;38;2;255;92;203:mi=1;38;2;255;92;203:su=1;38;2;255;92;203:sg=1;38;2;255;92;203:ca=38;2;179;136;255:tw=1;38;2;124;77;255:ow=1;38;2;124;77;255:st=1;38;2;124;77;255:ex=1;38;2;234;128;252:fi=38;2;179;136;255:*.yml=38;2;209;179;255:*.yaml=38;2;209;179;255:*.json=38;2;209;179;255:*.jsonl=38;2;206;147;216:*.parquet=38;2;206;147;216:*.safetensors=1;38;2;206;147;216:*.bin=38;2;206;147;216:*.gz=38;2;149;117;205:*.zst=38;2;149;117;205:*.tar=38;2;149;117;205:*.zip=38;2;149;117;205:*.py=38;2;225;190;231:*.sh=38;2;225;190;231:*.md=38;2;237;231;246:*.log=38;2;149;117;205"
export GREP_COLORS="ms=1;38;2;234;128;252:mc=1;38;2;234;128;252:sl=:cx=:fn=38;2;124;77;255:ln=38;2;209;179;255:bn=38;2;209;179;255:se=38;2;149;117;205"
export GCC_COLORS="error=1;38;2;255;92;203:warning=1;38;2;209;179;255:note=38;2;179;136;255:caret=1;38;2;234;128;252:locus=38;2;124;77;255:quote=38;2;179;136;255"
export LESS="-R"
export LESS_TERMCAP_mb="$(printf '\033[1;38;2;234;128;252m')"
export LESS_TERMCAP_md="$(printf '\033[1;38;2;124;77;255m')"
export LESS_TERMCAP_me="$(printf '\033[0m')${FORGE_SGR}"
export LESS_TERMCAP_so="$(printf '\033[1;38;2;26;16;40;48;2;179;136;255m')"
export LESS_TERMCAP_se="$(printf '\033[0m')${FORGE_SGR}"
export LESS_TERMCAP_us="$(printf '\033[4;38;2;234;128;252m')"
export LESS_TERMCAP_ue="$(printf '\033[0m')${FORGE_SGR}"

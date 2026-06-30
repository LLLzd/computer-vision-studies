#!/bin/zsh
# Project-scoped tab completion. Does NOT hook global `python`.
#
# Usage (once per terminal session):
#   source enable_completion.zsh
#
# Then use:
#   rdv inputs/object.MOV --row-step<Tab>

if [[ -z "${ZSH_VERSION:-}" ]]; then
  echo "enable_completion.zsh is for zsh only." >&2
  return 1 2>/dev/null || exit 1
fi

autoload -Uz compinit
compinit -i 2>/dev/null || compinit

# Undo a previously sourced argcomplete hook that breaks `python` file completion.
compdef -d python 2>/dev/null

ROW_DELAY_VIDEO_DIR="${0:A:h}"

rdv() {
  python "${ROW_DELAY_VIDEO_DIR}/process.py" "$@"
}

_rdv() {
  _arguments -C \
    '(-h --help)'{-h,--help}'[Show help]' \
    '(-o --output)'{-o,--output}'[Output path]:file:_files' \
    '--row-step[Rows per frame delay]:integer:' \
    '--fill-mode[Fill mode]:(clamp black)' \
    '1:input video:_files'
}

compdef _rdv rdv

echo "Tab completion enabled for: rdv ..."

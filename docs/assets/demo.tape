# Regenerate docs/assets/demo.gif:
#   brew install vhs
#   cd <repo root>
#   vhs docs/assets/demo.tape
#
# The hidden setup below shadows `uvx` with the local checkout, so the GIF
# is generated from the exact code the release ships while showing the
# command a user actually types (`uvx wattage demo`) — the published
# package produces this same output once the release is live.

Output docs/assets/demo.gif

Require uv

Set Shell "zsh"
Set FontSize 16
Set Width 1200
Set Height 900
Set Theme "GitHub Dark"
Set Padding 24
Set TypingSpeed 45ms
Set WindowBar Colorful
Set BorderRadius 8

Hide
Type `uvx() { uv run "$@" }`
Enter
Type "clear"
Enter
Show

Type "uvx wattage demo"
Sleep 500ms
Enter

Sleep 6s

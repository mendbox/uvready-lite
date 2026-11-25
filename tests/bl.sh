#!/bin/zsh
# Run ONE Blender of a given version in the background, under the memory guard, with a THROWAWAY user folder.
# usage: bl.sh <42|45|52> <profile name> <blender args...>
# The user folder is tests/profiles/<profile name> (config, scripts, extensions): the owner's real Blender
# configuration is never read or written.
# Rules of this machine (24 GB, froze once): one background Blender at a time, every run under memguard.sh 10.
# Exit 75 = the guard refused to start (memory pressure): retried six times. Exit 137 = killed by the guard.
HERE=${0:A:h}
case $1 in
  42) BL=$HOME/Applications/blender-versions/Blender-4.2.23.app/Contents/MacOS/Blender ;;
  45) BL=$HOME/Applications/blender-versions/Blender-4.5.14.app/Contents/MacOS/Blender ;;
  52) BL=/Applications/Blender.app/Contents/MacOS/Blender ;;
  *) echo "version must be 42, 45 or 52" >&2; exit 2 ;;
esac
U="$HERE/profiles/$2"; shift 2
mkdir -p "$U"
export BLENDER_USER_RESOURCES="$U"
export BLENDER_USER_CONFIG="$U/config" BLENDER_USER_SCRIPTS="$U/scripts" BLENDER_USER_EXTENSIONS="$U/extensions" BLENDER_USER_DATAFILES="$U/datafiles"
GUARD=${MEMGUARD:-$HOME/Projects/ai3d-gen/memguard.sh}
# wait while any other background Blender runs (process name exactly "Blender", arguments contain -b)
bg_blender() { for p in $(pgrep -x Blender); do ps -o args= -p $p 2>/dev/null | grep -q -- " -b" && return 0; done; return 1 }
n=0
while [[ -z $BL_PARALLEL ]] && bg_blender; do
  (( n++ > 240 )) && { echo "bl.sh: another background Blender is still running after 20 min, giving up" >&2; exit 76 }
  sleep 5
done
for try in 1 2 3 4 5 6; do
  # BL_GUI=1: a real window (tests/run_gui_smoke.sh); everything else runs in the background
  if [[ -n $BL_GUI ]]; then "$GUARD" ${MEMGUARD_GB:-10} "$BL" "$@"; else "$GUARD" ${MEMGUARD_GB:-10} "$BL" -b "$@"; fi; rc=$?
  [[ $rc != 75 ]] && exit $rc
  sleep 30
done
exit 75

#!/bin/zsh
# Headless test of the BUILT UVReady Lite on Blender 4.2, 4.5 and 5.2 (macOS), one Blender at a time, every run under
# the memory guard (tests/bl.sh, MEMGUARD_GB=6) in a throwaway user folder whose name has Cyrillic letters and a space.
# The paid UVReady zip (PAID_ZIP, default: the newest ../autouv/dist/uvready-*.zip) is installed next to Lite, so the
# coexistence checks run. usage: tests/run_all_versions.sh [out dir] ["42 45 52"]
D=${0:A:h}; R=${D:h}
O=${1:-$D/out/run}; [[ $O != /* ]] && O=$PWD/$O
VERS=${2:-"42 45 52"}
export MEMGUARD_GB=${MEMGUARD_GB:-6}
mkdir -p $O
[[ -z $NOBUILD ]] && { "$R/build.sh" > $O/build.log 2>&1 || { echo "BUILD FAILED (see $O/build.log)"; exit 1 } }
ZIP=$(ls -t $R/dist/uvready_lite-*.zip | head -1)
PAID_ZIP=${PAID_ZIP:-$(ls -t $R/../autouv/dist/uvready-*.zip 2>/dev/null | head -1)}
echo "ZIP $ZIP  PAID $PAID_ZIP"
blr() { local x; for try in $(seq 1 240); do "$D/bl.sh" "$@"; x=$?; [[ $x == 75 || $x == 137 ]] || return $x; sleep 60; done; return $x }
rc=0
for v in ${=VERS}; do
  PROFILE="лайт профиль $v"
  rm -rf "$D/profiles/$PROFILE"
  blr $v "$PROFILE" --command extension install-file -r user_default -e "$ZIP" > $O/$v-install.log 2>&1
  echo "BLENDER $v install exit $?"
  if [[ -n $PAID_ZIP ]]; then
    blr $v "$PROFILE" --command extension install-file -r user_default -e "$PAID_ZIP" >> $O/$v-install.log 2>&1
    echo "BLENDER $v paid install exit $?"
  fi
  blr $v "$PROFILE" --python "$D/test_lite.py" -- $O/$v > $O/$v.log 2>&1; x=$?
  line=$(grep "^CASE " $O/$v.log | tail -1)
  [[ -z $line ]] && line="CASE lite $v CRASH (blender exit $x, see $O/$v.log)"
  echo "$line"
  [[ $line == *PASS* ]] || rc=1
  grep -E "^CHECK.* FAIL|^NOTE|Traceback" $O/$v.log
done
exit $rc

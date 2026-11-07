#!/bin/sh
# A regression check of one CLI build against another: the same outputs and
# no slower decoding. Both builds run the same requests, one model process at
# a time. It fails when an output differs between the builds, when the batch
# check finds a difference, or when the candidate's median decode speed is
# below the baseline's by more than the baseline's own spread (at least 1%).
#
# Outputs: two greedy generations of 150 tokens, the logits of a perplexity
# run over the first 1,998 bytes of the checkpoint's README, and the batch
# check with two drafts. Speed: three 256-token decodes of the story prompt
# per build, the builds alternating.
#
# Before each model run it requires AC power, an idle GPU (at most 35% device
# use in three samples a second apart) and no other myds4 process. With
# UD-IQ2_XXS experts and a 40 GiB cache its 14 model runs take about five
# minutes.
#
# usage: tools/checks/regression.sh CANDIDATE BASELINE CHECKPOINT GGUF
set -eu
if [ $# -ne 4 ]; then
  echo "usage: $0 CANDIDATE BASELINE CHECKPOINT GGUF" >&2
  exit 2
fi
candidate=$1
baseline=$2
checkpoint=$3
gguf=$4
story="Write a short story about a lighthouse keeper who discovers a message in a bottle."
python="Write a Python function that parses an ISO 8601 date string and returns a datetime, with docstring and tests."

# Waits up to five minutes for three idle samples in a row; a busy desktop
# moment delays the next run instead of ending the check.
ready() {
  if ! pmset -g batt | grep -q "'AC Power'"; then
    echo "regression: not on AC power" >&2
    exit 1
  fi
  if pgrep -x myds4 >/dev/null || pgrep -x myds4-server >/dev/null; then
    echo "regression: another myds4 process runs" >&2
    exit 1
  fi
  idle=0
  waited=0
  while [ "$idle" -lt 3 ]; do
    use=$(ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
    if [ "${use:-100}" -le 35 ]; then
      idle=$((idle + 1))
    else
      idle=0
    fi
    if [ "$waited" -ge 300 ]; then
      echo "regression: the GPU stayed busy for five minutes (${use:-unknown}% device use)" >&2
      exit 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

# The outputs of one build, one line each.
outputs() {
  cli=$1
  ready
  echo "story: $("$cli" -m "$checkpoint" --experts "$gguf" --cache 40 -p "$story" -n 150 2>/dev/null | shasum | cut -c1-16)"
  ready
  echo "python: $("$cli" -m "$checkpoint" --experts "$gguf" --cache 40 -p "$python" -n 150 2>/dev/null | shasum | cut -c1-16)"
  ready
  echo "perplexity logits: $(head -c 1998 "$checkpoint/README.md" |
    xargs -0 "$cli" -m "$checkpoint" --experts "$gguf" --cache 40 --perplexity --logits-fd 3 --raw -p 3>&1 1>/dev/null 2>/dev/null |
    shasum | cut -c1-16)"
  ready
  "$cli" -m "$checkpoint" --experts "$gguf" --cache 40 -p "$story" --check-batch --draft 2 2>/dev/null | grep 'differ'
}

# One 256-token decode's speed in tokens per second.
speed() {
  ready
  tps=$("$1" -m "$checkpoint" --experts "$gguf" --cache 40 -p "$story" -n 256 2>&1 >/dev/null |
    tr '\r' '\n' | grep '^decode:' | sed -E 's/.* ([0-9.]+) tok\/s.*/\1/')
  if [ -z "$tps" ]; then
    echo "regression: $1 reported no decode speed" >&2
    exit 1
  fi
  echo "$tps"
}

failed=0
echo "== outputs: candidate"
new=$(outputs "$candidate")
echo "$new"
echo "== outputs: baseline"
old=$(outputs "$baseline")
echo "$old"
if [ "$new" != "$old" ]; then
  echo "FAIL: the candidate's outputs differ from the baseline's"
  failed=1
fi
if echo "$new$old" | grep -q da39a3ee5e6b4b0d; then  # the hash of no output
  echo "FAIL: a run produced no output"
  failed=1
fi
if echo "$new" | grep 'differ' | grep -v -q -E ': 0 (and 0 )?of '; then
  echo "FAIL: the batch check found different logits"
  failed=1
fi

new_speeds=""
old_speeds=""
for run in 1 2 3; do
  s=$(speed "$candidate")
  echo "decode run $run: candidate $s tok/s"
  new_speeds="$new_speeds $s"
  s=$(speed "$baseline")
  echo "decode run $run: baseline $s tok/s"
  old_speeds="$old_speeds $s"
done
if ! echo "$new_speeds|$old_speeds" | awk -F'|' '
  function median(s, v) { split(s, v, " "); asort3(v); return v[2] }
  function asort3(v, t) {
    if (v[1] > v[2]) { t = v[1]; v[1] = v[2]; v[2] = t }
    if (v[2] > v[3]) { t = v[2]; v[2] = v[3]; v[3] = t }
    if (v[1] > v[2]) { t = v[1]; v[1] = v[2]; v[2] = t }
  }
  function spread(s, v) { split(s, v, " "); asort3(v); return v[3] - v[1] }
  {
    n = median($1); o = median($2); allowed = spread($2)
    if (allowed < o / 100) allowed = o / 100
    printf "decode median: candidate %.2f, baseline %.2f tok/s (allowed drop %.2f)\n", n, o, allowed
    exit !(n >= o - allowed)
  }'; then
  echo "FAIL: the candidate decodes slower than the baseline"
  failed=1
fi
if [ "$failed" -eq 0 ]; then
  echo "regression: PASS"
else
  echo "regression: FAIL"
fi
exit "$failed"

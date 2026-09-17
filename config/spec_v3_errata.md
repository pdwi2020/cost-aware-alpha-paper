# Errata: config/spec_v3.yaml

`spec_v3.yaml` is the frozen specification for the Results in Engineering
revision. It was committed as `64b9feb` before the forward window was scored,
and its git blob hash `a36ad8991182e97bafd7a25e48d98588f35308cc` is recorded in
both entries of `results/forward_touch_log.json`.

**The file is deliberately not edited, including to correct the error below.**
It is the record of what was committed before the single touch, and a reader
should be able to retrieve exactly those bytes and confirm the hash. Errors in
it are corrected here instead.

## E1. Position-cap breach magnitude (descriptive comment only)

`screen0.position_rule` reads:

> The old post-hoc renormalisation breached the 5% cap on 10.7% of days
> (max weight 8.16%).

**The maximum weight was 9.09%, not 8.16%.**

Authoritative sources, which agree with each other:

- `results/search_ledger.csv`, row `clip_then_reinflate_position_cap`, quoting
  commit `254397d`: "replace clip-then-reinflate (re-inflated to 9.09% over a
  5% cap) with iterative water-fill".
- The manuscript, Section 9.2 and Section 12.9, both of which say 9.09%.

The value is a descriptive comment. It is not read by any code, so no result
depends on it. The 10.7%-of-days figure is unaffected.

## A note on why the hash is no longer load-bearing for the touch guard

`run_forward_holdout.py` originally matched prior touches on the spec blob
*and* the window. That made any edit to this file, including to a comment,
silently reset the single-touch counter: the log entries stopped matching, the
guard reported zero prior touches, and the window became re-scorable without
`--reevaluation-reason`. A single-touch mechanism that a one-character edit
disarms is not one.

The guard now matches on window and track. The blob is still recorded on every
touch, and a touch made under a different specification is reported as such
rather than ignored. The freeze is therefore protected by the log rather than
by the immutability of this file, which is the more robust arrangement, and
this file is kept byte-identical anyway because it costs nothing to do so.

See `tests/test_forward_touch_guard.py`.

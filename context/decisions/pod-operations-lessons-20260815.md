---
question: What did 2026-08-15 teach about renting, accepting and watching a training pod — and what would each lesson have cost again?
verdict: Eight measured lessons. The expensive one is that a pod must be ACCEPTED on its rate in the first 10 minutes, and the subtle one is that the naive way of doing that kills healthy pods.
status: MEASURED
date: 2026-08-15
measured_in: pods 5btpl229y7kuar (capped), 98yh51k6j6pv2h (healthy), 3h6c6y1dla6bsg (smoke)
question_derived: true
---
# Operational lessons from 2026-08-15 — each one already cost us something

One day cost **~$26 on a run that could never finish** and three near-misses that would each
have cost more. Every item below is measured, not advice.

## 1. 🔴 ACCEPT A POD ON ITS RATE, IN THE FIRST 10 MINUTES

`5btpl229y7kuar` ran **120.7 s/it** where the same recipe on the same-family GPU had run
**13.1 s/it** — 9.2× slower, constant from step 26. Nobody checked for 8 h 30. It died at step
252/1802 with **zero checkpoints** and **~$26** spent.

The cause was the host, not us:

```
clocks.sm            600 MHz of 3090      (19 % of rated)
SW Power Cap       : Active
temperature          36 °C                 → not thermal
utilization.memory   4 %                   → not I/O, not offload
```

⇒ **Read the rate within 10 minutes of the first step.** Killing a bad pod then costs ~$0.30.

## 2. ⚠️ …but the NAIVE way of doing that kills HEALTHY pods

On the **healthy** pod, **step 1 measured 112.38 s/it** — ten seconds from a "kill it" threshold
of 30 s/it. It is kernel compilation and the first dataloader fill, billed against one iteration.

Two rules that make the check correct:

- **Ignore steps 1–3.** Measure **deltas between consecutive `[step]` lines**.
- 🔴 **Never read tqdm's `s/it`** — it is a cumulative mean, so it drags step 1 forever. Its
  tell is an ETA that *plummets* instead of settling: `10297 → 5346 → 3694 → 2868` min.
- Under load the reliable discriminant is **`utilization.memory`**: **32 % healthy vs 4 % capped**.
  `utilization.gpu` reads 100 % in BOTH cases — a capped GPU is busy, just slow.

## 3. A rendered chain hardcodes the pod id it was rendered for

`armB_ep3_chain.sh` and its watchdog both carried `pods/5btpl229y7kuar`. Relaunching them on a
new pod would have left the `trap` and the watchdog pointing at a **dead** pod ⇒ **the new pod
would never stop itself.** Re-render (or copy + `sed`) the id, and `grep -o "pods/[a-z0-9]*"` the
result before launching. Keep the original — do not edit a teammate's script in place.

## 4. Checkpoint alarms must be read off `save_strategy`, not invented

`gen36_arm.py:554` sets `save_strategy="epoch"`. With `num_train_epochs=2` that is **exactly two
checkpoints, at steps 901 and 1802** — `ckpt/` is legitimately empty for the first ~3.3 h. An
alarm of "no checkpoint after step 200" is a guaranteed false positive.

🔑 It also explains the dead run with **one** fault instead of two: it never reached step 901, so
there was nothing to save. Slow *caused* no-checkpoints; they were not separate failures.

⇒ **Before step 901 the run is all-or-nothing.** Dying at hour 3 loses all three hours.

## 5. Disk thresholds must be derived from the operation that needs the space

The volume's wall is the **~640 GiB** the chain's own preflight uses, not the 670 GiB quota, and
🔴 **never `df`** (it reports the MooseFS cluster at 1.4 PB; a chain already died
`Disk quota exceeded` trusting it). Use `du -sx`.

The final merge writes **~52 GiB**, so the real threshold is `used + 52 ≤ 640` ⇒ **588 GiB**.
A first attempt set the alarm at 609, which would have fired **after** the merge was already
impossible — the worst kind of alarm, one that feels like monitoring and is not.

⇒ Alert at **575** (reaction room), treat **588** as the point of no return. And **poll fast
enough**: a 10-minute cadence let a **40 GiB jump in 2.5 h** pass unseen; 3 minutes did not.

## 6. Freeing space: verify the survivor BEFORE deleting the original

At 591 GiB the merge was 3 GiB short. Two 16.3 GiB copies of the same rung-42 checkpoint
existed — a directory and a `.zip`. **Delete the reconstructible one, keep the packaged one**
(the zip is likely what its owner wants to download), and prove the survivor is complete first:

```
zip:        17 entries, 16.3 GiB, testzip() -> None
directory:  17 files,   16.3 GiB
```

Plus a **304 MB adapter** that rebuilds the merge anyway ⇒ two independent recovery paths.
Deleting 16 GB of a teammate's work on an assumption is a bet; deleting it after this is a
decision. 🔴 Deleting third-party artifacts is the **lead's** call, never a watcher's.

## 7. `pkill` on a chain stops the pod

SIGTERM fires bash `EXIT` traps, and the chain's trap calls the RunPod stop endpoint. To swap a
job, kill **the training pid**, never the chain. It cost one restart.

## 8. 🔴 The RunPod API key is passed in `argv`

```
bash …/armB_ep23_v2_wd.sh <pid> rpa_…
```

It is therefore visible in **any `ps`** on the pod, and surfaces to anyone debugging something
unrelated. The exposure that matters is not an intruder — it is a `ps aux` pasted into a log, a
note or a message. ⇒ Pass it via **`RUNPOD_API_KEY`** or a `chmod 600` file read inside the
script. Do not change a watchdog that is currently running; fix it between runs.

## Sources

Pods `5btpl229y7kuar`, `98yh51k6j6pv2h`, `3h6c6y1dla6bsg`; `armB_ep3.log`;
`armB_ep23_v2.log`; `gen36_arm.py:554`; `chain40._shutdown_block`;
`experiments_segment/NOW.md` §"THREE POD-STOP LAYERS".

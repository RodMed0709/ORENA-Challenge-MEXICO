---
question: Submission 04's commit records that container images CANNOT be built on UNAM — "SELinux is Enforcing and blocks execmod inside containers for uaq_user, so buildah cannot run any RUN step" — which is why the 27B build was forced onto a 144 GB laptop partition. Is that true?
verdict: NO. SELinux is still Enforcing and `buildah bud --isolation chroot` runs RUN steps fine. What failed in August was buildah's DEFAULT isolation, not the machine. Docker is installed on UNAM but has no daemon, so `docker build` is genuinely unavailable there — which is a separate and real constraint, and the reason the final packaging still happened locally
status: MEASURED
date: 2026-09-02
measured_in: `buildah bud --isolation chroot` on a two-step Dockerfile with a RUN, on UNAM as uaq_user — "STEP 2/2: RUN echo … / HOLA-DESDE-UN-RUN / Successfully tagged". Zero GPU
---

# Correction: UNAM can build container images, with one flag

- **Applies when:** planning where a submission image gets built, or citing rung 04's
  packaging notes.

## What was believed, and what is true

`226246f` (2026-08-25) states building on UNAM "is impossible" and cites SELinux. That claim
sent a 34 GB weights build onto a box with 144 GB of root, where it peaked at ~145 GB, filled
the partition twice, and required a hand-split `Dockerfile.continue` to finish.

Re-tested 2026-09-02, unchanged machine, SELinux still `Enforcing`:

    buildah bud --isolation chroot -t unam-run-test .
    STEP 2/2: RUN echo HOLA-DESDE-UN-RUN > /x && cat /x
    HOLA-DESDE-UN-RUN
    Successfully tagged localhost/unam-run-test:latest

**The default isolation was the blocker, not the policy.** UNAM has 15 TB free and every merged
checkpoint already on disk; a build there needs no 17 GB download per model.

## The constraint that IS real

`docker` is installed on UNAM but `/var/run/docker.sock` does not exist — no daemon, so
`docker build` and `docker save` are unavailable. The submission tooling (`do_build.sh`,
`do_save.sh`, `do_test_run.sh`) is docker-based, so submission 06 was still packaged locally.
Anyone wanting to move packaging to UNAM must port that tooling to buildah/podman first, and
podman is not installed either.

## Local packaging constraints, since that is where it still happens

- **Delete the previous image before rebuilding.** The legacy builder (no buildx, no BuildKit
  on this box) streams the whole context and keeps the old weights layer. Not deleting first
  filled the root partition twice on 2026-09-02, exactly as it had in August.
- **`/mnt/datos` is NTFS**: `chmod` is inert, so the unprivileged container cannot write
  `test/output/`. Point `/output` at a POSIX filesystem for local runs.
- **`do_save.sh` writes the tar beside itself**, onto that same NTFS partition. Save to `/home`.

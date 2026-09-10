---
question: The repo went public for the final submission's "repository link" field, and it carried material we have no right to redistribute from a public URL. `git rm` leaves all of it downloadable from earlier commits. Rewrite the history, or accept the exposure?
verdict: REWRITE, with a verified full backup taken first. Four classes of material were purged from every commit on every branch — third-party papers, the organizers' documents, DUA-covered patient-derived frames, and a third party's SSH host/port/account. The cost is real and was paid knowingly: every clone made before 2026-09-09 no longer matches GitHub and must be re-cloned
status: SETTLED
date: 2026-09-09
measured_in: 23 PDFs + 10 frames + 62 sensitive path entries enumerated across `--all` history; backup verified by sha256 and by a test restore on the UNAM box
---

# Decision: the history was rewritten, and here is the backup

- **Applies when:** you have a clone older than 2026-09-09 and `git pull` refuses; or you are
  about to add any third-party file, dataset frame, or connection detail to this repository.

## What was purged, and why each one

| | why it could not stay |
|---|---|
| `literature/pdfs/` — 21 papers, 146 MB | third-party copyright. The catalogue (`literature/INDEX.md`) is the artifact and stays; every entry carries an arXiv/PMC link |
| the two challenge PDFs at the root | the organizers' documents, theirs to distribute |
| `_presentation_frames/` — 10 frames | **patient-derived surgical imagery under the ORENA DUA.** This is the one that actually mattered |
| the UNAM box's IP, SSH port, account and key fingerprint | a third party's infrastructure. No credential leaked — the account is key-only — but a public host+port+username triple is a target map |
| a teammate's personal email | redacted to their GitHub noreply address |

🔑 **`git rm` was not enough and that is the whole point of this note.** It removes a file from
the tip and leaves every earlier commit intact and downloadable. On a private repo that is fine.
On a public one it is a distinction without a difference.

## The backup, and how to restore

Taken before the rewrite, on the UNAM box:

    /mnt/storage/uaq_user/backups/repo-git-20260909-pre-force-push/
      repo-all-refs.bundle   every ref: main, all three branches, remote-tracking
      dotgit.tar.gz          the whole .git WITH reflog — reaches orphaned commits too
      MANIFEST.txt           the exact SHAs
      LEEME.md               the same explanation, in Spanish, next to the files

**Verified, not assumed:** sha256 identical to the local original on both files, and a test
clone from the bundle on that box recovered the merge commit, Rodrigo's full 18-commit line and
Yingyu's branch. Nothing was lost — it stopped being *published*.

⚠️ **Rodrigo has access to that box. Yingyu does not.** A copy on the pod's network volume is
still pending.

## If your clone is now out of sync

    git fetch origin && git reset --hard origin/main     # only if you have nothing unpushed

With unpushed work, do **not** reset: save your commits first (`git format-patch`, or a branch
kept aside), re-clone, then replay them.

## What this costs, stated plainly

A force-push rewrites every commit hash. Old clones, old branch pointers and any link to a
specific commit SHA from before 2026-09-09 are dead. That price was accepted because the
alternative was leaving patient-derived frames downloadable from a public URL.

## Links

- [[the-submission-dir-is-not-the-submission]] — the other place where "it is in the repo" and
  "it is actually shipped" came apart

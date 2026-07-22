# viewers — self-contained visual evidence

Open any file in a browser. Images are embedded (base64), so there is no server, no
relative paths, and nothing to install.

| viewer | what it shows | why it matters |
|---|---|---|
| `clip_viewer.html` | frames whose only annotated class is `Clip`, with the gold count, plus **temporal sequences** of adjacent frames where the count changes | Counting in this dataset is **counting clips**. The sequence tab shows the label swinging (e.g. `8 → 14 → 6` in 690 ms) between frames that look near-identical — the visual evidence behind the `number` ceiling in `context/CAMPAIGN_LOG.md` §10. |
| `transform_viewer.html` | 18 single image operators, original vs transformed, 1:1 | The candidate pool screened in rung 12d. |
| `combo_viewer.html` | 14 chained pipelines | The combinations that produced the five selected transforms — and the edge-family behaviour later retracted as a MAX artefact. |

⚠️ In `clip_viewer.html`, "only Clip" means **only clips were ever asked about in that frame**,
not that the image contains nothing else: `scene_inventory` is derived from the gold answers
(`CAMPAIGN_LOG.md` §9.3). A count that does not match what you see is the finding, not an error
on your part.

Regenerate: the builders live in the rung-12 experiment; frames come from the shared
`frames_cache` (gitignored, ~1.7 GB).

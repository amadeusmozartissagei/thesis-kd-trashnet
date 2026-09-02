---
name: feedback-docx-editing
description: How Hamza wants Word/.docx edits handled — one file, stop-and-ask if locked
metadata:
  type: feedback
---

When editing the thesis Word doc, Hamza wants:

1. **One single working file**, not a new copy per chapter. The canonical file is `C:\Users\Pratama\Downloads\smpro\sempro-hamza-1.docx` (now contains the rewritten BAB 3; he renamed the former `sempro-hamza-BAB3.docx` onto it).
2. **If you CANNOT edit the file (it's open/locked in Word), STOP the process and tell him** — do NOT silently work around it by making a copy. He will close Word so you can edit in place, then reopen to review.

**Why:** per-chapter copies fragment versions and cause confusion; he wants control over when the live file changes. **How to apply:** before a programmatic splice, if the lock file exists or the write fails, halt and ask him to close Word rather than routing around it. See [[thesis-writing-setup]].

**CRITICAL lock-file naming (learned 2026-06-29 the hard way):** Word's owner/lock file for `sempro-hamza-1.docx` is **`~$mpro-hamza-1.docx`** — Word DROPS THE FIRST 2 CHARS of the basename (names >8 chars). Checking for `~$sempro-hamza-1.docx` ALWAYS returns "not found" → false "Word closed", so I edited the live file 3× while Word was actually OPEN. Word doesn't hold an exclusive OS lock on the .docx (it reads to memory + holds only the `~$` owner file), so my writes SUCCEEDED silently underneath his open session — he kept seeing the STALE in-memory version and any Save from him would clobber my disk edits. **Always check `ls ~$mpro-hamza-1.docx` (or `ls ~$*.docx`), not the full name.** When the live doc shows old content after my edits: he's on a stale open session → tell him CLOSE WITHOUT SAVING, then reopen.

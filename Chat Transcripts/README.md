# Chat Transcripts

A historical archive of Cursor / AI-assistant conversations used to build this repo. Each file is the raw transcript of one session, stored verbatim except for redactions (see below).

## Naming convention

`chat_YYYYMMDD.md` — date is the date the chat happened (or the last day of a multi-day session). If multiple sessions on the same day: `chat_YYYYMMDD_<short-tag>.md`.

## Redaction policy

These files are committed to the public repo, so before adding a transcript:

1. **Scan for credentials** — API keys, OAuth tokens, passwords, signed URLs. The most common offenders match `sk-ant-`, `sk-`, `gh[ps]_`, `xoxb-`, `AIza`, `AKIA`. Replace any match with `[REDACTED]`.
2. **Scan for personal data** — email addresses other than the author's, phone numbers, addresses.
3. **Scan for non-public business info** — internal URLs, private repo names, customer names.

A useful pre-commit one-liner:

```bash
grep -nE 'sk-(ant-)?[a-zA-Z0-9_-]{20,}|gh[ps]_[a-zA-Z0-9]{20,}|xox[bp]-[a-zA-Z0-9-]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}' "Chat Transcripts/"*.md
```

If that prints anything, redact before committing.

## Why store transcripts at all?

Two reasons:
1. **Methodology trail.** A research repo that uses an AI assistant heavily benefits from a record of *which* design decisions came from the human, *which* came from the assistant, and *what alternatives were rejected and why*. The README + commit messages summarize the *what*; the transcripts preserve the *why*.
2. **Reproducibility of process.** Anyone wanting to rebuild this work — or audit how a particular module came to be designed the way it is — can read the relevant transcript and see the actual reasoning.

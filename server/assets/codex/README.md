# Codex's instructions

`prompt.md` is Codex's instruction text for models without their own catalog
entry. The server's Codex model catalog (`GET /v1/codex/models`,
`server/src/codex/`) gives it to Codex as each DeepSeek model's instructions,
because Codex refuses a catalog entry without instructions.

- Source: [openai/codex](https://github.com/openai/codex), tag `rust-v0.167.2`,
  file `codex-rs/models-manager/prompt.md`, unmodified.
- License: Apache License 2.0 (`LICENSE`), with Codex's `NOTICE`.

The text describes the tools of the Codex version it comes from. After a Codex
update, replace `prompt.md` with the same file from the matching tag and
rebuild the server:

```sh
gh api "repos/openai/codex/contents/codex-rs/models-manager/prompt.md?ref=rust-vX.Y.Z" \
  --jq .content | base64 -d > server/assets/codex/prompt.md
```

# Nodule: notes for Claude

## Working with the owner
- The owner directs and decides. They rarely edit code themselves. Be proactive:
  raise risks, suggestions and next steps, and check in at each milestone
  (see docs/roadmap.md).
- Ask before hard-to-reverse choices (taxonomy, model architecture, claim
  grades, stack). Record them as ADRs in docs/adr/.
- Literature first: ground each step in peer-reviewed sources read beyond the
  abstract. When a paper or image is bot-blocked or paywalled, ask the owner to
  obtain it (they download to `C:\Users\Lee\Downloads`); never settle for the
  abstract.

## Working efficiently (owner-approved)
- Keep sessions short: at each milestone update HANDOFF.md (local, gitignored,
  under ~80 lines) and suggest `/clear`.
- Start a session from CLAUDE.md and HANDOFF.md; open ADRs or the roadmap only
  when the task needs them.
- Grep first, then read only the lines needed; cap command output.
- Batch independent tool calls; use helper agents for bulky reading
  (papers, many-file surveys) so only their summary enters the session.

## Relation to agatesim
- agatesim (`C:\Users\Lee\Projects\agatesim`, github.com/lbillings-pixel/agatesim)
  is the first project and this one's library, pinned to a commit in
  pyproject.toml (`[tool.uv.sources]`). To use a newer agatesim, fix it there,
  push, and move the pin.
- Its ADR 0072 is this project's brief; cite its ADRs as "agatesim ADR N".
- Never copy agatesim code here; import it. Changes it needs are made in
  agatesim (with its own tests and rules).

## Environment (Windows 10, PowerShell 5.1)
- Tools: uv (Python 3.12), git, gh. If they aren't on PATH, refresh it:
  `$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")`
- GPU: NVIDIA RTX 4070 (12 GB). CI has no GPU. `uv sync --extra gpu` installs
  CuPy through agatesim; after that run everything as `uv run --no-sync ...`
  (a plain `uv run` re-syncs and removes CuPy).

## Commands
- `uv sync` / `uv run pytest` / `uv run ruff check` / `uv run ruff format` /
  `uv run pyright`
- Commit messages: write to a file in the scratchpad and use `git commit -F`.

## Rules
- **Graded claims (ADR 0002).** Every result is labelled verified, plausible or
  speculative. Procedural results are never evidence for a formation mechanism.
  Where sources disagree, offer the alternatives as options.
- Keep docs/science claims within each source's `checked` level in
  references/catalogue.toml. Cite as `[@id]`; add new sources to the catalogue.
- Label every model parameter `physical` or `effective`, and every coupling
  between processes with its source or as an assumption.
- Every new criterion needs a negative control: a result lacking the form that
  the criterion must reject.
- Anything random must be resolution-independent (spawned RNG streams; noise in
  world coordinates), so previews match full runs.
- Measure time and memory of anything new; flag actions over ~30 s or 2 GB
  before running them. Long work reports progress through `agatesim.progress`.
- The owner's specimen photos live in agatesim's `specimens/` (gitignored) and
  never enter this repository, a commit, an artifact or any external service.
  Anything derived from them is written without metadata.
- Commits use the no-reply email already set in git config. Never commit
  secrets or personal email addresses. Commit messages carry no
  `Claude-Session:` line; keep the `Co-Authored-By:` line.
- Ask before pushes of new work, merges, downloads, deletions, publishing, or
  anything leaving the PC.

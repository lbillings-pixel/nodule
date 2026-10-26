# Nodule

**Nodule** explores how the processes that form agate interact inside a single
nodule or seam: wall banding, settled layers, dendrites, crystals and colour,
growing together from one shared chemistry instead of one at a time. It asks
which combinations of conditions give the mixtures of forms that real
specimens show.

> **Status.** Founded 2026-10-26. No models yet: the first steps are a
> taxonomy of agate forms and an architecture decision, then a small 2D
> trial. See the [roadmap](docs/roadmap.md) and the
> [founding decision](docs/adr/0002-founding-nodule.md).

## Where it comes from

Nodule follows [agatesim](https://github.com/lbillings-pixel/agatesim), which
reproduces thirteen classic agate forms one at a time and asks which single
mechanism can make each. Real specimens mix several forms, and the papers
behind agatesim disagree on what drives banding, what the silica starts as,
and what twists the fibres. agatesim's
[ADR 0072](https://github.com/lbillings-pixel/agatesim/blob/main/docs/adr/0072-reframing-a-coupled-follow-on-project.md)
sets out the reasons for a follow-on project and what it should do
differently.

## How claims are graded

Every result says how far it is earned:

- **verified:** passes stated criteria, with every process taken from the
  literature and nothing judged imposed;
- **plausible:** every process has a source, but the way they're coupled is
  assumed and stated;
- **speculative:** some process or coupling has no source.

Where the literature disagrees, Nodule offers the alternatives rather than
picking one. These are explorations within models, not proof of how agate
forms.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync            # create the environment (installs agatesim as a library)
uv run pytest      # tests
uv run ruff check; uv run ruff format; uv run pyright
```

## Project layout

```
src/nodule/        the package (empty until the architecture ADR)
tests/             pytest suite
docs/adr/          architecture decision records
docs/roadmap.md    milestones
references/        literature catalogue with checking levels and licences
```

## License

[MIT](LICENSE). Literature keeps its own licences; see
[references/](references/README.md).

# Nodule

**Nodule** simulates agate nodules as heterogeneous 3D objects: exterior
shape, banding, inclusions, cavities, fractures, colour, optics, texture and
composition, organised across scales. It generates complete 3D nodules, cuts
through them and rendered images. Its goals, in order:

1. **image synthesis:** images of cut and whole agates that look real;
2. **realistic specimens:** 3D nodules whose exteriors, cuts and measured
   statistics match real ones;
3. **hypothesis testing:** how formation processes interact within one
   nodule or seam, and which conditions give the mixtures of forms that real
   specimens show.

It describes what is seen before trying to explain it: things that look
alike need not share an origin.

> **Status.** Founded 2026-10-26. No models yet: the first steps are a
> characterization schema (ten domains), then the 3D representation and one
> simple nodule built end to end. See the [scope brief](docs/scope.md), the
> [roadmap](docs/roadmap.md), the
> [founding decision](docs/adr/0002-founding-nodule.md) and the
> [scope decision](docs/adr/0003-scope-a-3d-specimen-simulator.md).

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

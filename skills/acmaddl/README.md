# acmaddl agent skill

An [Agent Skill](https://agentskills.io/) that teaches AI coding agents how to use the `acmadDL` package: `fetch()`/`assemble()` usage, the full product catalog, canonical data conventions, credentials, caching, and troubleshooting.

A skill is just a directory of Markdown and example files — no code runs from it. The agent reads `SKILL.md` when a task looks relevant (matching on the `name`/`description` frontmatter) and pulls in `references/` and `examples/` files only as needed, so the skill costs almost nothing until it's actually used.

## Layout

```
acmaddl/
├── SKILL.md                      # entry point: frontmatter + core instructions
├── references/
│   ├── api.md                    # complete API reference
│   ├── products.md               # every product id, adapter, credentials
│   ├── data-conventions.md       # normalization pipeline, units, dims
│   ├── plotting.md               # quick-look maps, facets, time series
│   └── troubleshooting.md        # credentials setup, errors, cache issues
└── examples/                     # runnable scripts
```

## Loading it

**Claude Code (this repo):** copy or symlink the skill into the project's `.claude/skills/` directory, which Claude Code auto-discovers:

```bash
mkdir -p .claude/skills
ln -s ../../skills/acmaddl .claude/skills/acmaddl
```

**Claude Code (all your projects):** put it in your personal skills directory instead:

```bash
ln -s "$(pwd)/skills/acmaddl" ~/.claude/skills/acmaddl
```

Then Claude uses it automatically when a task involves acmaddl, or invoke it explicitly with `/acmaddl`.

**Claude Agent SDK:** point the SDK at a directory containing the skill (e.g. via the `settingSources`/skills configuration) or copy it into the workspace's `.claude/skills/`.

**Claude API:** upload the directory via the Skills API (`/v1/skills`) and attach it to a container-use request; see the Anthropic docs for the current mechanism.

**Codex, OpenCode, and other agent coding frameworks:** the format (SKILL.md + frontmatter) is harness-agnostic by design, and any agentskills.io-compatible runtime can consume it — copy the `acmaddl/` directory into wherever that framework discovers skills (e.g. Codex and OpenCode both read `AGENTS.md`-style project instructions and can be pointed at skill directories). For any agent without native skill support, simply add an instruction to its project config (`AGENTS.md`, system prompt, etc.) to read `skills/acmaddl/SKILL.md` before working with acmaddl.

**Humans:** the same files work as documentation — start with `SKILL.md`.

## Validating

```bash
skills-ref validate skills/acmaddl   # from https://github.com/agentskills/agentskills
```

## Maintenance

When the public API, catalog, or conventions change, update the matching reference file (and `SKILL.md` if a core behavior changed). Everything in here was extracted from the source at the time of writing — treat drift as a bug.

# Conventional Commits 1.0.0

Source: https://www.conventionalcommits.org/en/v1.0.0/#specification

A lightweight convention on top of commit messages that produces an explicit,
machine-readable history — enabling automated changelogs and semantic-version
bumps.

## Structure

```
<type>[optional scope][optional !]: <description>

[optional body]

[optional footer(s)]
```

## Examples

```
feat(lang): add Polish language
```

```
fix: prevent racing of requests

Introduce a request id and a reference to latest request. Dismiss
responses other than from latest request.

Refs: #123
```

```
feat(api)!: send an email to the customer when a product is shipped
```

```
feat: allow provided config object to extend other configs

BREAKING CHANGE: `extends` key in config file is now used for extending
other config files
```

```
docs: correct spelling of CHANGELOG
```

## Types

- `feat` — a new feature (correlates with a MINOR SemVer bump).
- `fix` — a bug fix (correlates with a PATCH SemVer bump).
- `docs` — documentation only.
- `style` — formatting, whitespace, semicolons; no code-behavior change.
- `refactor` — code change that neither fixes a bug nor adds a feature.
- `perf` — a change that improves performance.
- `test` — adding or correcting tests.
- `build` — build system or external dependencies (e.g. npm, cargo).
- `ci` — CI configuration and scripts.
- `chore` — other changes that don't modify src or test files.

Types other than `feat` and `fix` are allowed and don't affect the version
unless they carry a breaking change.

## Rules (normative)

1. Commits MUST be prefixed with a type, followed by the OPTIONAL scope,
   OPTIONAL `!`, and REQUIRED terminal colon and space.
2. `feat` MUST be used when a commit adds a new feature.
3. `fix` MUST be used when a commit represents a bug fix.
4. A scope MAY follow the type; it MUST be a noun describing a section of the
   codebase, surrounded by parentheses, e.g. `fix(parser):`.
5. A description MUST immediately follow the colon and space.
6. A longer body MAY follow, beginning one blank line after the description.
7. A body is free-form and MAY be multiple newline-separated paragraphs.
8. One or more footers MAY be provided one blank line after the body.
9. A footer's token MUST use `-` in place of whitespace, e.g. `Acked-by`
   (exception: `BREAKING CHANGE`). A footer is `token: value` or `token #value`.
10. A footer's value MAY contain spaces and newlines; parsing terminates at the
    next valid footer token/separator pair.
11. Breaking changes MUST be indicated in the type/scope prefix, or as a footer.
12. As a footer, a breaking change MUST be `BREAKING CHANGE:` followed by a
    description.
13. In the prefix, a breaking change MUST be indicated by `!` immediately before
    the `:`. When `!` is used, `BREAKING CHANGE:` in the footer MAY be omitted,
    and the description then states the breaking change.
14. Types other than `feat` and `fix` MAY be used.
15. Units of information MUST NOT be treated as case-sensitive, except
    `BREAKING CHANGE`, which MUST be uppercase.
16. `BREAKING-CHANGE` MUST be synonymous with `BREAKING CHANGE` as a footer token.

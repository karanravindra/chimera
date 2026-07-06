---
name: commit
description: Commit changes to a git repository.
disable-model-invocation: true
---

# commit

Create a git commit using the [Conventional Commits](references/conventional-commit.md)
format. Read that reference for the full spec, type list, and breaking-change rules.

## Workflow

1. **Inspect the state.** Run these in parallel to understand what you're committing:
   - `git status` — what's staged, unstaged, and untracked.
   - `git diff HEAD` — the full change (staged + unstaged).
   - `git log --oneline -10` — match the repo's existing message style.

2. **Decide the scope of the commit.** If nothing is staged, stage the relevant
   files with `git add`. Only commit what belongs together — if the working tree
   holds unrelated changes, stage a coherent subset rather than everything. Never
   add secrets, credentials, or large build artifacts.

3. **Write the message.** Format:

   ```
   <type>(<optional scope>): <description>

   <optional body explaining what and why>

   <optional footers>
   ```

   - Pick the `type` from the change's intent, not its file paths (a bug fix in a
     test file is still `fix`, or `test` if it only adds test coverage).
   - `description`: imperative mood, lowercase, no trailing period, ≤ ~72 chars.
   - Add a body when the *why* isn't obvious from the description.
   - Mark breaking changes with `!` before the colon and/or a `BREAKING CHANGE:`
     footer.

4. **Commit.** Pass the message via multiple `-m` flags or a heredoc so the body
   and footers are preserved:

   ```bash
   git commit -m "feat(parser): add support for nested arrays" \
              -m "Handles arbitrarily deep nesting by recursing over tokens."
   ```

5. **Confirm.** Run `git status` after committing to verify the result and report
   the commit hash and subject line back to the user.

## Rules

- Commit only when the user asks. If on the default branch and the change is
  substantial, offer to branch first.
- Do not push unless the user asks.
- Do not amend or rebase existing commits unless explicitly requested.
- Never include tooling attribution or co-author trailers unless the user wants them.
- Keep each commit atomic: one logical change per commit.

# AGENTS.md

## Scope
**In scope**
Assist with Python + marimo development for the systemique dimensionning application
Focus on code maintenance, bug fixes, and incremental improvements to existing functionality

**Out of scope**
- Architecture changes
- New features without approval
- Algorithm modifications

## Rules
- Avoid using the function `copy()` in your marimo file
- Limit the number of variables to the strictly essential ones
- Do not write docstrings under the functions
- By default, set the parameter `hide_code` in `@app.cell` to True -> `@app.cell(hide_code=True)`
- Write all markdown text inside the marimo notebook (and only inside the marimo notebook) in French. For the discussions via the terminal interface, stick to English. 

## Workflow
- Small diffs
- After each change in the Marimo notebook, check if there are errors with the tool `get_notebook_errors` from the marimo MCP
- When editing Marimo notebooks, always run `uvx marimo check`on the file and fix all issues that you find
- After each edit, perform a git commit and after a big coherent sequence of code edits, perform a git commit and git push

## Style

## Communication
- Ask when unsure
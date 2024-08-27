\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Command-line interface (CLI) {#tutorial_cli}

\tableofcontents

SpECTRE has a command-line interface (CLI) that allows you to run executables,
work with output data, and generate plots and visualizations. To get started,
compile the `cli` target in your build directory. Then run:

```sh
./bin/spectre --help
```

All available commands are listed in the [Python documentation](py/cli.html).

## Autocompletion

The CLI supports autocompletion for Bash, Zsh, and Fish. Depending on your
shell, source the corresponding file:

```sh
# In the build directory:
# - Bash:
. ./bin/python/shell-completion.bash
# - Fish:
cp ./bin/python/shell-completion.fish ~/.config/fish/completions/spectre.fish
# - Zsh:
. ./bin/python/shell-completion.zsh
```

You may want to source the shell completion script on startup, e.g., in your
`.bashrc` or `.zshrc` file.

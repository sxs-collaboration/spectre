\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Code development quick-start with Docker and Visual Studio Code {#dev_guide_quick_start_docker_vscode}

This page describes how to get started developing SpECTRE on Mac, Linux, or
Windows using [Docker](https://docker.com) and the [Visual Studio
Code](https://code.visualstudio.com) editor. This is a particularly quick way to
get up and running with SpECTRE code development, though of course not the only
way. If you prefer setting up your development environment differently, we
suggest you read the \ref installation "Installation" page. If you would like to
jump right into a working development environment, read on!

## Fork the SpECTRE repository on GitHub

The SpECTRE code lives on GitHub in the
[sxs-collaboration/spectre](https://github.com/sxs-collaboration/spectre)
repository. Developers work on their own copy (or
["fork"](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-forks))
of the repository and contribute back to
[sxs-collaboration/spectre](https://github.com/sxs-collaboration/spectre)
through [pull requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).
Fork the repository to your own account:

- [Fork sxs-collaboration/spectre on GitHub](https://github.com/sxs-collaboration/spectre/fork)

## Clone the SpECTRE repository to your computer

To work on SpECTRE code you will need a local copy (or "clone") of the
repository on your computer. We use SSH to communicate with GitHub, so you need
to set up your SSH keys first. Follow GitHub's instructions to generate an SSH
key and add it to your GitHub account:

- [Generating an SSH key and adding to to GitHub](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

Now you can download the repository via SSH. Navigate to **your fork** of the
SpECTRE repository (i.e. the repository at the URL
https://github.com/YOURNAME/spectre). Follow GitHub's instructions to clone the
repository to your computer, selecting the SSH option:

- [Cloning a repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)

## Enable the development environment in the repository

The development environment is included in the repository, but not enabled by
default. To enable it, copy or symlink the directory
`support/DevEnvironments/.devcontainer` to the repository root. This is easiest
done with the command line. Navigate into the repository that you just cloned to
your computer:

```
cd spectre
```

Now symlink the development environment:

```
ln -s support/DevEnvironments/.devcontainer .devcontainer
```

## Install Docker

On your computer you will need Docker to run the containerized development
environment. Install and start Docker:

- [Get Docker](https://docs.docker.com/get-docker/)

If you're new to Docker, you can read through Docker's [Getting
started](https://docs.docker.com/get-started/) documentation to learn about
their basic concepts. We will use Docker to download and jump into a prebuilt
development environment that has everything installed to compile and run
SpECTRE.

## Install Visual Studio Code

We will use the Visual Studio Code editor to jump into the containerized
development environment, edit code, compile executables and run them. Download
and install Visual Studio Code:

- [Get Visual Studio Code](https://code.visualstudio.com)

Microsoft maintains [extensive
documentation](https://code.visualstudio.com/docs) for Visual Studio Code that
you can read to get started using the editor. We recommend you take a look
through the [Tips and
Tricks](https://code.visualstudio.com/docs/getstarted/tips-and-tricks) to get an
idea of the editor's features.

## Install the "Remote - Containers" extension

Visual Studio Code's "Remote - Containers" extension lets you run the editor in
a Docker container. Install the extension:

- [Install the "Remote - Containers"
  extension](vscode:extension/ms-vscode-remote.remote-containers)

## Open the SpECTRE repository in Visual Studio Code

Now open the SpECTRE repository that you have cloned to your computer in Visual
Studio Code. Depending on your operating system you can select `File > Open`,
type `Cmd+O` (macOS) or `Ctrl+O` (Linux or Windows), drag the repository folder
onto the Visual Studio Code icon or any other means to open the folder in Visual
Studio Code.

Now is also time to learn how to use the single most important tool in Visual
Studio Code, the command palette. Try it now: Hit `Cmd+P` (macOS) or `Ctrl+P`
(Linux or Windows) and start typing the name of any file, for example
`QuickStart.md`. Hit `Enter` to open the file. This is how you can quickly open
any file in the repository. Note that the search is fuzzy, so you can type any
few letters in the path to the file. In addition to opening files, you can hit
`Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Linux or Windows) and type the name of
any command that Visual Studio Code supports (or parts of it), for example
`Preferences: Open User Settings`.

## Reopen the SpECTRE repository in the development container

Open the command palette by hitting `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P`
(Linux or Windows) and run the command `Remote-Containers: Reopen in container`
by starting to type a few letters of this command and then hitting `Enter`.

Visual Studio Code will download and run the container and drop you into a fully
configured environment where you can proceed to compile and run SpECTRE.

If you are interested to learn more about this feature works you can
read through the [VS Code documentation on remote containers](https://code.visualstudio.com/docs/remote/containers)
and inspect the `.devcontainer/devcontainer.json` file that's included in the
SpECTRE repository.

## Configure, compile and run SpECTRE

With Visual Studio Code running in the development container you can now
configure, compile and run SpECTRE with no additional setup. Hit `Cmd+Shift+P`
(macOS) or `Ctrl+Shift+P` (Linux or Windows) to open the command palette and run
the command `CMake: Configure`. It will set up a build directory. You can open
Visual Studio Code's [integrated
terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) with
the keyboard shortcut ``Ctrl+` `` and navigate to the newly created build
directory to inspect it:

```
cd build-Default-Debug
```

For compiling and running SpECTRE you can either use Visual Studio Code's CMake
integration by looking up further commands in the command palette, or use the
terminal. We will be using the terminal from now on. Try compiling an
executable, for example:

```
make -j4 ExportCoordinates3D
```

Once the executable has compiled successfully you can try running it:

```
./bin/ExportCoordinates3D \
  --input-file $SPECTRE_HOME/tests/InputFiles/ExportCoordinates/Input3D.yaml
```

This executable produced a volume data file that we can visualize in ParaView.
Generate an XMF file from the volume data file that ParaView understands:

```
$SPECTRE_HOME/src/Visualization/Python/GenerateXdmf.py \
  --file-prefix ExportCoordinates3DVolume --subfile-name element_data \
  --output ExportCoordinates3DVolume
```

Since the build directory is shared with the host file system you can now open
ParaView on your computer and load the generated XMF file as described in the
\ref tutorial_visualization "visualization tutorial".

## Edit and contribute code to SpECTRE with VS Code

You are now ready to code! The other \ref dev_guide "dev guides" will teach you
how to write SpECTRE code. Return to the [Visual Studio Code
documentation](https://code.visualstudio.com/docs) to learn more about editing
code with Visual Studio Code.

In particular, the Visual Studio Code documentation can teach you how to use Git
to commit your code changes to your repository:

- [Git support in Visual Studio Code](https://code.visualstudio.com/docs/editor/versioncontrol#_git-support)

SpECTRE code development follows a pull-request model. You can learn more about
this process in our contributing guide:

- [Contributing to SpECTRE through pull requests](https://spectre-code.org/contributing_to_spectre.html#pull-requests)

Visual Studio Code can also help you create and review pull requests:

- [Working with pull requests in VS Code](https://code.visualstudio.com/docs/editor/github#_pull-requests)

## Interactively debug a SpECTRE test with VS Code

To track down an issue with your code it can be very useful to interactively
debug it. First, make sure you have written a test case where the issue occurs.
If you have not written a test yet, this is a great time to do it. Refer to the
\ref writing_unit_tests "Writing Unit Tests" dev guide to learn how to write
tests.

Now configure the `RunSingleTest` executable to run your particular test so you
don't have to repeatedly compile the extensive `RunTests` executable. Also refer
to the \ref writing_unit_tests "Writing Unit Tests" dev guide for this.

To launch the interactive debugger, hit `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P`
(Linux or Windows) to open the command palette, run the command `CMake: Set
Debug Target` and select `RunSingleTest`. Then run the command `CMake: Debug`.
The test executable will compile, run and stop on any breakpoints. Follow the
Visual Studio Code documentation to learn how to set breakpoints in your code
and inspect the state of your variables:

- [Debug actions in VS Code](https://code.visualstudio.com/docs/editor/debugging#_debug-actions)

## Real-time collaboration with Live Share

You can use the [Live
Share](https://docs.microsoft.com/en-us/visualstudio/liveshare/) extension to
work together with others and edit code simultaneously. Live Share can be very
useful to debug code together. Follow these instructions to share a link to your
current Visual Studio Code workspace:

- [Live Share Quickstart: Share your first project](https://docs.microsoft.com/en-us/visualstudio/liveshare/quickstart/share)


## Tips and tricks

- The [**GitLens extension**](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)
  is very useful to browse your repository. Select the "Source Control" icon and
  explore the various panels.

- When you build the **documentation** (e.g. with `make doc`), you can open it
  in a web server within VS Code:

  ```
  python3 -m http.server -d docs/html
  ```

  The web server launches on port 8000 by default, which is being forwarded
  outside the container, so you can just open http://127.0.0.1:8000 in your
  browser to view the documentation.

- Instead of working in the Docker container, you can use the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
  extension to **work on a remote machine** such as your local supercomputing
  cluster. Ask the cluster administrators or users for suggestions concerning
  installing and running SpECTRE on the particular supercomputer.

- You can work with **Jupyter notebooks in Visual Studio Code**. First, install
  Jupyter and all other Python packages you want to work with in the container:

  ```
  pip3 install jupyter matplotlib
  ```

  Any `.ipynb` notebook files you create or open will be displayed in VS Code's
  notebook editor. This is very useful for quickly plotting data from SpECTRE
  runs or using SpECTRE's Python bindings. Refer to the [VS Code documentation
  on Jupyter support](https://code.visualstudio.com/docs/python/jupyter-support)
  for more information.

- Docker can quickly use up a lot of disk space. From time to time you
  can "prune" unneeded images and containers to reclaim disk space:

  - [Prune unused Docker objects](https://docs.docker.com/config/pruning/)

  When pruning containers, make sure no data is deleted that you care about!

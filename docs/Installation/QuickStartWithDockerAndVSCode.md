\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Quick Start with Docker and Visual Studio Code {#quick_start_docker_vscode}

This page describes how to get started setting up an environment for developing
SpECTRE on Mac, Linux, or Windows using [Docker](https://docker.com) and the
[Visual Studio Code](https://code.visualstudio.com) editor. Both Docker and
Visual Studio Code work on Mac, Windows, and Linux.

Note: Visual Studio Code has different keyboard shortcuts on different
platforms. You can find summaries of the shortcuts for different platforms
here:

* https://code.visualstudio.com/shortcuts/keyboard-shortcuts-macos.pdf
* https://code.visualstudio.com/shortcuts/keyboard-shortcuts-linux.pdf
* https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf

## Install Visual Studio Code

[Download](https://code.visualstudio.com) and install Visual Studio Code from
the offical website. Microsoft provides binaries for Mac, Windows, and Linux.
The editor binaries are freeware.

Microsoft provides [installation
instructions](https://code.visualstudio.com/docs/setup/setup-overview), a
[FAQ](https://code.visualstudio.com/docs/supporting/faq), and [intro
videos](https://code.visualstudio.com/docs/getstarted/introvideos) to help you
get started. The FAQ includes instructions for disabling sending usage
statistics to Microsoft, if you prefer.

Alternatively, you can get the [source
code](https://github.com/microsoft/vscode) and compile it yourself.

Microsoft publishes an extensive [User's
Guide](https://code.visualstudio.com/docs/editor/codebasics) for Visual Studio
Code.


## Install recommended Visual Studio Code extensions

We recommend installing the following extensions. To install an extension, click
the "extensions" icon on the left (it looks like a loose square and 3 connected
squares), then search, then click Install. Microsoft provides instructions for
installing
[extensions](https://code.visualstudio.com/docs/editor/extension-gallery).

There are many similarly named extensions, so please double-check that you
install the correct extension by the correct author:

  * C/C++ (by Microsoft)
  * Python (by Microsoft)
  * Github Pull Requests (by GitHub)
  * Live Share (by Microsoft)
  * Format Modified (by gruntfuggly)
  * Docker (by Microsoft)
  * Remote-containers (by Microsoft)
  * Remote-ssh (by Microsoft)
  * Remote-ssh:editing config files (by Microsoft)
  * (Optional) LaTeX Workshop (by James Yu)
  * (Optional) Git Lens (git log, git blame, etc.)

These plugins should work with a reasonably new OS, but if your OS is more than
a few years old, you might need to update your OS for all of the
plugins to work.

Finally, we recommend, in Visual Studio Code, that you

  1. Press `Shift+Command+P` (`Shift+Control+P` on Linux and Windows) to open
  the command pallete. *This is the most important keyboard shortcut. You
  can search for and run all commands through the pallete.*
  2. Start typing the command `Shell Command: Install 'code' command in PATH`.
  3. Select that command and press enter.

After doing this, you can open files in Visual Studio Code from the terminal.
E.g., enter the command `code Hello.cpp` to open the C++ source file `Hello.cpp`
in Visual Studio Code.

## Configure ssh
If you haven't already set up ssh keys with your GitHub account, follow GitHub's
[instructions](https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
on creating an ssh key and [uploading your ssh
key](https://help.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account)
to your GitHub account.

***Mac only:*** You can configure ssh to save the passphrases for your keys in
the macOS Keychain. This will enable you to use Visual Studio Code's git
integration without having to enter any passphrases.

Add these lines to your `.ssh/config` file.

~~~~
Host *
   AddKeysToAgent yes
   UseKeychain yes
~~~~

## Install Docker

For Mac and Windows, install [Docker
Desktop](https://www.docker.com/products/docker-desktop).

For Linux, follow installation instructions for your OS to install Docker:
  * [CentOS](https://docs.docker.com/install/linux/docker-ce/centos/)
  * [Debian](https://docs.docker.com/install/linux/docker-ce/debian/)
  * [Fedora](https://docs.docker.com/install/linux/docker-ce/fedora/)
  * [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
  * [Other Linux](https://docs.docker.com/install/linux/docker-ce/binaries/)

If you're new to docker, you might wish to take a look at their Getting Started
documentation for [Mac](https://docs.docker.com/docker-for-mac/),
[Windows](https://docs.docker.com/docker-for-windows/), or
[Linux](https://docs.docker.com/get-started/).

Note that Docker can quickly use up a lot of disk space. From time to time, we
suggest running the following commands to reclaim unneeded space, ***after you
have made sure all docker containers you want to keep are running***:
  * `docker system df` shows how much disk space Docker is using
  * `docker system prune` will reclaim disk space by deleting all containers
    that are not running.  to reclaim disk space when using docker after a month
    or two)

Docker provides more [detailed
instructions](https://docs.docker.com/config/pruning/) for managing disk space.

## Clone the SpECTRE source code from GitHub

The SpECTRE source code is at (https://github.com/sxs-collaboration/spectre).
Fork
the repository to your GitHub account, then clone your fork. From the
command line, you can enter the command

~~~~
git clone git@github.com/YOUR_GITHUB_USER_NAME/spectre.git
~~~~

replacing `YOUR_GITHUB_USER_NAME` with your GitHub user name.

If you're new to git, GitHub has [extensive
documentation](https://help.github.com/en/github/using-git), including help with
forking a repo at
(https://help.github.com/en/github/getting-started-with-github/fork-a-repo).

## Create a SpECTRE docker container

From the command line (`/Applications/Terminal.app` on macOS, the Command Prompt
or Power Shell on Windows, or the terminal/shell in Linux), create a Docker
container for SpECTRE.

A container acts kind of like a virtual machine running a known Linux
environment. By using the SpECTRE docker container, you're free from worrying
about installing SpECTRE's prerequisites on your platform.

Here's an example command to create the container. Before running it, you'll
want to edit it.

~~~~
docker run -p 4444:4444 \
-v /Users/geoffrey/Codes/spectre/spectre-alt:\
/Users/geoffrey/Codes/spectre/spectre-alt:delegated \
-v /Users/geoffrey/Dropbox\ \(Personal\)/:/work/dropbox:delegated \
-v /Users/geoffrey/Documents:/work/documents:delegated \
--name spectre_vscode -i -t sxscollaboration/spectrebuildenv:latest \
/bin/bash -l
~~~~

You'll want to edit parts of this command:
  * `-p 4444:4444` forwards port 4444, in case you'd like to run jupyter
    notebooks from inside your docker container. If you don't plan to do this,
    you can omit this option.
  * In the option `-v` option containing
    `/Users/geoffrey/Codes/spectre/spectre-alt`,
    replace each occurence of
    `/Users/geoffrey/Codes/spectre/spectre-alt` with the path where you
    cloned your SpECTRE fork. ***If you are not using macOS, omit `:delegated`
    at the end of this option. This option improves file system performance.***
  * This command includes a few other `-v` options. Feel free to add a `-v`
    option to link any paths you might wish to have accessible from inside the
    Docker container. If you don't wish to make paths (besides your SpECTRE
    clone's path) accessible inside the container, you can omit these extra `-v`
    options. ***Omit `:delegate` unless you are running macOS.***
  * Feel free to change the name of the container, if you wish, by changing
    `spectre_vscode` in the option `--name spectre_vscode`.

After running this command, you'll get a command prompt inside the container,
running as root. Enter the following commands:

~~~~
# Create the directory where you will build spectre
mkdir /work/spectre-build-clang

# Install extra python packages in your container
apt-get install python3-pip
pip3 install autopep8 pylint jupyter notebook scipy numpy matplotlib
pip install autopep8 pylint jupyter notebook scipy numpy matplotlib

exit
~~~~

## Configure a Visual Studio Code workspace for your container

### Set up extensions in the container

Open Visual Studio Code. Click the docker icon on the left (looks like the
docker menubar icon while with boxes on top), and right-click your docker
container in the list at the top left, and start it (equivalent to `docker start
spectre_vscode` in terminal).

In Visual Studio Code, do `Shift+Command+P`, and run the command `Remote
containers: Attach to running container...` and choose the SpECTRE container
(named `spectre_vscode` in this guide, though you're free to change that
name).

Go to the "extensions" tab, and for each of the following extensions, install
them in the container. You must install these extensions in the container, even
though you already installed them in your local copy of Visual Studio Code.
  * C/C++
  * Format Modified
  * GitHub Pull Requests
  * Live Share
  * Python

You'll need to quit and re-open Visual Studio Code for these plugins to take
effect inside the container.

### Add the SpECTRE source directory to your workspace

After re-opening Visual Studio Code, run the `Remote containers: Attach to
running container...` again, if it doesn't automatically return you to a window
connected to your container. (It should say `Container
sxscollaboration/spectrebuildenv:latest` in the lower left corner if it is
connected to your Docker container.)

The first time you connect to your container, you'll need to open the SpECTRE
source code's root folder. Click the button to add a folder to the workspace, or
equivalently, un the command `Workspaces: Add Folder to Workspace`. Choose the
path to your spectre clone (`/Users/geoffrey/Codes/spectre/spectre-alt` in the
example command above).

### Add a keyboard shortcut for git-clang-format

The command `Format modified` runs git-clang-format. Choose
`Code->Preferences->Keyboard shortcuts` (macOS) or
open the command pallete (`Shift+Command+P` on Mac, `Shift+Control+P`
otherwise) and find the command to modify keyboard shortcuts
or keybindings. Then, search for `Format Modified: Format modified sections`,
click under keybinding", and press a keyboard shortcut
(we suggest `control+command+x`).

Then every time you are editing code, press this key command to format modified
sections (before committing) with `git clang format`.

### Add .vscode configuration files

If it doesn't already exist, create a directory named `.vscode` in the top level
directory of your clone of the SpECTRE code. One way to do this is clicking the
New Folder Icon above the list of files and folders on the left (the Explorer
tab); another is to open a terminal in the container (`Control + ~`, without
`Shift`) and use `mkdir`.

Add the following files to the `.vscode` folder:

* `.vscode/c_cpp_properties.json` This file configures the C/C++
intelligence features, which rely on a file called `compile_commands.json`
that will be created when you configure spectre.
~~~~
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**"
            ],
            "defines": [],
            "compilerPath": "/usr/local/bin/clang",
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "clang-x64",
            "compileCommands": "/work/spectre-build-clang/compile_commands.json"
        }
    ],
    "version": 4
}
~~~~

* `.vscode/tasks.json` This file sets up "tasks," essentially custom Visual
Studio Code commands that will configure, compile, and test SpECTRE.
***NOTE: update the path `/Users/geoffrey/Codes/spectre/spectre-alt` in
`command:` below to match your spectre clone's path.*** *Also change `-j4`
depending on the number of processors and amount of RAM available to
docker on your system.*
~~~~
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Configure spectre",
            "type": "shell",
            "command": "bash /work/cmake_spectre.sh"
        },
        {
            "label": "Build spectre",
            "type": "shell",
            "command": "cd /work/spectre-build-clang; make -j4",
            "group": {
                "kind": "build",
                "isDefault": true
            }
        },
        {
            "label": "Build spectre test-executables",
            "type": "shell",
            "command": "cd /work/spectre-build-clang; make test-executables",
            "group": "build"
        },
        {
            "label": "Test spectre",
            "type": "shell",
            "command": "cd /work/spectre-build-clang;ctest --output-on-failure",
            "group": {
                "kind": "test",
                "isDefault": true
            }
        }
    ]
}
~~~~

* `.vscode/launch.json`. This file configures for using Visual Studio Code's
debugger. Later on, you can change the `args:` option to choose a different
test to debug, or change the `program` and `args` tags for whatever
executable you would like to debug. This example would debug the unit test
`Unit.ApparentHorizons.StrahlkorperGr.RicciScalar`.
~~~~
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "/work/spectre-build-clang/bin/RunTests",
            "args": ["Unit.ApparentHorizons.StrahlkorperGr.RicciScalar"],
            "stopAtEntry": true,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        }
    ]
}
~~~~

Finally, add the shell script used to run `cmake`. Create a file
`/work/cmake_spectre.sh` and put the following into it, replacing the
last line with the path to your spectre clone.
~~~~
#!/bin/bash
cd /work/spectre-build-clang
cmake -D CMAKE_CXX_COMPILER=clang++ -D CMAKE_C_COMPILER=clang \
-D CMAKE_Fortran_COMPILER=gfortran-8 \
-D CHARM_ROOT=/work/charm/multicore-linux64-clang \
/Users/geoffrey/Codes/spectre/spectre-alt

~~~~

## Edit git configuration

Configure your git username and password. In the container (and, if you plan to
push commits to GitHub from outside your container, in a terminal not inside
your container), run these commands:

~~~~
git config --global user.name "Albert Einstein"
git config --global user.email "albert.einstein@gmail.com"
~~~~

Here, replace `Albert Einstein` with your name, and similarly replace
the email address `albert.einstein@gmail.com` with your email address.

Then, open `/root/.gitconfig` in your container, and add the following
options (or edit them if they exist):

~~~~
[core]
    editor = code --wait
~~~~

~~~~
[diff]
    tool = vscode
[difftool "vscode"]
    cmd = code --wait --diff $LOCAL $REMOTE
~~~~

You can add these options to your `.gitconfig` outside the container as well
(e.g., `$HOME/.gitconfig` on Mac and Linux), if you wish.

These options configure Visual Studio Code to be your text editor for
git.

## Configure SpECTRE

Open the command pallete (`Shift+Command+P` on Mac, `Shift+Control+P` otherwise)
and run the command `Tasks: Run Task`. Select `Configure spectre`. Select
`Continue without checking for warnings or errors` (the default option). This
will run cmake and configure spectre in `/work/spectre-build-clang` inside the
Docker container.

### Try out C++ intelligence features

To try out the C++ intelligence features, click the Explorer icon (at the top
left of the icon bar on the left side of the Visual Studio Code window). In the
list of files on the left, browse to src/ApparentHorizons/StrahlkorperGr. Scroll
down until you see `DataVector`. Mouse over `DataVector` and other symbols.

## Compile SpECTRE

Run `Tasks: Run Task` again, the same as when configuring, but choose the task
`Build spectre`. This will compile the SpECTRE libraries.

Run `Tasks: Run Task` again, the same as when compiling, but choose the task
`Build spectre test-executables`. This will compile the SpECTRE test
executables. *This step can take a long time. You can skip it, but then the
tests with `InputFiles` in the test name will not run when you run the tests.*

## Run the SpECTRE tests

Then, run `Tasks: Run Task` again, the same as before, but choose the task `Test
spectre`. This will run the SpECTRE tests. All tests should run and pass, unless
you chose to skip compiling the test executables. If you skip compiling the test
executables, `InputFiles` tests will not run.

## Real-time collaboration with Live Share

[Live Share](https://code.visualstudio.com/blogs/2017/11/15/live-share) lets you
and your collaborators work together to edit code simultaneously.

Click `Live Share` in the bottom blue bar (all of you should do this at least
once). The first time you do this, you are asked to sign in with GitHub. Do it.
Then, you get a link.

Share the link with anyone (e.g., via slack), and they can click it. When they
do, they'll be able to edit whatever tabs the you (the host) have open in real
time, similarly to real-time editing in google docs. This will let you and your
collaborators all work together to write code at the same time (with a google
hangout, teamspeak, discord, ... for voice if you aren't all in the same place).

## Tips and tricks

* **Changing git branches** Look at the blue bar at the bottom that says
  develop. You can click this to make new branches or check out different
  branches.

* **Git and GitHub integration** — Check out the "git tab" and "github pull
  requests" tab. The git tab (icon is 3 circles with curves connecting them)
  lists all modified files. The github pull requests tab lets you look at and
  check out any pull request. You'll see any review comments right there in your
  code, and you can make your changes in response right there in the same window
  as the comments.

* **Terminal** — `Control + ~` (don't press `Shift`) opens a terminal right in
  Visual Studio Code.

* **Linting with clang-tidy** — Use the terminal to `cd
  /work/spectre-build-clang`. Then, run `make clang-tidy FILE=/path/to/file.cpp`
  on the path of any cpp file in the spectre directory, and the clang-tidy
  linter will report any issues it might have.

* **Building the documentation** — Use the terminal to
  `cd /work/spectre-build-clang`. Then, do `make doc`.
  Copy the `docs/html` folder
  to somewhere accessible outside Docker, and open `docs/html/index.html` in
  your web browser.

* **Code completion** — When writing code, you can hit control+space to get
  suggestions. You can tab complete. Mouse over a symbol, like DataVector, to
  see its help text. Right-click and you can go to the declaration (in an hpp
  file) or definition (in a cpp file) to see the code where it's defined. The Go
  menu has a back command to go back after you do this.

* **Navigation** — `Control + G` asks for a line number in the current file,
  then goes to it. `Command + P` (macOS) or `Control + P` (Windows and Linux)
  lets you open to any file in spectre, by typing the first few letters of the
  file.

* **Editing files on remote servers over ssh** — The `remote-ssh` extension
  works just like `remote-container`, except instead of connecting to a running
  docker container, you can connect to a remote system, such as a cluster. Git
  integration and clang-format might not work without proper configuration,
  which is beyond the scope of this guide. But you can edit code there using
  Visual Studio Code instead of a terminal and emacs. Intellisense
  (autocomplete, etc.) should still work, but it might be slow, depending on
  your network connection.

## Interactively debug a SpECTRE test

Visual Studio Code includes graphical debugging. Press `Shift+Command+P` (Mac)
or `Shift+Control+P` (Windows/Linux) and run `Debugging: Start Debugging` in
Visual Studio Code. Then, wait for a little while as the debugger loads (this
can take a few minutes). You can now debug, stepping into and over lines of
code.

The executable and arguments that will be debugged are set in
`.vscode/launch.json`. To debug a specific test, use
`/work/spectre-build-clang/bin/RunTests` as the executable and the test name as
the argument.

You can click on any function in the Call Stack list on the left to go to it in
the source code.

You can inspect and edit local variables in the Variables list to the left. You
can drill down inside local variables, even objects like `Tensor<DataVector>`,
to see (and modify) their values.

The debug console shows the output of gdb. The integrated terminal shows the
program output.

You can set breakpoints by clicking near the line number in any source file.

Run `Debug: Stop Debugging` when you're done debugging.

Microsoft's documentation includes [information on
debugging](https://code.visualstudio.com/docs/editor/debugging).

## Running python in the container

When editing a python source file (i.e., a `*.py` file) in the spectre
docker container, select any code and press `Shift+Enter` to evaluate it.
You can use this to test python code as you write or edit it.

You can choose to evaluate it in the interactive python interpreter or in
the terminal. We recommend using the interactive python interpreter. The
setting can be found under `Settings` as
`Python:Data Science:Send Selection to Interactive Window`.

## Configure Visual Studio Code for running Jupyter notebooks on your system

We suggest installing [Anaconda 3](https://www.anaconda.com) to get a python 3
distribution with `scipy`, `numpy`, `matplotlib`, and Jupyter notebooks on
your system (i.e., not in the spectre container).

Download a jupyter notebook, such as one from
(https://github.com/catalog_tools/Examples). Open it in Visual Studio Code. As
long as the Python extension can find your Anaconda distribution
(the installer should offer to add it to your `PATH`), it can
natively run Jupyter notebooks.

**Mac only** — In the Finder, choose "Get Info...", and make Visual Studio Code
the default app for ipynb files. Push the "change all" button. You can now
double-click any jupyter notebook and run it in Visual Studio Code.

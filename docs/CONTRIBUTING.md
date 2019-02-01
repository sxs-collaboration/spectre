\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond


# Contributing to SpECTRE {#contributing_to_spectre}

\tableofcontents

# Contributing to SpECTRE {#contributing_to_spectre_2}

The following is a set of guidelines for contributing to SpECTRE,
which is hosted in the [Simulating eXtreme Spacetimes
Organization](https://github.com/sxs-collaboration) on GitHub.

## Code of Conduct

This project and everyone participating in it is governed by the \ref
code_of_conduct "SpECTRE Code of Conduct". By participating, you are
expected to uphold this code. Please report possible violations of the
code of conduct to conduct@spectre-code.org.

## What should I know before I get started? {#getting-started}

SpECTRE is being developed in support of our collaborative Simulating
eXtreme Spacetimes (SXS) research program into the multi-messenger
astrophysics of neutron star mergers, core-collapse supernovae, and
gamma-ray bursts.  As such, almost all of the current contributors to
SpECTRE are members of SXS institutions, and a large amount of
discussion about SpECTRE is done in internal SXS meetings.  If you are
a member of SXS and wish to get involved, please contact one of the
project leaders.

In the future, we hope that SpECTRE can be applied to problems across
discipline boundaries, and that it can be a true community code.  At
the present time, however, SpECTRE cannot yet solve realistic
problems, and broad overview documentation is in an early, incomplete
stage.  Therefore, if you are not a member of SXS, but are interested
in contributing to SpECTRE, we strongly encourage you to contact us at
questions@spectre-code.org to discuss possible contributions.

## How Can I Contribute? {#how-can-i-contribute}

### Reporting Bugs {#reporting-bugs}

This section guides you through submitting a bug report for
SpECTRE. Following these guidelines helps maintainers and the
community understand your report, reproduce the behavior, and find
related reports.

Before creating bug reports, please **perform a
[search](https://github.com/sxs-collaboration/spectre/issues)** to see
if the problem has already been reported. If it has **and the issue is
still open**, please add a comment to the existing issue instead of
opening a new one.

> **Note:** If you find a **Closed** issue that seems like it is the
> same thing that you're experiencing, please open a new issue and
> include a link to the original issue in the body of your new one.

#### How Do I Submit A (Good) Bug Report? {#submit-bug-report}

Bugs are tracked as [GitHub
issues](https://guides.github.com/features/issues/). When you are
creating a bug report, please **include as many details as
possible**. Please fill out the template completely.  The provided
information helps us resolve issues faster.

Explain the problem and include additional details to help maintainers
reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the
  problem.
* **Describe the exact steps which reproduce the problem** in as much
  detail as possible. For example, start by explaining how you started
  SpECTRE, e.g. the exact command you used in the terminal, or the
  contents of the batch job script you used.
* **Describe the behavior you observed after following the steps** and
  point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**

Provide more context by answering these questions:

* **Can you reproduce the problem in both debug and release mode?**
(this is controlled by the CMake flag `CMAKE_BUILD_TYPE`)
* **Did the problem start happening recently** (e.g. after updating to
  a new version/commit of SpECTRE) or was this always a problem?
* If the problem started happening recently, **can you reproduce the
  problem in an older version/commit?** What's the most recent
  version/commit in which the problem doesn't happen?
* **Can you reliably reproduce the issue?** If not, provide details
  about how often the problem happens and under which conditions it
  normally happens.
* **Can you reproduce the problem on another machine?**
* **Can you reproduce the problem in the docker container?** (see the
  \ref installation "Installation notes")

Include details about your configuration and environment:

* **Add as an attachment** (or add the contents of) the following:
  - The text output by SpECTRE (including any stack trace)
  - The input file(s)
  - $SPECTRE_BUILD_DIR/LibraryVersions.txt
* **What is the name and version of the OS you're using**?
* If possible (for SXS computers or HPC systems), a **path to a run
  directory** that is accessible by SpECTRE core developers.

### Suggesting Enhancements {#suggesting-enhancements}

This section guides you through submitting an enhancement suggestion
for SpECTRE, including completely new features and minor improvements
to existing functionality. Following these guidelines helps
maintainers and the community understand your suggestion and find
related suggestions.

Before creating enhancement suggestions, please **perform a
[search](https://github.com/sxs-collaboration/spectre/issues)** to see
if the enhancement has already been suggested. If it has, add a
comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Enhancement Suggestion? {#submit-enhancement}

Enhancement suggestions are tracked as [GitHub
issues](https://guides.github.com/features/issues/).  When you are
creating an enhancement suggestion, please **include as many details
as possible** as you fill in the template.

* **Use a clear and descriptive title** for the issue to identify the
  suggestion.
* **Provide a step-by-step description of the suggested enhancement**
  in as many details as possible.
* **Explain why this enhancement would be useful** to most SpECTRE users.

### Your First Code Contribution {#first-code-contribution}

Unsure where to begin contributing to SpECTRE? You can start by
looking through these `good first issue` and `help wanted` issues:

* [Good first issues][good-first-issue] - issues which should only
  require a few lines of code, and a test or two.
* [Help wanted issues][help-wanted] - issues which should be a bit
  more involved than `good first issue`s.

#### Local development {#local-development}

SpECTRE can be developed locally. For instructions on how to do this,
see the following sections in the [SpECTRE
documentation](https://spectre-code.org/):

* [Installing SpECTRE](installation.html)
* [Running Status Checks Locally](travis_guide.html#perform-checks-locally)

## Pull Requests {#pull-requests}

Code contributions to SpECTRE follow a [pull request
model](https://help.github.com/articles/about-pull-requests/)

The process described here has several goals:

- Maintain SpECTRE's code and documentation quality
- Reach science goals in a timely manner
- Fix problems that are important to users
- Engage the community in working toward the best possible code
- Enable a sustainable system for SpECTRE's maintainers to review
  contributions

Please follow these steps to have your contribution considered by the
maintainers:

1. Follow the \ref code_review_guide "code review guidelines", the
  \ref writing_unit_tests "guide to writing unit tests", and the \ref
  writing_good_dox "guide to writing documentation"
2. Follow all instructions in the pull request template. Reference
   related issues and pull requests.
3. After you submit your pull request, verify that all [status
   checks](https://help.github.com/articles/about-status-checks/) are
   passing

   > If a status check is failing, and you believe that the failure is
   > unrelated to your change, please leave a comment on the pull
   > request explaining why you believe the failure is unrelated. A
   > maintainer will re-run the status check for you. If we conclude
   > that the failure was a false positive, then we will open an issue
   > to track that problem with our status check suite.

While the prerequisites above must be satisfied prior to having your
pull request reviewed, the reviewers may ask you to complete
additional design work, tests, or other changes before your pull
request can be ultimately accepted.

## How SpECTRE pull request reviews are conducted {#pull-request-reviews}

> Note that these are guidelines and not rigid rules.

Please make your pull requests as small as reasonably possible, as
smaller pull requests are easier to review. In general, longer pull
requests take longer to review, with the time scaling exponentially
with the number of lines changed.  Therefore if your pull request is
too large, we may ask you to break it up into several smaller pull
requests.

> Below, days mean business days, so if the time period includes the
> weekend, add two days, and for major holidays add a day.
> Furthermore, most of us are academics, and we occasionally go to
> conferences which may lead to delays in the review process.  Also do
> not expect much to happen between December 20th and January 3rd.

Most pull requests submitted to SpECTRE will be reviewed in the
following manner:
- When creating the pull request, the author should add the
  appropriate labels.  If the pull request is not a `new design` or
  `in progress` (discussed below), the author may either assign two
  reviewers (if GitHub suggested any) or add the `reviewers wanted`
  label.
- Within two days, assigned reviewers should either confirm that they
  are able to review, or remove themselves while adding the `reviewers
  wanted` label.
- Anyone is welcome to self-assign themselves as a reviewer; when
  there are at least two reviewers, the `reviewers wanted` label
  should be removed.
- If there are not two reviewers after two days, a [SpECTRE core
  developer](#core-developers) will assign reviewers, removing the
  `reviewers wanted` label.  If for some reason, no reviewers have
  been assigned after three days, the author of the pull request
  should feel free to [ping the core
  developers](https://github.blog/2011-03-23-mention-somebody-they-re-notified/)
  (e.g `@sxs-collaboration/spectre-core-devs please assign reviewers`)
- Assigned reviewers should submit their review in as timely a manner
  as possible.
- Anyone can request changes within either the first two days of the
  pull request, or within a day after the initial reviews of the
  assigned reviewers.  After this period, only the assigned reviewers
  can request changes, unless someone believes the code is wrong.
  Non-reviewers are allowed to make comments, which the pull request
  author is encouraged to address.  Alternatively the author can
  create an issue with the suggested changes, assigned to themselves,
  which would then be addressed in a subsequent pull request by the
  author.
- If any requested change is unclear to the author, they should ping
  the reviewer and ask for clarification.  Authors and reviewers are
  encouraged to talk to one another (in person, via Google hangout, or
  some other verbal method if possible) to resolve any issues.
- Reviewers are encouraged to [ping others
  \@GITHUB_USERNAME](https://github.blog/2011-03-23-mention-somebody-they-re-notified/)
  to get opinions on code they are unsure about.
- It is permissible to have a group code review led by one of the
  reviewers. The reviewer should comment on who was present at the
  group review.
- If necessary, pull requests can also be discussed at one of the
  weekly SpECTRE meetings.
- If changes are requested, the author should fix all of them in one
  or more fixup commits (where the first line of the commit message
  should begin with `fixup`), and then ping the reviewers that the
  pull request is updated.  In addition the `updated` label can be
  added.
- Once a pull request is updated, the reviewers should either request
  further changes (removing the `updated` label) or tell the pull
  request author to squash their commits.
- Once all reviewers give the okay to squash, the author should
  [rebase on develop and then squash their
  commits](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)
  into one or more self-contained commits (such that the code will
  compile and pass all tests after each commit).  The squashed commits
  will need to be force pushed.
- Once the commits are squashed, the author should ping the reviewers.
- Once all reviewers have approved the pull request, someone should
  ping the `@sxs-collaboration/spectre-core-devs`.  If one of the
  original reviewers is a core developer, this is not necessary and
  the core developer can merge the pull request.
- One of the core developers will perform a final cursory review,
  requesting changes only for major problems, and commenting on other
  possible changes.
- The pull request author should address all final requested changes
  and may either also fix final suggested changes, or create an issue
  with the suggested changes, which will be addressed in a subsequent
  pull request by the author.
- The SpECTRE core developer will merge the pull request once all
  comments have been addressed, the code passes CI, and all pull
  requests the pull request depends on have been merged.

Critical bug fixes (i.e. the code is broken) can be merged after two
expedited reviews by SpECTRE core developers.  If necessary, an issue
can be created if further changes are desired.

Pull requests that are designated `new design` are expected to have a
longer review period, including discussions during weekly SpECTRE
meetings.  SpECTRE core developers will provide reasonable review
deadlines once the new design is finalized.

If you would like feedback on a pull request prior to it being ready
for formal review, please label it with `in progress` and ping
whomever you wish to get feedback from.  As long as the `in progress`
label remains, no one should review the pull request.


### Git Commit Message Guidelines {#git-commit-messages}

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* If needed, a blank second line followed by a more complete description

### SpECTRE core developers {#core-developers}

SpECTRE core developers are people who are very familiar with the
entire code, comfortable with modern C++, and willing to take the
responsibility of overseeing the code as a whole.
[Current SpECTRE core
developers](https://github.com/orgs/sxs-collaboration/teams/spectre-core-devs)
can be pinged on GitHub at `@sxs-collaboration/spectre-core-devs`.  It
is expected that as more contributors become familiar with SpECTRE,
additional people will be added to the list of core developers.


[good-first-issue]:https://github.com/sxs-collaboration/spectre/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22+sort%3Acomments-desc
[help-wanted]:https://github.com/sxs-collaboration/spectre/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+sort%3Acomments-desc


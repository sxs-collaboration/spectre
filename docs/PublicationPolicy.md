\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Publication policies {#publication_policies}

SpECTRE is an open-source code that is developed and maintained by the
[Simulating eXtreme Spacetimes (SXS) collaboration](https://black-holes.org). We
ask all authors of scientific publications that make use of SpECTRE's code or
data to follow the citation guidelines laid out in the section \ref citing. An
additional set of guidelines, laid out in the section \ref
sxs_publication_policies applies to publications by members of the SXS
collaboration.

## Citing SpECTRE {#citing}

Please cite SpECTRE in any publications that make use of its code or data. Cite
the latest version that you use in your publication. The DOI for this version
is:

- DOI: [\spectredoi](https://doi.org/\spectredoi)

You can cite this BibTeX entry in your publication:

- [Find BibTeX entry for this version on Zenodo](https://zenodo.org/record/\spectrezenodoid/export/hx)

To aid reproducibility of your scientific results with SpECTRE, we recommend you
keep track of the version(s) you used and report this information in your
publication. We also recommend you supply the YAML input files and, if
appropriate, any additional C++ code you wrote to compile SpECTRE executables as
supplemental material to the publication.

## Policies that apply to SXS publications {#sxs_publication_policies}

These policies apply to all "science papers" by members of the SXS collaboration
that make use of SpECTRE’s code or data, as defined by the
[SXS policies](https://github.com/sxs-collaboration/WelcomeToSXS/blob/master/SxsPolicies.md).

Our publication policies have the goal that contributions to SpECTRE gain value
in academia. For SpECTRE to be successful, it must be worth a junior
researcher's time to contribute to the code. Contributions are already partially
acknowledged through authorship on the DOI, as laid out in the `Metadata.yaml`
file. However, authorship on the DOI alone doesn’t hold enough value in academia
to make code contributions worthwhile that have no immediate science paper
associated with them, but that enable science papers down the line. Therefore,
our policies have the purpose to grant authorship rights on science papers to
the developers of code that enabled them.

### Guidelines for SXS paper authors

- If you're a member of the SXS collaboration and use SpECTRE for your paper,
  you should probably include a list of SpECTRE developers as co-authors per the
  [SXS policies](https://github.com/sxs-collaboration/WelcomeToSXS/blob/master/SxsPolicies.md).
- To obtain the list of co-authors, reach out to any of the people listed in the
  [SpECTRE core developers](https://github.com/orgs/sxs-collaboration/teams/spectre-core-devs/members)
  team on GitHub. Use a communication channel of your choice, here are a few
  possibilities:

  - Send a message on Slack.
  - Knock on their office door.
  - Send an email to one of the core developers, or to
    [spectre-core-devs@black-holes.org](mailto:spectre-core-devs@black-holes.org).
    Possible wording of the email:

    > Dear SpECTRE core devs,
    >
    > I am / we are preparing a paper on [title or topic]. I am / we are using
    > SpECTRE in this way:
    >
    > - [Describe your use of SpECTRE briefly, e.g. as a bulleted list. Mention
    >   which parts of SpECTRE you are using in particular, e.g. the initial
    >   data solver, the evolution scheme, the wave extraction, etc.]
    >
    > Please respond with the list of co-authors to include in the paper.
    >
    > Best regards,
    > [you]

  The core developers will discuss among themselves, following the guidelines
  listed below, and will get back to you with a list of authors.

## Guidelines to decide who has authorship rights on SXS science papers that use SpECTRE

- A contribution to SpECTRE on the order of ~500 lines of code or more should
  earn you authorship rights on the first science paper that uses the feature.
  Major contributions, such as writing a full evolution system, earns you
  authorship rights on the first three science papers that use the feature.
  "Using the feature" means the code contributes to the result of the paper,
  either obviously like a new coordinate map in the domain, or in a more subtle
  way like an optimization that made the simulation run faster.
- Core developers earn authorship rights to all science papers that use SpECTRE.
- In case of controversy, contact any person who you feel comfortable raising
  your concern with, such as one of the core developers, a member of the
  [executive committee](https://github.com/sxs-collaboration/WelcomeToSXS/blob/master/SxsPolicies.md#executive-committee),
  or the [ombudsperson](https://github.com/sxs-collaboration/WelcomeToSXS/blob/master/SxsPolicies.md#ombudsperson).

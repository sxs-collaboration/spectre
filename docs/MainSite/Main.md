\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
\mainpage Documentation Overview

<div class="toc">
Table of Contents
<ul>
<li class="level1"><a href="#intro_sec">Introduction </a></li>
<li class="level1"><a href="installation.html">Installation </a></li>
<li class="level1"><a href="installation_on_clusters.html">Installation On Clusters</a></li>
<li class="level1"><a href="doc_coverage/index.html">Documentation Coverage </a></li>
<li class="level1"><a href="unit-test-coverage/index.html">Unit Test Coverage </a></li>
</ul>
</div>

Introduction {#intro_sec}
==============================================================================

This is the introduction.

We need an introduction because we don't have one...

Navigating the Documentation {#navigate_documentation_sec}
-----

The SpECTRE documentation is organized into tutorials, developer guides, groups
of related code, namespaces, and by files for easier navigation. For someone
looking to run simulations with SpECTRE we recommend starting with the
\ref tutorials "tutorials", which are located under the `Documentation` menu
item at the top. The tutorials are designed to get users up and running with a
simulation, and analyzing and visualizing the output. For people looking to
contribute to SpECTRE there are tutorials under the \ref dev_guide "Dev Guide"
part of the `Documentation` menu. The dev guide includes things such as
\ref code_review_guide "code review guidelines", how to \ref writing_unit_tests
"write unit tests", how to \ref writing_good_dox "write documentation", as well
as some information about C++ and the philosophy behind SpECTRE development. The
[Reference](modules.html) sections contains groups (managed through doxygen
groups) of related code. For example, there is a
group for all the data structures we use, a group for utility functions and
classes, as well as groups for coordinate maps, domain creation, and many
others. The [Reference](modules.html) is designed to help developers discover
existing functionality so that things are not re-implemented several times. You
can also get an overview of the code base by namespace by visiting the
[Namespaces](namespaces.html) section under the `Reference` menu item. Finally,
it is also possible to browse the repository by files under the
[Files](files.html) menu item, though it is recommended that in that case you
browse the [GitHub repository](https://github.com/sxs-collaboration/spectre)
directly.

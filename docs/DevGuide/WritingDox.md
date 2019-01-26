\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing Good Documentation {#writing_good_dox}

\tableofcontents

# Tutorials, Instructions, and Dev Guide {#writing_dox_writing_help}

All non-code documentation such as tutorials, installation instructions, and
the developer guide is written inside Markdown files such as this one. These
files must be placed inside `docs/Tutorials`, `docs/MainSite`, or
`docs/DevGuide` according to what they document. Each Markdown file must start
with the following line

``` markdown
# The title {#the_tag}
```

where `The title` is replaced with your desired title, and `the_tag` with the
tag you want to use to reference the Markdown file and documentation. Each
main heading of the file starts with a single octothorpe
and can have a tag. For example,

``` markdown
# My Section {#file_name_my_section}
```

While the `file_name` portion is not necessary, it is useful for reducing the
likelihood of reference collisions. You can add a table of contents using the
Doxygen command <code>\\tableofcontents</code>. All sections, subsections,
subsubsections, etc. are shown in the table of contents if they have a tag,
if not they do not appear.

# C++ Documentation {#writing_dox_cpp_dox_help}

We require you to add Doxygen documentation for any of the following
you may have added to the public interface in any hpp file:

+ classes
+ structs
+ enums
+ functions
+ namespaces
+ macros
+ type aliases

Documentation begins on the line immediately above the declaration with
either a triple slash `///` or a `/*!`.

Examples:
\verbatim
/// \brief A brief description of the object to be documented
///
/// Doxygen comments can be made
/// using triple slashes...
class ExampleClass{
  ... rest of code
}
\endverbatim

\verbatim
/*!
 * \brief A brief description of the object to be documented
 *
 * Doxygen comments can also be made
 * using the "slash-star, star-slash" pattern, in this way.
 */
class ExampleClass{
  ... rest of code
}
\endverbatim

Build your documentation by running `make doc` in your build directory.
You can then view it by opening `BUILD_DIR/docs/html/index.html` in a
browser and using the file navigator in Doxygen to locate your file.

In addition to providing a file directory of all the source files in SpECTRE,
Doxygen also conveniently provides two additional organizations of the files,
Topics and Namespaces. To ensure that your documentation is easily found from
within Doxygen, we recommend that you add any new objects to Topics and any
new namespaces to Namespaces.

## Add your object to an existing Topic:

Within a Doxygen comment, use the Doxygen keyword
\verbatim
\ingroup
\endverbatim

followed by the name of the Topic (you can find the list of existing Topics in
`docs/GroupDefs.hpp`).

## Add a new Topic:

Within `docs/GroupDefs.hpp`, add the name of your Topic (which follows the
naming convention "YourNameForTopic" followed by "Group") among the rest,
taking care to keep the list of Topics in alphabetical order.

## Add a new namespace:

In the hpp file in which the namespace is declared for the first time,
add a Doxygen comment to the line directly above the namespace. Subsequent
files which use this namespace will not need a Doxygen comment.

We also strongly encourage you to:

## Put any mathematical expressions in your documentation into LaTeX form:

Within a Doxygen comment, begin and conclude your expression with

\verbatim
\f$
\endverbatim

Example:
\verbatim
\\\ We define \f$ \xi : = \eta^2 \f$
\endverbatim

Note that if this expression spans more than one line,
your Doxygen comment must of the "slash-star, star-slash" form shown in the
previous section. One can also use (within a Doxygen comment) the form

\verbatim
\f[ expression \f]
\endverbatim

to put the expression on its own line. We also encourage you to use the latex
env `align` for formatting these multiple-line equations.

## Cite publications in your documentation {#writing_dox_citations}

When you refer to publications or books in the documentation, add a
corresponding entry to `docs/References.bib`. Follow these guidelines when
editing `docs/References.bib`:

- Ideally, find the publication on [INSPIRE HEP](https://inspirehep.net), copy
its BibTeX entry, remove the colon `:` from its key and add the `url` field (see
below). We remove the colon because it can create problems in HTML related to
its function as a CSS selector.
- For publications that are not listed on [INSPIRE HEP](https://inspirehep.net),
make sure to format the new entry's key in the same style, i.e.
`(<Author>[a-zA-Z]+)(<Year>[0-9]{4})(<ID>[a-z]*)`. Good keys are, for
instance, `Einstein1915` or `LVC2016a`. For books you may omit the year.
- Sort the list of BibTeX entries in the file alphabetically by their keys.
- Provide open access or preprint information whenever possible. For
publications available on [arXiv](https://arxiv.org), for instance, add the
following fields (filling in the correct values):
```
archivePrefix = {arXiv},
eprint = {1609.00098},
primaryClass = {astro-ph.HE},
```
- Provide the `url` field whenever possible. If a DOI has been issued by the
publisher, use `https://doi.org/<doi>` and also fill the `doi` field of the
entry. Else, use the URL provided by the publisher.
- Make sure to wrap strings in the BibTeX entries in `{}` when capitalization is
important, or when BibTeX keywords should be ignored (e.g. `and` in author
lists).

To cite an entry from the `docs/References.bib` file in the documentation, use
the Doxygen keyword
\verbatim
\cite
\endverbatim
followed by the BibTeX key at the place in the documentation where you want the
citation to appear. It will render as a numbered link to the bibliography
page. It will also show a popover when hovering over the link, which displays
the bibliographic information and provides quick access to the publication.

## Include any pictures that aid in the understanding of your documentation:

First, please compress your image to be under 130kB. As most images will be
diagrammatical in nature this is certainly achievable!

Second, add the directory that will contain your image to the `docs/Images/`
directory. Create the appropriate directories such that the directory structure
in `docs/Images` matches that of `src` and `tests/Unit`.

Finally, include the image by using the Doxygen keyword

\verbatim
\image html NameOfImage.png
\endverbatim

in your hpp file.

# Python Documentation {#writing_dox_python_dox_help}

DocStrings...

\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing Good Documentation {#writing_good_dox}

\tableofcontents

# Tutorials, Instructions, and Dev Guide {#writing_dox_writing_help}

All non-code documentation such as tutorials, installation instructions, and
the developer guide is written inside Markdown files such as this one. These
files must be placed inside `doc/Tutorials`, `doc/MainSite`, or `doc/DevGuide`
according to what they document. Each Markdown file must start with the
following line

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

Doxygen comments...

# Python Documentation {#writing_dox_python_dox_help}

DocStrings...

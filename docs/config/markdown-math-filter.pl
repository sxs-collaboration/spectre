# Distributed under the MIT License.
# See LICENSE.txt for details.

# Doxygen filter to format markdown math for Doxygen
#
# In Doxygen documentation, including markdown files or Jupyter notebooks that
# get converted to markdown, you can use standard $...$ syntax for inline math,
# and "naked LaTeX" syntax for display-style math:
#
# \begin{equation}
# ...
# \end{equation}
#
# You can also replace `equation` with another supported environment, e.g.
# `align`.
#
# This Doxygen filter formats the math for Doxygen, i.e., uses \f$...\f$ for
# inline math and \f{equation}{ ... \f} for display-style math. This is needed
# until Doxygen adds support for markdown-style math, see
# https://github.com/doxygen/doxygen/issues/8009.

# Wrap inline math in \f$...\f$
# - Match code blocks (wrapped in ``` or \code...\endcode) and inline code
#   (wrapped in ` or ``), and print them directly to the output with no changes.
# - Match display math blocks (wrapped in \f[...\f], \f{...\f}, or
#   \begin...\end), and print them directly to the output with no changes.
# - Replace $...$ by \f$...\f$, unless already preceded by '\f$'.
# - The '?' makes the pattern match lazily, i.e., match as few characters as
#   possible.
# - Modifiers:
#     s: Allow '.' to match newlines
#     g: Replace all occurences
#     e: Evaluate replacement as perl code, so we can switch between
#        replacements depending on the matched alternative, using the '//'
#        (definedness) operator.
#     x: Ignore spaces for better readability
s{ (```.*?```)
   | (``.*?``)
   | (`.*?`)
   | (\\code.*?\\endcode)
   | (\\f\[.*?\\f\])
   | (\\f\{.*?\\f\})
   | (\\begin\{(.*?)\}(.*?)\\end\{\6\})
   | (?<!\\f)\$(.*?)\$
}{ $1 // $2 // $3 // $4 // $5 // $6 // $7 // "\\f\$$10\\f\$" }sgex;

# Wrap display math in \f{equation}{ ... \f}
# - Match code blocks (wrapped in ``` or \code...\endcode) and Doxygen-style
#   display equations formatted either \f[...\f] or \f{...}{...\f}, and print
#   them to the output with no changes.
# - Replace \begin{...}...\end{...} with \f{...}{...\f}.
# - Modifiers: see above
s{ (```.*?```)
   | (\\code.*?\\endcode)
   | (\\f\[.*?\\f\])
   | (\\f\{.*?\\f\})
   | \\begin\{(.*?)\}(.*?)\\end\{\5\}
}{ $1 // $2 // $3 // $4 // "\\f\{$5\}\{$6\\f\}" }sgex;

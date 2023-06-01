# Distributed under the MIT License.
# See LICENSE.txt for details.

# Targets to preprocess the documentation
# - Convert all `.ipynb` files in `docs/` to markdown, so they get picked up by
#   Doxygen.
get_filename_component(_PY_BIN_DIR ${Python_EXECUTABLE} DIRECTORY)
find_program(NBCONVERT jupyter-nbconvert HINTS ${_PY_BIN_DIR})
if (NBCONVERT STREQUAL "NBCONVERT-NOTFOUND")
  message(STATUS "jupyter-nbconvert not found. Preprocessing .ipynb files for \
documentation disabled.")
else()
  message(STATUS "Found jupyter-nbconvert: ${NBCONVERT}")
  add_custom_target(
    doc-notebooks-to-markdown
    COMMAND find ${CMAKE_SOURCE_DIR}/docs -name "*.ipynb"
      | xargs ${NBCONVERT} --to markdown --log-level=WARN
        --output-dir ${CMAKE_BINARY_DIR}/docs/tmp/notebooks_md
    )
endif()

find_package(Doxygen)
if (DOXYGEN_FOUND)
  set(SPECTRE_DOXYGEN_GROUPS "${CMAKE_BINARY_DIR}/docs/tmp/GroupDefs.hpp")

  # For INPUT_FILTER in Doxyfile. Using Python instead of Perl here increases
  # docs generation time by ~25%. Runtimes are 102s (no filter), 108s (Perl) and
  # 135s (Python) at the time of writing (Mar 2022).
  find_package(Perl)

  set(SPECTRE_DOX_GENERATE_HTML "YES")
  set(SPECTRE_DOX_GENERATE_XML "NO")
  configure_file(
    docs/Doxyfile.in
    ${PROJECT_BINARY_DIR}/docs/DoxyfileHtml @ONLY IMMEDIATE
    )
  # Configure file that contains doxygen groups
  configure_file(docs/GroupDefs.hpp ${SPECTRE_DOXYGEN_GROUPS})

  # Construct the command that calls Doxygen
  set(
    GENERATE_DOCS_COMMAND
    "${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/docs/DoxyfileHtml"
    )

  # Make sure the Doxygen version is compatible with the CSS, or print a
  # warning. Notes:
  # - We use https://github.com/jothepro/doxygen-awesome-css release v2.2.0,
  #   which ensures compatibility with Doxygen v1.9.1 - v1.9.4 and v1.9.6.
  #   When upgrading Doxygen, it's probably also a good idea to upgrade the CSS
  #   files in `docs/config/`.
  # - The Doxygen release v1.9.1 has a bug so namespaces don't show up in
  #   groups. It is fixed in v1.9.2.
  # - The Doxygen release v1.9.2 breaks the ordering of pages in the tree view
  #   (sidebar). It is fixed in v1.9.3.
  if(DOXYGEN_VERSION VERSION_LESS 1.9.3)
    set(_DOX_WARNING "Your Doxygen version ${DOXYGEN_VERSION} may not be \
compatible with the stylesheet, so the documentation may look odd or not \
function correctly. Use Doxygen version 1.9.3 or higher.")
    message(STATUS ${_DOX_WARNING})
    # The 'warning' in this message will fail the `doc-check` target (see below)
    set(
      GENERATE_DOCS_COMMAND
      "${GENERATE_DOCS_COMMAND} && echo 'WARNING: ${_DOX_WARNING}'"
      )
  endif()

  # Construct the command that calls doxygen, but only outputs warnings with a
  # few lines of context to stderr. Fails with exit code 1 if warnings are
  # found. Remains silent and succeeds with exit code 0 if everything is fine.
  # We write this into a shell script to make it easier to append the
  # postprocessing command.
  set(
    GENERATE_AND_CHECK_DOCS_SCRIPT "\
#!/bin/sh\n\
! (${GENERATE_DOCS_COMMAND}) 2>&1 | grep -A 6 -i 'warning' >&2\n"
    )

  include(FindPythonModule)
  find_python_module(bs4 FALSE)
  find_python_module(pybtex FALSE)
  if (PY_BS4 AND PY_PYBTEX)
    # Construct the command that runs the postprocessing over the Doxygen HTML
    # output
    set(
      DOCS_POST_PROCESS_COMMAND
      "${Python_EXECUTABLE} \
${CMAKE_SOURCE_DIR}/docs/config/postprocess_docs.py \
--html-dir ${PROJECT_BINARY_DIR}/docs/html \
--references-files ${CMAKE_SOURCE_DIR}/docs/References.bib \
${CMAKE_SOURCE_DIR}/docs/Dependencies.bib"
      )
    # Append postprocessing to doxygen commands
    # The commands are supposed to run the postprocessing even if the doc
    # generation failed with warnings, so that we output useful documentation
    # in any case. The commands exit successfully only if both generation and
    # postprocessing succeeded.
    set(
      GENERATE_DOCS_COMMAND
      "${GENERATE_DOCS_COMMAND} && ${DOCS_POST_PROCESS_COMMAND} -v"
      )
    set(
      GENERATE_AND_CHECK_DOCS_SCRIPT "${GENERATE_AND_CHECK_DOCS_SCRIPT}\
generate_docs_exit=$?\n\
${DOCS_POST_PROCESS_COMMAND} && exit \${generate_docs_exit}\n"
      )
  else (PY_BS4 AND PY_PYBTEX)
    message(WARNING "Doxygen documentation postprocessing is disabled because"
    " Python dependencies were not found:")
    if (NOT PY_BS4)
      message(WARNING "BeautifulSoup4 missing. "
        "Install with: pip install beautifulsoup4")
    endif()
    if (NOT PY_PYBTEX)
      message(WARNING "Pybtex missing. Install with: pip install pybtex")
    endif()
  endif (PY_BS4 AND PY_PYBTEX)

  # Parse the command into a CMake list for the `add_custom_target`
  separate_arguments(GENERATE_DOCS_COMMAND)

  add_custom_target(
    doc
    COMMAND ${GENERATE_DOCS_COMMAND}
    DEPENDS
    ${PROJECT_BINARY_DIR}/docs/DoxyfileHtml
    ${SPECTRE_DOXYGEN_GROUPS}
    )

  # Write the shell script to a file to call it in the `add_custom_target`
  file(
    WRITE
    ${PROJECT_BINARY_DIR}/docs/tmp/GenerateAndCheckDocs.sh
    ${GENERATE_AND_CHECK_DOCS_SCRIPT}
    )

  add_custom_target(
    doc-check
    COMMAND sh ${PROJECT_BINARY_DIR}/docs/tmp/GenerateAndCheckDocs.sh
    DEPENDS
    ${PROJECT_BINARY_DIR}/docs/DoxyfileHtml
    ${SPECTRE_DOXYGEN_GROUPS}
    )

  set(SPECTRE_DOX_GENERATE_HTML "NO")
  set(SPECTRE_DOX_GENERATE_XML "YES")
  configure_file(
    docs/Doxyfile.in
    ${PROJECT_BINARY_DIR}/docs/DoxyfileXml @ONLY IMMEDIATE
    )

  add_custom_target(
    doc-xml
    COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/docs/DoxyfileXml
    DEPENDS
    ${PROJECT_BINARY_DIR}/docs/DoxyfileXml
    ${SPECTRE_DOXYGEN_GROUPS}
    )

  if (TARGET doc-notebooks-to-markdown)
    add_dependencies(doc doc-notebooks-to-markdown)
    add_dependencies(doc-check doc-notebooks-to-markdown)
    add_dependencies(doc-xml doc-notebooks-to-markdown)
  endif()

  find_program(LCOV lcov)
  find_program(GENHTML genhtml)
  find_program(SED sed)

  # Use [coverxygen](https://github.com/psycofdj/coverxygen) to check the level
  # of documentation coverage.
  find_python_module(coverxygen FALSE)
  if (LCOV AND GENHTML AND SED AND PY_COVERXYGEN
      AND EXISTS ${CMAKE_SOURCE_DIR}/.git AND Git_FOUND)
    set(DOX_COVERAGE_OUTPUT "${CMAKE_BINARY_DIR}/docs/html/doc_coverage/")
    add_custom_target(
      doc-coverage

      COMMAND ${Python_EXECUTABLE}
      -m coverxygen
      --xml-dir ${CMAKE_BINARY_DIR}/docs/xml
      --src-dir ${CMAKE_SOURCE_DIR}
      --output ${CMAKE_BINARY_DIR}/docs/tmp/doc_coverage.info

      COMMAND ${LCOV}
      --remove ${CMAKE_BINARY_DIR}/docs/tmp/doc_coverage.info
      '${CMAKE_SOURCE_DIR}/src/Executables/*'
      '${CMAKE_SOURCE_DIR}/tests/*'
      '${CMAKE_SOURCE_DIR}/citelist'
      '${CMAKE_SOURCE_DIR}/[generated]'
      --output ${CMAKE_BINARY_DIR}/docs/tmp/doc_coverage.info

      COMMAND
      ${LCOV} --summary ${CMAKE_BINARY_DIR}/docs/tmp/doc_coverage.info

      COMMAND
      ${GENHTML} --legend
      --no-function-coverage
      --no-branch-coverage
      --title ${GIT_HASH}
      ${CMAKE_BINARY_DIR}/docs/tmp/doc_coverage.info
      -o ${DOX_COVERAGE_OUTPUT}

      COMMAND
      find ${DOX_COVERAGE_OUTPUT} -type f -print
      | xargs file
      | grep text
      | cut -f1 -d:
      | xargs ${SED} -i'.bak' 's/LCOV - code coverage report/
      SpECTRE Documentation Coverage Report/g'

      COMMAND find ${DOX_COVERAGE_OUTPUT} -type f -print | xargs file
      | grep text
      | cut -f1 -d:
      | xargs ${SED} -i'.bak' 's^<td class="headerItem">Test:</td>^
      <td class="headerItem">Commit:</td>^g'

      COMMAND find ${DOX_COVERAGE_OUTPUT} -type f -print | xargs file
      | grep text
      | cut -f1 -d:
      | xargs ${SED} -i'.bak' 's^<td class="headerValue">\\\([a-z0-9]\\{40\\}\\\)^
      <td class="headerValue"><a target="_blank"
      href="https://github.com/sxs-collaboration/spectre/commit/\\1">\\1</a>^g'

      # Delete backup files created by sed
      COMMAND find ${DOX_COVERAGE_OUTPUT} -type f -name \"*.bak\" -print
      | xargs file | grep text | cut -f1 -d: | xargs rm

      DEPENDS
      ${PROJECT_BINARY_DIR}/docs/DoxyfileXml
      ${SPECTRE_DOXYGEN_GROUPS}
      doc-xml

      # Set work directory for target
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}

      COMMENT "SpECTRE Documentation Coverage"
      )
  endif(LCOV AND GENHTML AND SED AND PY_COVERXYGEN
    AND EXISTS ${CMAKE_SOURCE_DIR}/.git AND Git_FOUND)
else(DOXYGEN_FOUND)
  message(WARNING "Doxygen is needed to build the documentation.")
endif (DOXYGEN_FOUND)

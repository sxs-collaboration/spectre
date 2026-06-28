# Distributed under the MIT License.
# See LICENSE.txt for details.

# Try to identify the machine we are configuring on when the user did not
# select one explicitly with `-D MACHINE=...`. We match the fully-qualified
# hostname against the `HostnameRegex` of every machine definition in
# `support/Machines/`. This mirrors the runtime Python helper
# `this_machine_by_hostname` in `support/Python/Machines.py`, but is done in
# pure CMake because this runs before Python is found.
if (NOT MACHINE)
  # Get the fully-qualified hostname. The patterns often match FQDNs (e.g.
  # `mbot.cac.cornell.edu`), so we need more than the short hostname.
  execute_process(
    COMMAND hostname -f
    OUTPUT_VARIABLE _hostname
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _hostname_result
    ERROR_QUIET
    )

  set(_matched_machines "")
  if (_hostname_result EQUAL 0 AND _hostname)
    file(GLOB _machine_yamls ${CMAKE_SOURCE_DIR}/support/Machines/*.yaml)
    foreach (_machine_yaml ${_machine_yamls})
      # Read the `HostnameRegex` line, if any. Machines without one are skipped.
      file(STRINGS ${_machine_yaml} _regex_line REGEX "HostnameRegex:")
      if (NOT _regex_line)
        continue()
      endif ()
      # Extract the pattern between single quotes.
      string(REGEX MATCH "HostnameRegex:[ \t]*'([^']*)'" _ "${_regex_line}")
      set(_hostname_regex "${CMAKE_MATCH_1}")
      if (NOT _hostname_regex)
        continue()
      endif ()
      # Unanchored search, matching Python's `re.search`. The patterns carry
      # their own `^`/`$` anchors where appropriate.
      string(REGEX MATCH "${_hostname_regex}" _match "${_hostname}")
      if (_match)
        file(STRINGS ${_machine_yaml} _name_line REGEX "Name:")
        string(REGEX MATCH "Name:[ \t]*([^ \t\r\n]+)" _ "${_name_line}")
        list(APPEND _matched_machines "${CMAKE_MATCH_1}")
      endif ()
    endforeach ()
  endif ()

  list(LENGTH _matched_machines _num_matches)
  if (_num_matches EQUAL 1)
    set(MACHINE "${_matched_machines}")
    message(STATUS "Detected machine: ${MACHINE}")
  elseif (_num_matches GREATER 1)
    string(REPLACE ";" ", " _matched_machines_str "${_matched_machines}")
    message(FATAL_ERROR
      "The hostname '${_hostname}' matches multiple machines: "
      "${_matched_machines_str}. Make their 'HostnameRegex' patterns more "
      "specific.")
  else ()
    message(STATUS "Unknown Machine")
  endif ()
endif ()

if (NOT MACHINE)
  # Since we use the variable in InfoAtCompile.cpp we always need
  # it to be set.
  set(MACHINE "UNKNOWN")
endif()

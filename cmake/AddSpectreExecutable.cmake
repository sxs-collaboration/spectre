# Distributed under the MIT License.
# See LICENSE.txt for details.

# Adds an executable by wrapping CMake's 'add_executable' but allows
# us to inject dependencies, flags, etc. into the targets.
function(add_spectre_executable TARGET_NAME)
  add_executable(${TARGET_NAME} ${ARGN})
endfunction()

# A function to add a SpECTRE executable that uses Charm++
#
# EXECUTABLE_NAME is the name of the executable (with no extension)
# HPP_NAME        is the name of the hpp file (without the .hpp extension) that
#                 contains the metavariables
# SUBDIR_NAME     is the name of the directory relative to src that contains
#                 a header file EXECUTABLE_NAME.hpp
# METAVARS        is the name of the metavariables struct that will be used
#                 as the template argument for Parallel::Main.
# LINK_LIBS       is a CMake list of SpECTRE libraries that are needed in
#                 order for the executable to link
#
# See src/Executables/Examples/HelloWorld/CMakeLists.txt for an example
#
# The function creates EXECUTABLE_NAME.cpp in the build tree which is then
# used to build EXECUTABLE_NAME which is put into the bin directory of the
# the build tree
function(
    add_spectre_parallel_executable EXECUTABLE_NAME
    HPP_NAME SUBDIR_NAME METAVARS LINK_LIBS)
  set(BUILD_TARGET_FILENAME
    "${CMAKE_BINARY_DIR}/${SUBDIR_NAME}/${EXECUTABLE_NAME}.cpp"
    )
  # We use both file(WRITE) and configure_file so that we
  # cause a rebuild only if the contents of the written file
  # has actually changed.
  file(WRITE
    "${BUILD_TARGET_FILENAME}.out"
    "// Distributed under the MIT License.\n"
    "// See LICENSE.txt for details.\n"
    "\n"
    "#include \"${SUBDIR_NAME}/${HPP_NAME}Fwd.hpp\"\n"
    "using metavariables = ${METAVARS};\n"
    "#include \"${SUBDIR_NAME}/${HPP_NAME}.hpp\"\n"
    "#include \"Parallel/Main.hpp\"\n"
    "\n"
    "using charmxx_main_component = Parallel::Main<${METAVARS}>;\n"
    "\n"
    "#include \"Parallel/CharmMain.tpp\"\n"
    )
  configure_file(
    "${BUILD_TARGET_FILENAME}.out"
    ${BUILD_TARGET_FILENAME}
    )

  add_spectre_executable(
    ${EXECUTABLE_NAME}
    EXCLUDE_FROM_ALL
    ${BUILD_TARGET_FILENAME}
    )

  add_dependencies(
    ${EXECUTABLE_NAME}
    module_ConstGlobalCache
    module_Main
    )

  target_link_libraries(
    ${EXECUTABLE_NAME}
    ${LINK_LIBS}
    ${SPECTRE_LIBRARIES}
    )
endfunction(add_spectre_parallel_executable)

# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_scalar_wave_executable DIM)
  set(EXECUTABLE_NAME "EvolveScalarWave${DIM}D")

  set(SUBDIR_NAME "src/Evolution/Executables/ScalarWave/")
  set(BUILD_TARGET_FILENAME
    "${CMAKE_BINARY_DIR}/${SUBDIR_NAME}${EXECUTABLE_NAME}.cpp"
    )
  file(WRITE
    ${BUILD_TARGET_FILENAME}
    "// Distributed under the MIT License.\n"
    "// See LICENSE.txt for details.\n"
    "\n"
    "#include \"Evolution/Executables/ScalarWave/EvolveScalarWave.hpp\"\n"
    "\n"
    "using charm_metavariables = EvolutionMetavars<${DIM}>;\n"
    "\n"
    "#include \"Parallel/CharmMain.cpp\"\n"
    )

  add_executable(
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
    CoordinateMaps
    DomainCreators
    Informer
    ScalarWave
    Time
    Utilities
    WaveEquation
    ${SPECTRE_LIBRARIES}
    )
endfunction(add_scalar_wave_executable)

add_scalar_wave_executable(1)
add_scalar_wave_executable(2)
add_scalar_wave_executable(3)

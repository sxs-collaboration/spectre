# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Informer
  Utilities
  )

add_spectre_parallel_executable(
  SingletonHelloWorld
  SingletonHelloWorld
  Executables/Examples/HelloWorld
  Metavars
  "${LIBS_TO_LINK}"
  )

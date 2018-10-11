# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  GeneralRelativitySolutions
  Informer
  Utilities
  )

add_spectre_executable(
  ObserveKerrSchild
  ObserveKerrSchild
  Executables/Examples/ObserveKerrSchild
  Metavars
  "${LIBS_TO_LINK}"
  )

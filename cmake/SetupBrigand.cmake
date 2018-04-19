# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Brigand REQUIRED)

spectre_include_directories("${BRIGAND_INCLUDE_DIR}")
message(STATUS "Brigand include: ${BRIGAND_INCLUDE_DIR}")

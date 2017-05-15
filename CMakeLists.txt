
# Distributed under the MIT License.
# See LICENSE.txt for details.

cmake_minimum_required(VERSION 3.3.2)

project(SpECTRE VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)

site_name(HOSTNAME)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(ProhibitInSourceBuild)
include(SetBuildType)
include(SetCxxStandard)

# In order to use certain code analysis tools like clang-tidy and cppcheck the
# compile commands need to be accessible. CMake can write these to a
# "compile_commands.json" file.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

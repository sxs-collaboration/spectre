
# Distributed under the University of Illinois/NCSA Open Source License.
# See LICENSE.txt for details.

cmake_minimum_required(VERSION 3.3.2)

project(SpECTRE VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(ProhibitInSourceBuild)
include(SetBuildType)

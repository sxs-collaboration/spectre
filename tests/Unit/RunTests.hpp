// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>

#include "tests/Unit/RunTests.decl.h"

// Need to register derived classes to be serialized.
void register_derived_classes_for_pup_stl_cpp11();

/// Main executable for running the unit tests.
class RunTests : public CBase_RunTests {
 public:
  [[noreturn]] explicit RunTests(CkArgMsg* msg);
};

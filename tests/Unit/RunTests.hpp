// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>

#include "tests/Unit/RunTests.decl.h"

/// Main executable for running the unit tests.
class RunTests : public CBase_RunTests {
 public:
  [[noreturn]] explicit RunTests(CkArgMsg* msg);
};

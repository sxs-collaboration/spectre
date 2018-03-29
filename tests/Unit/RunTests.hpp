// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/RunTests.decl.h"

/// \cond
class CkArgMsg;
/// \endcond

/// Main executable for running the unit tests.
class RunTests : public CBase_RunTests {
 public:
  [[noreturn]] explicit RunTests(CkArgMsg* msg);
};

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "TestMainCharm.decl.h"

/// \cond
class CkArgMsg;
/// \endcond

/// Main executable for running unit tests in a Charm++ environment
class TestMainCharm : public CBase_TestMainCharm {
 public:
  [[noreturn]] explicit TestMainCharm(CkArgMsg* msg);
};

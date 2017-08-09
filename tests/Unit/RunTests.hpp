// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>

#include "tests/Unit/RunTests.decl.h"

// Need to register derived classes to be serialized.
void register_derived_classes_for_pup_stl_cpp11();

class TestArrayChare : public CBase_TestArrayChare {
 public:
  TestArrayChare() = default;
  explicit TestArrayChare(CkMigrateMessage* /*unused*/) {}
};

class TestChare : public CBase_TestChare {
 public:
  TestChare() = default;
  explicit TestChare(CkMigrateMessage* /*unused*/) {}
};

class TestGroupChare : public CBase_TestGroupChare {
 public:
  TestGroupChare() = default;
  explicit TestGroupChare(CkMigrateMessage* /*unused*/) {}
};

class TestNodeGroupChare : public CBase_TestNodeGroupChare {
public:
 TestNodeGroupChare() = default;
 explicit TestNodeGroupChare(CkMigrateMessage* /*unused*/) {}
};

/// Main executable for running the unit tests.
class RunTests : public CBase_RunTests {
 public:
  [[noreturn]] explicit RunTests(CkArgMsg* msg);
};

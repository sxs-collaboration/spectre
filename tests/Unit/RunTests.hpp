// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>

#include "tests/Unit/RunTests.decl.h"

// Need to register derived classes to be serialized.
void register_derived_classes_for_pup_stl_cpp11();

class TestArrayChare : public CBase_TestArrayChare {
public:
  TestArrayChare() {}
  explicit TestArrayChare(CkMigrateMessage*) {}
};

class TestChare : public CBase_TestChare {
public:
  TestChare() {}
  explicit TestChare(CkMigrateMessage*) {}
};

class TestGroupChare : public CBase_TestGroupChare {
public:
  TestGroupChare() {}
  explicit TestGroupChare(CkMigrateMessage*) {}
};

class TestNodeGroupChare : public CBase_TestNodeGroupChare {
public:
  TestNodeGroupChare() {}
  explicit TestNodeGroupChare(CkMigrateMessage*) {}
};

/// Main executable for running the unit tests.
class RunTests : public CBase_RunTests {
 public:
  [[noreturn]] explicit RunTests(CkArgMsg* msg);
};

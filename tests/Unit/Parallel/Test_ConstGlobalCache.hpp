// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "src/Parallel/Info.hpp"
#include "tests/Unit/Parallel/Test_ConstGlobalCache.decl.h"

/// \cond
class CkArgMsg;
/// \endcond

class TestArrayChare : public CBase_TestArrayChare {
 public:
  TestArrayChare() = default;
  explicit TestArrayChare(CkMigrateMessage* /*unused*/) {}
  int my_index() const noexcept { return thisIndex; }
};

class ParallelComponent : public CBase_ParallelComponent {
 public:
  explicit ParallelComponent(int id) noexcept : id_(id) {}
  explicit ParallelComponent(CkMigrateMessage* /*unused*/) {}
  int my_id() const noexcept { return id_; }

 private:
  int id_{std::numeric_limits<int>::max()};
};

class TestGroupChare : public CBase_TestGroupChare {
 public:
  TestGroupChare() = default;
  explicit TestGroupChare(CkMigrateMessage* /*unused*/) {}
  int my_proc() const noexcept { return Parallel::my_proc(); }
};

class TestNodeGroupChare : public CBase_TestNodeGroupChare {
 public:
  TestNodeGroupChare() = default;
  explicit TestNodeGroupChare(CkMigrateMessage* /*unused*/) {}
  int my_node() const noexcept { return Parallel::my_node(); }
};

/// Main executable for running the unit tests.
class Test_ConstGlobalCache : public CBase_Test_ConstGlobalCache {
 public:
  [[noreturn]] explicit Test_ConstGlobalCache(CkArgMsg* msg);
};

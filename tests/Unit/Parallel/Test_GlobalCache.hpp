// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <set>

#include "Parallel/CharmRegistration.hpp"
#include "Parallel/GlobalCache.hpp"

#include "tests/Unit/Parallel/Test_GlobalCache.decl.h"

template <typename Metavariables>
class Test_GlobalCache : public CBase_Test_GlobalCache<Metavariables> {
 public:
  /// \cond HIDDEN_SYMBOLS
  /// The constructor used to register the class
  explicit Test_GlobalCache(
      const Parallel::charmxx::
          MainChareRegistrationConstructor& /*used 4 registration*/) {}
  ~Test_GlobalCache() override {
    (void)Parallel::charmxx::RegisterChare<
        Test_GlobalCache<Metavariables>,
        CkIndex_Test_GlobalCache<Metavariables>>::registrar;
  }
  Test_GlobalCache(const Test_GlobalCache&) = default;
  Test_GlobalCache& operator=(const Test_GlobalCache&) = default;
  Test_GlobalCache(Test_GlobalCache&&) = default;
  Test_GlobalCache& operator=(Test_GlobalCache&&) = default;
  /// \endcond

  explicit Test_GlobalCache(CkArgMsg* msg);
  explicit Test_GlobalCache(CkMigrateMessage* /*msg*/) {}

  void exit_if_done(int index);

  void run_single_core_test();

 private:
  Parallel::CProxy_MutableGlobalCache<Metavariables>
      mutable_global_cache_proxy_{};
  Parallel::CProxy_GlobalCache<Metavariables> global_cache_proxy_{};
  size_t num_elements_{4};
  std::set<int> elements_that_are_finished_{};
};

template <typename Metavariables>
class TestArrayChare : public CBase_TestArrayChare<Metavariables> {
 public:
  explicit TestArrayChare(
      CProxy_Test_GlobalCache<Metavariables> main_proxy,
      Parallel::CProxy_GlobalCache<Metavariables> global_cache_proxy)
      : main_proxy_(std::move(main_proxy)),
        global_cache_proxy_(std::move(global_cache_proxy)) {}
  explicit TestArrayChare(CkMigrateMessage* /*msg*/) {}
  ~TestArrayChare() override {
    (void)Parallel::charmxx::RegisterChare<
        TestArrayChare<Metavariables>,
        CkIndex_TestArrayChare<Metavariables>>::registrar;
  }
  TestArrayChare(const TestArrayChare&) = default;
  TestArrayChare& operator=(const TestArrayChare&) = default;
  TestArrayChare(TestArrayChare&&) = default;
  TestArrayChare& operator=(TestArrayChare&&) = default;

  void run_test_one();
  void run_test_two();
  void run_test_three();
  void run_test_four();
  void run_test_five();

 private:
  CProxy_Test_GlobalCache<Metavariables> main_proxy_;
  Parallel::CProxy_GlobalCache<Metavariables> global_cache_proxy_;
};

#define CK_TEMPLATES_ONLY
#include "tests/Unit/Parallel/Test_GlobalCache.def.h"
#undef CK_TEMPLATES_ONLY

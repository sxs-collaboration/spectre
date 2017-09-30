// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <catch.hpp>
#include <exception>
#include <memory>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Parallel/Abort.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/Main.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Parallel/ParallelTestChares.hpp"
#include "tests/Unit/Parallel/ParallelTestClasses.hpp"
#include "tests/Unit/Parallel/ParallelTestTentacles.hpp"
#include "tests/Unit/TestHelpers.hpp"

#include "tests/Unit/Parallel/Test_Main.decl.h"

 void setup_error_handling() {
  std::set_terminate(
      []() { Parallel::abort("Called terminate. Aborting..."); });
  enable_floating_point_exceptions();
  Parallel::printf("Calling setup_error_handling on processor %i\n",
                   Parallel::my_proc());
}

void register_derived_classes_for_pup() {
  PUPable_reg(Test_Classes::DerivedInPupStlCpp11);
}

// clang-tidy: may throw an exception tat cannot be caught [cert-err58-cpp]
Catch::Session session;  // NOLINT

struct TestMetavariables;

// This pointer is used because no arguments can be passed into a
// SPECTRE_TEST_CASE (which is a wrapped Catch TEST_CASE)
const Parallel::CProxy_ConstGlobalCache<TestMetavariables>* global_cache_proxy =
    nullptr;

struct TestMetavariables {
  using const_global_cache_tag_list = typelist<>;
  using tentacle_list = typelist<Test_Tentacles::Group<TestMetavariables>,
                                 Test_Tentacles::NodeGroup<TestMetavariables>,
                                 Test_Tentacles::Chare<TestMetavariables>,
                                 Test_Tentacles::Array<TestMetavariables>,
                                 Test_Tentacles::BoundArray<TestMetavariables>>;
  enum class phase { Initialization, Exit };
  static phase determine_next_phase(
      const Parallel::CProxy_ConstGlobalCache<TestMetavariables>& cache_proxy) {
    Parallel::printf("Determing next phase\n");
    global_cache_proxy = &cache_proxy;
    int result = session.run();
    if (0 != result) {
      Parallel::abort("A catch test has failed");
    }
    return phase::Exit;
  }
};

SPECTRE_TEST_CASE("Unit.Parallel.Main", "[Unit][Parallel]") {
  const auto& cache = *(global_cache_proxy->ckLocalBranch());
  CHECK(Parallel::my_proc() ==
        cache.get_tentacle<Test_Tentacles::Group<TestMetavariables>>()
            .ckLocalBranch()
            ->my_proc());
  CHECK(Parallel::my_node() ==
        cache.get_tentacle<Test_Tentacles::NodeGroup<TestMetavariables>>()
            .ckLocalBranch()
            ->my_node());
  auto& retrieved_proxy =
      cache.get_tentacle<Test_Tentacles::Chare<TestMetavariables>>();
  if (nullptr != retrieved_proxy.ckLocal()) {
    CHECK(-1 == retrieved_proxy.ckLocal()->my_id());
  }
  auto& retrieved_array_proxy =
      cache.get_tentacle<Test_Tentacles::Array<TestMetavariables>>();
  auto& retrieved_bound_array_proxy =
      cache.get_tentacle<Test_Tentacles::BoundArray<TestMetavariables>>();
  for (int i = 0; i < 40; ++i) {
    if (nullptr != retrieved_array_proxy[i].ckLocal()) {
      CHECK(i == retrieved_array_proxy[i].ckLocal()->my_index());
      CHECK(i == retrieved_bound_array_proxy[i].ckLocal()->my_index());
    }
  }
}

#include "src/Parallel/ConstGlobalCache.def.h"
#include "src/Parallel/Main.def.h"
#include "tests/Unit/Parallel/ParallelTestChares.def.h"

#include "tests/Unit/Parallel/Test_Main.def.h"

/// \cond
// clang-tidy: possibly throwing constructor static storage
// clang-tidy: false positive: redundant declaration
PUP::able::PUP_ID Test_Classes::DerivedInPupStlCpp11::my_PUP_ID = 0;  // NOLINT
/// \endcond

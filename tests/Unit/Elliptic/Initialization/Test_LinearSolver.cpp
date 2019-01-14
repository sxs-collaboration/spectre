// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Initialization/LinearSolver.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare Variables

namespace {
struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarFieldTag"; };
  using type = Scalar<DataVector>;
};

struct LinearSolverSourceTag : db::SimpleTag {
  static std::string name() noexcept { return "LinearSolverSourceTag"; };
  using type = Scalar<DataVector>;
};

struct LinearSolverAxTag : db::SimpleTag {
  static std::string name() noexcept { return "LinearSolverAxTag"; };
  using type = Scalar<DataVector>;
};

struct System {
  using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<>;
  using const_global_cache_tag_list = tmpl::list<>;
  struct linear_solver {
    struct tags {
      using simple_tags =
          db::AddSimpleTags<LinearSolverSourceTag, LinearSolverAxTag>;
      using compute_tags = db::AddComputeTags<>;
      template <typename TagsList, typename ArrayIndex,
                typename ParallelComponent>
      static auto initialize(
          db::DataBox<TagsList>&& box,
          const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
          const ArrayIndex& /*array_index*/,
          const ParallelComponent* const /*meta*/,
          const db::item_type<
              db::add_tag_prefix<::Tags::Source, typename system::fields_tag>>&
              b,
          const db::item_type<
              db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                                 typename system::fields_tag>>& Ax) noexcept {
        return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
            std::move(box), get<::Tags::Source<ScalarFieldTag>>(b),
            get<LinearSolver::Tags::OperatorAppliedTo<ScalarFieldTag>>(Ax));
      }
    };
  };
};

struct MockParallelComponent {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Initialization.LinearSolver",
                  "[Unit][Elliptic][Actions]") {
  db::item_type<db::add_tag_prefix<Tags::Source, typename System::fields_tag>>
      sources{3, 0.};
  get<Tags::Source<ScalarFieldTag>>(sources) =
      Scalar<DataVector>{{{{1., 2., 3.}}}};
  auto arguments_box = db::create<db::AddSimpleTags<
      db::add_tag_prefix<Tags::Source, typename System::fields_tag>>>(
      std::move(sources));

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}, {}};
  MockParallelComponent component{};
  const auto box =
      Elliptic::Initialization::LinearSolver<Metavariables>::initialize(
          std::move(arguments_box), runner.cache(), 0, &component);

  const DataVector b_expected{1., 2., 3.};
  CHECK(get<LinearSolverSourceTag>(box) == Scalar<DataVector>(b_expected));
  const DataVector Ax_expected(3, 0.);
  CHECK(get<LinearSolverAxTag>(box) == Scalar<DataVector>(Ax_expected));
}

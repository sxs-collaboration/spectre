// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
struct AnalyticSolution;

struct AnalyticSolutionOrData : PUP::able {
  AnalyticSolutionOrData() = default;
  explicit AnalyticSolutionOrData(CkMigrateMessage* m) : PUP::able(m) {}
  WRAPPED_PUPable_decl(AnalyticSolutionOrData);

  // Base class does _not_ provide variables for all system fields
};

PUPable_def(AnalyticSolutionOrData)

struct AnalyticSolution : AnalyticSolutionOrData {
  AnalyticSolution() = default;
  explicit AnalyticSolution(CkMigrateMessage* m) : AnalyticSolutionOrData(m) {}
  WRAPPED_PUPable_decl(AnalyticSolution);

  static tuples::TaggedTuple<ScalarFieldTag> variables(
      const tnsr::I<DataVector, 1>& x, tmpl::list<ScalarFieldTag> /*meta*/) {
    Scalar<DataVector> solution{2. * get<0>(x)};
    return {std::move(solution)};
  }
};

PUPable_def(AnalyticSolution)
#pragma GCC diagnostic pop

struct AnalyticSolutionTag : db::SimpleTag {
  using type = AnalyticSolution;
};

struct AnalyticSolutionOrDataTag : db::SimpleTag {
  using type = std::unique_ptr<AnalyticSolutionOrData>;
};

template <bool Optional, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<1>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<domain::Tags::Coordinates<1, Frame::Inertial>>>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              Actions::SetupDataBox,
              tmpl::conditional_t<
                  Optional,
                  elliptic::Actions::InitializeOptionalAnalyticSolution<
                      AnalyticSolutionOrDataTag, tmpl::list<ScalarFieldTag>,
                      AnalyticSolution>,
                  elliptic::Actions::InitializeAnalyticSolution<
                      AnalyticSolutionTag, tmpl::list<ScalarFieldTag>>>>>>;
};

template <bool Optional>
struct Metavariables {
  using element_array = ElementArray<Optional, Metavariables>;
  using const_global_cache_tags =
      tmpl::list<tmpl::conditional_t<Optional, AnalyticSolutionOrDataTag,
                                     AnalyticSolutionTag>>;
  using component_list = tmpl::list<element_array>;
  enum class Phase { Initialization, Testing, Exit };
};

template <bool Optional>
void test_initialize_analytic_solution(
    const tnsr::I<DataVector, 1>& inertial_coords,
    const Scalar<DataVector>& expected_solution) {
  using metavariables = Metavariables<Optional>;
  using element_array = typename metavariables::element_array;

  const auto initialize_analytic_solution =
      [&inertial_coords](auto analytic_solution_or_data) {
        ActionTesting::MockRuntimeSystem<metavariables> runner{
            {std::move(analytic_solution_or_data)}};
        const ElementId<1> element_id{0};
        ActionTesting::emplace_component_and_initialize<element_array>(
            &runner, element_id, {inertial_coords});
        ActionTesting::set_phase(make_not_null(&runner),
                                 metavariables::Phase::Testing);
        for (size_t i = 0; i < 2; ++i) {
          ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                    element_id);
        }
        return ActionTesting::get_databox_tag<element_array,
                                              ::Tags::AnalyticSolutionsBase>(
            runner, element_id);
      };

  if constexpr (Optional) {
    {
      INFO("Analytic solution is available");
      const auto analytic_solutions =
          initialize_analytic_solution(std::make_unique<AnalyticSolution>());
      REQUIRE(analytic_solutions.has_value());
      CHECK_ITERABLE_APPROX(
          get(get<::Tags::Analytic<ScalarFieldTag>>(*analytic_solutions)),
          get(expected_solution));
    }
    {
      INFO("No analytic solution is available");
      const auto no_analytic_solutions = initialize_analytic_solution(
          std::make_unique<AnalyticSolutionOrData>());
      CHECK_FALSE(no_analytic_solutions.has_value());
    }
  } else {
    const auto analytic_solutions =
        initialize_analytic_solution(AnalyticSolution{});
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::Analytic<ScalarFieldTag>>(analytic_solutions)),
        get(expected_solution));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeAnalyticSolution",
                  "[Unit][Elliptic][Actions]") {
  PUPable_reg(AnalyticSolutionOrData);
  PUPable_reg(AnalyticSolution);

  test_initialize_analytic_solution<false>(
      tnsr::I<DataVector, 1>{{{{1., 2., 3., 4.}}}},
      Scalar<DataVector>{{{{2., 4., 6., 8.}}}});
  test_initialize_analytic_solution<true>(
      tnsr::I<DataVector, 1>{{{{1., 2., 3., 4.}}}},
      Scalar<DataVector>{{{{2., 4., 6., 8.}}}});
}

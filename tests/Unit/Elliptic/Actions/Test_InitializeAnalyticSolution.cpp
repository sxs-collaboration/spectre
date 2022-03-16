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
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
struct NoAnalyticSolution : elliptic::analytic_data::Background {
  NoAnalyticSolution() = default;
  explicit NoAnalyticSolution(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m) {}
  WRAPPED_PUPable_decl_template(NoAnalyticSolution);

  // Does _not_ provide variables for all system fields
};

PUP::able::PUP_ID NoAnalyticSolution::my_PUP_ID = 0;  // NOLINT

struct AnalyticSolution : elliptic::analytic_data::AnalyticSolution {
  AnalyticSolution() = default;
  explicit AnalyticSolution(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  WRAPPED_PUPable_decl_template(AnalyticSolution);
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<AnalyticSolution>(*this);
  }

  static tuples::TaggedTuple<ScalarFieldTag> variables(
      const tnsr::I<DataVector, 1>& x, tmpl::list<ScalarFieldTag> /*meta*/) {
    Scalar<DataVector> solution{2. * get<0>(x)};
    return {std::move(solution)};
  }
};

PUP::able::PUP_ID AnalyticSolution::my_PUP_ID = 0;  // NOLINT
#pragma GCC diagnostic pop

template <typename Metavariables>
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
          tmpl::list<Actions::SetupDataBox,
                     elliptic::Actions::InitializeOptionalAnalyticSolution<
                         elliptic::Tags::Background<
                             elliptic::analytic_data::Background>,
                         tmpl::list<ScalarFieldTag>,
                         elliptic::analytic_data::AnalyticSolution>>>>;
};

struct Metavariables {
  using element_array = ElementArray<Metavariables>;
  using const_global_cache_tags = tmpl::list<
      elliptic::Tags::Background<elliptic::analytic_data::Background>>;
  using component_list = tmpl::list<element_array>;
  enum class Phase { Initialization, Testing, Exit };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<elliptic::analytic_data::Background,
                             tmpl::list<AnalyticSolution, NoAnalyticSolution>>,
                  tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                             tmpl::list<AnalyticSolution>>>;
  };
};

void test_initialize_analytic_solution(
    const tnsr::I<DataVector, 1>& inertial_coords,
    const Scalar<DataVector>& expected_solution) {
  using metavariables = Metavariables;
  using element_array = typename metavariables::element_array;

  Parallel::register_factory_classes_with_charm<metavariables>();

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

  {
    INFO("Analytic solution is available");
    const auto analytic_solutions =
        initialize_analytic_solution(std::make_unique<AnalyticSolution>());
    REQUIRE(analytic_solutions.has_value());
    CHECK_ITERABLE_APPROX(get(get<::Tags::detail::AnalyticImpl<ScalarFieldTag>>(
                              *analytic_solutions)),
                          get(expected_solution));
  }
  {
    INFO("No analytic solution is available");
    const auto no_analytic_solutions =
        initialize_analytic_solution(std::make_unique<NoAnalyticSolution>());
    CHECK_FALSE(no_analytic_solutions.has_value());
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeAnalyticSolution",
                  "[Unit][Elliptic][Actions]") {
  test_initialize_analytic_solution(
      tnsr::I<DataVector, 1>{{{{1., 2., 3., 4.}}}},
      Scalar<DataVector>{{{{2., 4., 6., 8.}}}});
}

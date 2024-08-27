// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Events/ErrorIfDataTooBig.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct Inertial;
}  // namespace Frame

namespace {
namespace TestTags {
struct ScalarVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct TensorVar : db::SimpleTag {
  using type = tnsr::ii<DataVector, 2>;
};

struct OptionalScalar : db::SimpleTag {
  using type = std::optional<Scalar<DataVector>>;
};

using VariablesOfTensor = ::Tags::Variables<tmpl::list<TensorVar>>;

struct VariablesOfTensorCompute : VariablesOfTensor, db::ComputeTag {
  using base = VariablesOfTensor;
  using argument_tags =
      tmpl::list<::domain::Tags::Coordinates<2, Frame::Inertial>>;
  static void function(
      const gsl::not_null<::Variables<tmpl::list<TensorVar>>*> vars,
      const tnsr::I<DataVector, 2>& coordinates) {
    vars->initialize(get<0>(coordinates).size());
    get<0, 0>(get<TensorVar>(*vars)) =
        get<0>(coordinates) + get<1>(coordinates);
    get<0, 1>(get<TensorVar>(*vars)) = get<0, 0>(get<TensorVar>(*vars)) + 1.0;
    get<1, 1>(get<TensorVar>(*vars)) = get<0, 0>(get<TensorVar>(*vars)) + 2.0;
  }
};
}  // namespace TestTags

using TooBig = Events::ErrorIfDataTooBig<
    2,
    tmpl::list<TestTags::ScalarVar, TestTags::TensorVar,
               TestTags::OptionalScalar>,
    tmpl::list<TestTags::VariablesOfTensorCompute>>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<Event, tmpl::list<TooBig>>>;
  };
};

void run_event(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    tnsr::I<DataVector, 2> coordinates,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    Scalar<DataVector> scalar_var,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::optional<Scalar<DataVector>> optional_scalar,
    const std::string& creation_string) {
  const auto too_big =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
          creation_string);
  CHECK(too_big->needs_evolved_variables());

  using my_component = Component<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component<my_component>(&runner, 0);

  auto databox = db::create<db::AddSimpleTags<
      domain::Tags::Element<2>, domain::Tags::Coordinates<2, Frame::Inertial>,
      TestTags::ScalarVar, TestTags::OptionalScalar>>(
      Element<2>{ElementId<2>{0}, {}}, std::move(coordinates),
      std::move(scalar_var), std::move(optional_scalar));

  auto obs_box = make_observation_box<tmpl::filter<
      TooBig::compute_tags_for_observation_box, db::is_compute_tag<tmpl::_1>>>(
      make_not_null(&databox));
  auto& cache = ActionTesting::cache<my_component>(runner, 0);
  my_component* const component_ptr = nullptr;
  too_big->run(make_not_null(&obs_box), cache, 0, component_ptr,
               {"Unused", -1.0});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Events.ErrorIfDataTooBig",
                  "[Unit][ParallelAlgorithms]") {
  run_event(tnsr::I<DataVector, 2>{{{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}}},
            Scalar<DataVector>{{{{1.0, 2.0, 3.0}}}}, std::nullopt,
            "ErrorIfDataTooBig:\n"
            "  VariablesToCheck: [ScalarVar, TensorVar, OptionalScalar]\n"
            "  Threshold: 8.5");
  run_event(tnsr::I<DataVector, 2>{{{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}}},
            Scalar<DataVector>{{{{1.0, 2.0, 3.0}}}},
            std::optional{Scalar<DataVector>{{{{1.0, 2.0, 3.0}}}}},
            "ErrorIfDataTooBig:\n"
            "  VariablesToCheck: [ScalarVar, TensorVar, OptionalScalar]\n"
            "  Threshold: 8.5");
  run_event(tnsr::I<DataVector, 2>{{{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}}},
            Scalar<DataVector>{{{{1.0, 2.0, 9.0}}}}, std::nullopt,
            "ErrorIfDataTooBig:\n"
            "  VariablesToCheck: [TensorVar]\n"
            "  Threshold: 8.5");
  CHECK_THROWS_WITH(
      run_event(tnsr::I<DataVector, 2>{{{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}}},
                Scalar<DataVector>{{{{1.0, 2.0, 9.0}}}}, std::nullopt,
                "ErrorIfDataTooBig:\n"
                "  VariablesToCheck: [ScalarVar, TensorVar, OptionalScalar]\n"
                "  Threshold: 8.5"),
      Catch::Matchers::ContainsSubstring("ScalarVar too big") and
          Catch::Matchers::ContainsSubstring("value T()=9") and
          Catch::Matchers::ContainsSubstring("at position") and
          Catch::Matchers::ContainsSubstring("T(0)=3") and
          Catch::Matchers::ContainsSubstring("T(1)=3") and
          Catch::Matchers::ContainsSubstring("with ElementId:"));
  CHECK_THROWS_WITH(
      run_event(tnsr::I<DataVector, 2>{{{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}}},
                Scalar<DataVector>{{{{1.0, 2.0, 3.0}}}},
                std::optional{Scalar<DataVector>{{{{1.0, 2.0, 9.0}}}}},
                "ErrorIfDataTooBig:\n"
                "  VariablesToCheck: [ScalarVar, TensorVar, OptionalScalar]\n"
                "  Threshold: 8.5"),
      Catch::Matchers::ContainsSubstring("OptionalScalar too big") and
          Catch::Matchers::ContainsSubstring("value T()=9") and
          Catch::Matchers::ContainsSubstring("at position") and
          Catch::Matchers::ContainsSubstring("T(0)=3") and
          Catch::Matchers::ContainsSubstring("T(1)=3") and
          Catch::Matchers::ContainsSubstring("with ElementId:"));
  CHECK_THROWS_WITH(
      run_event(tnsr::I<DataVector, 2>{{{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}}},
                Scalar<DataVector>{{{{1.0, 2.0, -9.0}}}}, std::nullopt,
                "ErrorIfDataTooBig:\n"
                "  VariablesToCheck: [ScalarVar, TensorVar, OptionalScalar]\n"
                "  Threshold: 8.5"),
      Catch::Matchers::ContainsSubstring("ScalarVar too big") and
          Catch::Matchers::ContainsSubstring("value T()=-9") and
          Catch::Matchers::ContainsSubstring("at position") and
          Catch::Matchers::ContainsSubstring("T(0)=3") and
          Catch::Matchers::ContainsSubstring("T(1)=3") and
          Catch::Matchers::ContainsSubstring("with ElementId:"));
  CHECK_THROWS_WITH(
      run_event(tnsr::I<DataVector, 2>{{{{1.0, 5.0, 3.0}, {1.0, 2.0, 3.0}}}},
                Scalar<DataVector>{{{{1.0, 2.0, 3.0}}}}, std::nullopt,
                "ErrorIfDataTooBig:\n"
                "  VariablesToCheck: [ScalarVar, TensorVar, OptionalScalar]\n"
                "  Threshold: 8.5"),
      Catch::Matchers::ContainsSubstring("TensorVar too big") and
          Catch::Matchers::ContainsSubstring("value T(0,0)=7") and
          Catch::Matchers::ContainsSubstring("T(1,0)=8") and
          Catch::Matchers::ContainsSubstring("T(1,1)=9") and
          Catch::Matchers::ContainsSubstring("at position") and
          Catch::Matchers::ContainsSubstring("T(0)=5") and
          Catch::Matchers::ContainsSubstring("T(1)=2"));
}

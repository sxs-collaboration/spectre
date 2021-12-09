// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Events/ObserveVolumeIntegrals.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace observers::Actions {
struct ContributeReductionData;
}  // namespace observers::Actions

namespace {

struct ObservationTimeTag : db::SimpleTag {
  using type = double;
};

struct TestSectionIdTag {};

struct MockContributeReductionData {
  struct Results {
    observers::ObservationId observation_id;
    std::string subfile_name;
    std::vector<std::string> reduction_names{};
    double time;
    double volume;
    std::vector<double> volume_integrals{};
  };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static Results results;

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex, typename... Ts>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    observers::ArrayComponentId /*sender_array_id*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& reduction_names,
                    Parallel::ReductionData<Ts...>&& reduction_data) {
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.reduction_names = reduction_names;
    results.time = std::get<0>(reduction_data.data());
    results.volume = std::get<1>(reduction_data.data());
    results.volume_integrals = std::get<2>(reduction_data.data());
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
MockContributeReductionData::Results MockContributeReductionData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using array_index = int;
  using chare_type = ActionTesting::MockArrayChare;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeReductionData>;
  using with_these_simple_actions = tmpl::list<MockContributeReductionData>;

  using metavariables = Metavariables;
  using array_index = int;
  using chare_type = ActionTesting::MockGroupChare;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

struct ScalarVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct ScalarVarTimesTwo : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t SpatialDim>
struct VectorVar : db::SimpleTag {
  using type = tnsr::I<DataVector, SpatialDim>;
};

template <size_t SpatialDim>
struct TensorVar : db::SimpleTag {
  using type = tnsr::Ij<DataVector, SpatialDim>;
};

struct ScalarVarTimesTwoCompute
    : db::ComputeTag,
      ::Tags::Variables<tmpl::list<ScalarVarTimesTwo>> {
  using base = ::Tags::Variables<tmpl::list<ScalarVarTimesTwo>>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<ScalarVar>;
  static void function(const gsl::not_null<type*> result,
                       const Scalar<DataVector>& scalar_var) {
    result->initialize(get(scalar_var).size());
    get(get<ScalarVarTimesTwo>(*result)) = 2.0 * get(scalar_var);
  }
};

template <size_t SpatialDim>
using variables_for_test = tmpl::list<ScalarVar, VectorVar<SpatialDim>,
                                      TensorVar<SpatialDim>, ScalarVarTimesTwo>;

template <size_t VolumeDim, typename ArraySectionIdTag>
struct Metavariables {
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  using const_global_cache_tags = tmpl::list<>;  //  unused

  using ObserveEvent = dg::Events::ObserveVolumeIntegrals<
      VolumeDim, ObservationTimeTag, variables_for_test<VolumeDim>,
      tmpl::list<ScalarVarTimesTwoCompute>, ArraySectionIdTag>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<ObserveEvent>>>;
  };
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t SpatialDim>
std::unique_ptr<DomainCreator<SpatialDim>> domain_creator();

template <>
std::unique_ptr<DomainCreator<1>> domain_creator() {
  return std::make_unique<domain::creators::Interval>(
      domain::creators::Interval({{-0.5}}, {{0.5}}, {{0}}, {{4}}, {{false}},
                                 nullptr));
}

template <>
std::unique_ptr<DomainCreator<2>> domain_creator() {
  return std::make_unique<domain::creators::Rectangle>(
      domain::creators::Rectangle({{-0.5, -0.5}}, {{0.5, 0.5}}, {{0, 0}},
                                  {{4, 4}}, {{false, false}}));
}

template <>
std::unique_ptr<DomainCreator<3>> domain_creator() {
  return std::make_unique<domain::creators::Brick>(domain::creators::Brick(
      {{-0.5, -0.5, -0.5}}, {{0.5, 0.5, 0.5}}, {{0, 0, 0}}, {{4, 4, 4}},
      {{false, false, false}}));
}

template <size_t VolumeDim, typename ArraySectionIdTag, typename ObserveEvent>
void test_observe(const std::unique_ptr<ObserveEvent> observe,
                  const std::optional<std::string>& section) {
  using metavariables = Metavariables<VolumeDim, ArraySectionIdTag>;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;

  const typename element_component::array_index array_index(0);
  const ElementId<VolumeDim> element_id{array_index};

  // ObserveVolumeIntegrals requires the domain mesh in the DataBox.
  // Any domain, mesh and basis should be fine--we'll just check received data
  const auto creator = domain_creator<VolumeDim>();
  Mesh<VolumeDim> mesh{creator->initial_extents()[0],
                       Spectral::Basis::Chebyshev,
                       Spectral::Quadrature::GaussLobatto};

  // Any data held by tensors to integrate should be fine
  MAKE_GENERATOR(gen);
  using value_type = DataVector::value_type;
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-10.0,
                                                                         10.0};
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  Variables<variables_for_test<VolumeDim>> vars(number_of_grid_points);
  fill_with_random_values(make_not_null(&vars), make_not_null(&gen),
                          make_not_null(&dist));
  auto det_inv_jacobian = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), make_not_null(&dist), number_of_grid_points);

  // Compute expected data for checks
  size_t num_tensor_components = 0;
  const DataVector det_jacobian = 1.0 / get(det_inv_jacobian);
  const double expected_volume = definite_integral(det_jacobian, mesh);
  std::vector<double> expected_volume_integrals{};
  std::vector<std::string> expected_reduction_names = {
      db::tag_name<ObservationTimeTag>(), "Volume"};
  tmpl::for_each<typename std::decay_t<decltype(vars)>::tags_list>(
      [&num_tensor_components, &vars, &expected_volume_integrals,
       &expected_reduction_names, &det_jacobian, &mesh](auto tag) {
        using tensor_tag = tmpl::type_from<decltype(tag)>;
        const auto tensor = get<tensor_tag>(vars);
        for (size_t i = 0; i < tensor.size(); ++i) {
          expected_reduction_names.push_back("VolumeIntegral(" +
                                             db::tag_name<tensor_tag>() +
                                             tensor.component_suffix(i) + ")");
          expected_volume_integrals.push_back(
              definite_integral(det_jacobian * tensor[i], mesh));
          ++num_tensor_components;
        }
      });

  const double observation_time = 2.0;
  const auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavariables>, ObservationTimeTag,
      domain::Tags::Mesh<VolumeDim>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      Tags::Variables<typename decltype(vars)::tags_list>,
      observers::Tags::ObservationKey<ArraySectionIdTag>>>(
      metavariables{}, observation_time, mesh, det_inv_jacobian, vars, section);

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      0);
  ActionTesting::emplace_group_component<observer_component>(&runner);

  const auto ids_to_register =
      observers::get_registration_observation_type_and_key(*observe, box);
  const std::string expected_subfile_name{
      "/volume_integrals" +
      (std::is_same_v<ArraySectionIdTag, void> ? ""
                                               : section.value_or("Unused"))};
  const observers::ObservationKey expected_observation_key_for_reg(
      expected_subfile_name);
  if (std::is_same_v<ArraySectionIdTag, void> or section.has_value()) {
    CHECK(ids_to_register->first == observers::TypeOfObservation::Reduction);
    CHECK(ids_to_register->second == expected_observation_key_for_reg);
  } else {
    CHECK_FALSE(ids_to_register.has_value());
  }

  observe->run(make_observation_box<db::AddComputeTags<>>(box),
               ActionTesting::cache<element_component>(runner, array_index),
               array_index, std::add_pointer_t<element_component>{});

  if (not std::is_same_v<ArraySectionIdTag, void> and not section.has_value()) {
    CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));
    return;
  }

  // Process the data
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results = MockContributeReductionData::results;
  CHECK(results.observation_id.value() == observation_time);
  CHECK(results.observation_id.observation_key() ==
        expected_observation_key_for_reg);
  CHECK(results.subfile_name == expected_subfile_name);
  CHECK(results.reduction_names == expected_reduction_names);
  CHECK(results.time == observation_time);
  CHECK(results.volume == expected_volume);
  CHECK(results.volume_integrals.size() == num_tensor_components);
  CHECK(results.volume_integrals == expected_volume_integrals);

  CHECK(observe->needs_evolved_variables());
}
}  // namespace

template <size_t VolumeDim, typename ArraySectionIdTag = void>
void test_observe_system(
    const std::optional<std::string>& section = std::nullopt) {
  using metavariables = Metavariables<VolumeDim, ArraySectionIdTag>;
  {
    INFO("Testing observation for Dim = " << VolumeDim);
    test_observe<VolumeDim, ArraySectionIdTag>(
        std::make_unique<typename metavariables::ObserveEvent>(
            "volume_integrals"),
        section);
  }
  {
    INFO("Testing create/serialize for Dim = " << VolumeDim);
    Parallel::register_factory_classes_with_charm<metavariables>();
    const auto factory_event =
        TestHelpers::test_creation<std::unique_ptr<Event>, metavariables>(
            "ObserveVolumeIntegrals:\n"
            "  SubfileName: volume_integrals");
    auto serialized_event = serialize_and_deserialize(factory_event);
    test_observe<VolumeDim, ArraySectionIdTag>(std::move(serialized_event),
                                               section);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.dG.ObserveVolumeIntegrals",
                  "[Unit][Evolution]") {
  test_observe_system<1, void>();
  test_observe_system<1, void>("Section0");
  test_observe_system<1, TestSectionIdTag>(std::nullopt);
  test_observe_system<1, TestSectionIdTag>("Section0");
  test_observe_system<2>();
  test_observe_system<3>();
}

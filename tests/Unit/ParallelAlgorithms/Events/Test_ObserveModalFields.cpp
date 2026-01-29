// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Events/ObserveModalFields.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct MockContributeVolumeData {
  struct Results {
    observers::ObservationId observation_id{};
    std::string subfile_name{};
    Parallel::ArrayComponentId array_component_id{};
    ElementVolumeData received_volume_data{};
  };

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static Results results;

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& /*box*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const observers::ObservationId& observation_id,
      const std::string& subfile_name,
      const Parallel::ArrayComponentId& array_component_id,
      ElementVolumeData&& received_volume_data,
      const std::optional<std::string>& /*dependency*/ = std::nullopt) {
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.array_component_id = array_component_id;
    results.received_volume_data = std::move(received_volume_data);
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
MockContributeVolumeData::Results MockContributeVolumeData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeVolumeData>;
  using with_these_simple_actions = tmpl::list<MockContributeVolumeData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct MinimalSystem {
  using data_type = DataVector;
  static constexpr size_t volume_dim = 3;

  struct ScalarVar : db::SimpleTag {
    static std::string name() { return "Scalar"; }
    using type = Scalar<DataVector>;
  };

  struct VectorVar : db::SimpleTag {
    static std::string name() { return "Vector"; }
    using type = tnsr::i<DataVector, volume_dim, Frame::Inertial>;
  };

  using variables_tag = ::Tags::Variables<tmpl::list<ScalarVar, VectorVar>>;
  using ObserveEvent =
      dg::Events::ObserveModalFields<volume_dim,
                                     tmpl::list<ScalarVar, VectorVar>>;
};

struct Metavariables {
  using system = MinimalSystem;
  static constexpr size_t volume_dim = system::volume_dim;
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<typename system::ObserveEvent>>>;
  };
};

template <typename ObserveEvent>
void test_modal_observe(
    const ObserveEvent& observe,
    const std::optional<std::array<size_t, Metavariables::volume_dim>>&
        truncation_extents) {
  using metavariables = Metavariables;
  constexpr size_t volume_dim = Metavariables::volume_dim;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;

  const ElementId<volume_dim> element_id(0);
  const Element<volume_dim> element(element_id, {});
  const domain::creators::Rectilinear<volume_dim> rectilinear{
      make_array<volume_dim>(-2.0), make_array<volume_dim>(2.0),
      make_array<volume_dim>(0_st), make_array<volume_dim>(5_st),
      make_array<volume_dim>(true)};
  const Mesh<volume_dim> mesh(5, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto);

  const double observation_time = 2.0;
  Variables<typename MinimalSystem::variables_tag::tags_list> vars(
      mesh.number_of_grid_points());
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      element_id);
  ActionTesting::emplace_group_component<observer_component>(&runner);

  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavariables>,
      domain::Tags::Domain<volume_dim>, domain::Tags::Element<volume_dim>,
      domain::Tags::Mesh<volume_dim>,
      ::Tags::Variables<typename decltype(vars)::tags_list>,
      observers::Tags::ObservationKey<void>>>(
      metavariables{}, rectilinear.create_domain(), element, mesh, vars,
      std::optional<std::string>{});

  MockContributeVolumeData::results = {};

  auto obs_box = make_observation_box<tmpl::push_back<
      tmpl::filter<typename ObserveEvent::compute_tags_for_observation_box,
                   db::is_compute_tag<tmpl::_1>>,
      ::Events::Tags::ObserverMeshCompute<volume_dim>>>(make_not_null(&box));

  const Event::ObservationValue observation_value{"TestObservation",
                                                  observation_time};

  observe(obs_box, mesh,
          ActionTesting::cache<element_component>(runner, element_id),
          element_id, static_cast<const element_component*>(nullptr),
          observation_value);
  runner.template invoke_queued_simple_action<observer_component>(0);

  const auto& results = MockContributeVolumeData::results;
  const Mesh<volume_dim> mesh_for_output =
      truncation_extents.has_value()
          ? Mesh<volume_dim>(truncation_extents.value(), mesh.basis(),
                             mesh.quadrature())
          : mesh;
  const auto mesh_extents = mesh_for_output.extents();
  const std::vector<size_t> expected_extents{mesh_extents.begin(),
                                             mesh_extents.end()};
  CHECK(results.received_volume_data.extents == expected_extents);
  std::unordered_map<std::string, DataVector> expected_components{};
  const auto modal_data = [&mesh, &mesh_for_output](const DataVector& nodal) {
    ModalVector modal = to_modal_coefficients(nodal, mesh);
    if (mesh_for_output != mesh) {
      ModalVector truncated(mesh_for_output.number_of_grid_points());
      const Index<volume_dim> source_extents(mesh.extents());
      const Index<volume_dim> target_extents(mesh_for_output.extents());
      for (size_t target_linear = 0;
           target_linear < mesh_for_output.number_of_grid_points();
           ++target_linear) {
        const Index<volume_dim> target_multi =
            expanded_index(target_linear, target_extents);
        Index<volume_dim> source_multi{0};
        for (size_t d = 0; d < volume_dim; ++d) {
          source_multi[d] = target_multi[d];
        }
        const size_t source_linear =
            collapsed_index(source_multi, source_extents);
        truncated[target_linear] = modal[source_linear];
      }
      modal = std::move(truncated);
    }
    DataVector result(modal.size());
    for (size_t i = 0; i < modal.size(); ++i) {
      result[i] = modal[i];
    }
    return result;
  };

  expected_components["Scalar"] =
      modal_data(get<MinimalSystem::ScalarVar>(vars).get());
  const auto& vector = get<MinimalSystem::VectorVar>(vars);
  for (size_t i = 0; i < volume_dim; ++i) {
    expected_components["Vector" + vector.component_suffix(i)] =
        modal_data(vector.get(i));
  }

  REQUIRE(results.received_volume_data.tensor_components.size() ==
          expected_components.size());
  for (const auto& component : results.received_volume_data.tensor_components) {
    const auto expected_it = expected_components.find(component.name);
    REQUIRE(expected_it != expected_components.end());
    CHECK(std::holds_alternative<DataVector>(component.data));
    CHECK_ITERABLE_APPROX(std::get<DataVector>(component.data),
                          expected_it->second);
  }
}

void test_modal_factory_creation() {
  const auto obs =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
          "ObserveModalFields:\n"
          "  SubfileName: element_data\n"
          "  VariablesToObserve: [Scalar, Vector]\n"
          "  BlocksToObserve: All\n"
          "  TruncateToExtents: None\n");
  CHECK(obs != nullptr);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Events.ObserveModalFields",
                  "[Unit][ParallelAlgorithms]") {
  const MinimalSystem::ObserveEvent observe{
      "element_data", std::vector<std::string>{"Scalar", "Vector"},
      std::nullopt, std::nullopt};

  test_modal_observe(observe, std::nullopt);

  const std::array<size_t, MinimalSystem::volume_dim> truncation_extents{3, 3,
                                                                         3};
  const MinimalSystem::ObserveEvent observe_truncated{
      "element_data", std::vector<std::string>{"Scalar", "Vector"},
      std::nullopt, truncation_extents};
  test_modal_observe(observe_truncated, truncation_extents);

  test_modal_factory_creation();
}

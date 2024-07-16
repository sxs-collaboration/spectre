// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Actions/NumericInitialData.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/PureSphericalHarmonic.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace CurvedScalarWave::Actions {

namespace {

using all_scalar_vars =
    tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
               CurvedScalarWave::Tags::Phi<3>>;

template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<
              ::Tags::Variables<all_scalar_vars>, domain::Tags::Mesh<3>,
              domain::Tags::Coordinates<3, Frame::Inertial>,
              domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                            Frame::Inertial>>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<CurvedScalarWave::Actions::ReadNumericInitialData,
                     CurvedScalarWave::Actions::SetNumericInitialData>>>;
};

struct MockReadVolumeData {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      DataBox& /*box*/, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const importers::ImporterOptions& options, const size_t volume_data_id,
      tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
          importers::Tags::Selected, NumericInitialData::all_vars>>
          selected_fields) {
    const auto& initial_data = dynamic_cast<const NumericInitialData&>(
        get<evolution::initial_data::Tags::InitialData>(cache));
    CHECK(options == initial_data.importer_options());
    CHECK(volume_data_id == initial_data.volume_data_id());
    CHECK(get<importers::Tags::Selected<CurvedScalarWave::Tags::Psi>>(
              selected_fields) == "CustomPsi");
    CHECK(get<importers::Tags::Selected<CurvedScalarWave::Tags::Pi>>(
              selected_fields) == "CustomPi");
  }
};

template <typename Metavariables>
struct MockVolumeDataReader {
  using component_being_mocked = importers::ElementDataReader<Metavariables>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using replace_these_simple_actions =
      tmpl::list<importers::Actions::ReadAllVolumeDataAndDistribute<
          metavariables::volume_dim, NumericInitialData::all_vars,
          MockElementArray<Metavariables>>>;
  using with_these_simple_actions = tmpl::list<MockReadVolumeData>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 3;
  using component_list = tmpl::list<MockElementArray<Metavariables>,
                                    MockVolumeDataReader<Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<evolution::initial_data::InitialData,
                             tmpl::list<NumericInitialData>>>;
  };
};

void test_numeric_initial_data(const NumericInitialData& initial_data,
                               const std::string& option_string) {
  {
    INFO("Factory creation");
    const auto created = TestHelpers::test_creation<
        std::unique_ptr<evolution::initial_data::InitialData>, Metavariables>(
        option_string);
    CHECK(dynamic_cast<const NumericInitialData&>(*created) == initial_data);
  }

  using reader_component = MockVolumeDataReader<Metavariables>;
  using element_array = MockElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      initial_data.get_clone()};

  const double wave_radius = 2.0;

  CurvedScalarWave::AnalyticData::PureSphericalHarmonic csw_sh_wave{
      wave_radius, 1.0, std::pair<size_t, int>{0, 0}};

  // Setup mock data file reader
  ActionTesting::emplace_nodegroup_component<reader_component>(
      make_not_null(&runner));

  // Setup element
  const ElementId<3> element_id{0, {{{1, 0}, {1, 0}, {1, 0}}}};
  const Mesh<3> mesh{6, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  const auto map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          domain::CoordinateMaps::Wedge<3>{
              wave_radius / 2., wave_radius * 2., 1., 1., {}, true});
  const auto logical_coords = logical_coordinates(mesh);
  const auto coords = map(logical_coords);
  const auto inv_jacobian = map.inv_jacobian(logical_coords);
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), element_id,
      {Variables<all_scalar_vars>{num_points}, mesh, coords, inv_jacobian});

  const auto get_element_tag = [&runner,
                                &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // ReadNumericInitialData
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  REQUIRE_FALSE(ActionTesting::next_action_if_ready<element_array>(
      make_not_null(&runner), element_id));

  // MockReadVolumeData
  ActionTesting::invoke_queued_simple_action<reader_component>(
      make_not_null(&runner), 0);

  // Insert Csw data into the inbox
  auto& inbox = ActionTesting::get_inbox_tag<
      element_array, importers::Tags::VolumeData<NumericInitialData::all_vars>,
      Metavariables>(make_not_null(&runner),
                     element_id)[initial_data.volume_data_id()];
  auto csw_sh_wave_vars = csw_sh_wave.variables(coords, all_scalar_vars{});
  get<CurvedScalarWave::Tags::Psi>(inbox) =
      get<CurvedScalarWave::Tags::Psi>(csw_sh_wave_vars);
  get<CurvedScalarWave::Tags::Pi>(inbox) =
      get<CurvedScalarWave::Tags::Pi>(csw_sh_wave_vars);

  // SetNumericInitialData
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  // Check result
  tmpl::for_each<all_scalar_vars>([&get_element_tag,
                                   &csw_sh_wave_vars](const auto tag_v) {
    using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
    CAPTURE(pretty_type::name<tag>());
    CHECK_ITERABLE_APPROX(get_element_tag(tag{}), (get<tag>(csw_sh_wave_vars)));
  });
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.SetInitialData",
                  "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();
  test_numeric_initial_data(NumericInitialData{"TestInitialData.h5",
                                               "VolumeData",
                                               0.,
                                               false,
                                               {"CustomPsi", "CustomPi"}},
                            "NumericInitialData:\n"
                            "  FileGlob: TestInitialData.h5\n"
                            "  Subgroup: VolumeData\n"
                            "  ObservationValue: 0.\n"
                            "  Interpolate: False\n"
                            "  Variables:\n"
                            "    Psi: CustomPsi\n"
                            "    Pi: CustomPi");
}

}  // namespace CurvedScalarWave::Actions

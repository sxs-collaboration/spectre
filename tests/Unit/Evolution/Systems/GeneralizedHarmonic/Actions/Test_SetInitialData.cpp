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
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/SetInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.tpp"
#include "Time/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace gh {
namespace {

using gh_system_vars = gh::System<3>::variables_tag::tags_list;

template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::append<
              gh_system_vars,
              tmpl::list<domain::Tags::Mesh<3>,
                         domain::Tags::Coordinates<3, Frame::Inertial>,
                         domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                                       Frame::Inertial>,
                         ::Tags::Time>>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<gh::Actions::SetInitialData,
                     gh::Actions::ReceiveNumericInitialData>>>;
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
    const auto selected_vars = initial_data.selected_variables();
    if (std::holds_alternative<NumericInitialData::GhVars>(selected_vars)) {
      CHECK(get<importers::Tags::Selected<
                gr::Tags::SpacetimeMetric<DataVector, 3>>>(selected_fields) ==
            "CustomSpacetimeMetric");
      CHECK(get<importers::Tags::Selected<Tags::Pi<DataVector, 3>>>(
                selected_fields) == "CustomPi");
      CHECK_FALSE(
          get<importers::Tags::Selected<
              gr::Tags::SpatialMetric<DataVector, 3>>>(selected_fields));
      CHECK_FALSE(get<importers::Tags::Selected<gr::Tags::Lapse<DataVector>>>(
          selected_fields));
      CHECK_FALSE(
          get<importers::Tags::Selected<gr::Tags::Shift<DataVector, 3>>>(
              selected_fields));
    } else if (std::holds_alternative<NumericInitialData::AdmVars>(
                   selected_vars)) {
      CHECK(get<importers::Tags::Selected<
                gr::Tags::SpatialMetric<DataVector, 3>>>(selected_fields) ==
            "CustomSpatialMetric");
      CHECK(get<importers::Tags::Selected<gr::Tags::Lapse<DataVector>>>(
                selected_fields) == "CustomLapse");
      CHECK(get<importers::Tags::Selected<gr::Tags::Shift<DataVector, 3>>>(
                selected_fields) == "CustomShift");
      CHECK(get<importers::Tags::Selected<
                gr::Tags::ExtrinsicCurvature<DataVector, 3>>>(
                selected_fields) == "CustomExtrinsicCurvature");
      CHECK_FALSE(
          get<importers::Tags::Selected<
              gr::Tags::SpacetimeMetric<DataVector, 3>>>(selected_fields));
      CHECK_FALSE(get<importers::Tags::Selected<Tags::Pi<DataVector, 3>>>(
          selected_fields));
    } else {
      REQUIRE(false);
    }
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
    using factory_classes = tmpl::map<tmpl::pair<
        evolution::initial_data::InitialData,
        tmpl::list<NumericInitialData,
                   gh::Solutions::WrappedGr<gr::Solutions::KerrSchild>>>>;
  };
};

void test_set_initial_data(
    const evolution::initial_data::InitialData& initial_data,
    const std::string& option_string, const bool is_numeric) {
  {
    INFO("Factory creation");
    const auto created = TestHelpers::test_creation<
        std::unique_ptr<evolution::initial_data::InitialData>, Metavariables>(
        option_string);
    if (is_numeric) {
      CHECK(dynamic_cast<const NumericInitialData&>(*created) ==
            dynamic_cast<const NumericInitialData&>(initial_data));
    } else {
      CHECK(dynamic_cast<
                const gh::Solutions::WrappedGr<gr::Solutions::KerrSchild>&>(
                *created) ==
            dynamic_cast<
                const gh::Solutions::WrappedGr<gr::Solutions::KerrSchild>&>(
                initial_data));
    }
  }

  using reader_component = MockVolumeDataReader<Metavariables>;
  using element_array = MockElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      initial_data.get_clone()};

  // Setup mock data file reader
  ActionTesting::emplace_nodegroup_component<reader_component>(
      make_not_null(&runner));

  // Setup element
  const ElementId<3> element_id{0, {{{1, 0}, {1, 0}, {1, 0}}}};
  const Mesh<3> mesh{8, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          domain::CoordinateMaps::Wedge<3>{2., 4., 1., 1., {}, true});
  const auto logical_coords = logical_coordinates(mesh);
  const auto coords = map(logical_coords);
  const auto inv_jacobian = map.inv_jacobian(logical_coords);
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), element_id,
      {tnsr::aa<DataVector, 3>{}, tnsr::aa<DataVector, 3>{},
       tnsr::iaa<DataVector, 3>{}, mesh, coords, inv_jacobian, 0.});

  const auto get_element_tag = [&runner,
                                &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  // We use a Kerr solution to generate data
  gh::Solutions::WrappedGr<gr::Solutions::KerrSchild> kerr{
      1., {{0., 0., 0.}}, {{0., 0., 0.}}};
  const auto kerr_gh_vars = kerr.variables(coords, 0., gh_system_vars{});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // SetInitialData
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  if (is_numeric) {
    INFO("Numeric initial data");
    const auto& numeric_id =
        dynamic_cast<const NumericInitialData&>(initial_data);

    REQUIRE_FALSE(ActionTesting::next_action_if_ready<element_array>(
        make_not_null(&runner), element_id));

    // MockReadVolumeData
    ActionTesting::invoke_queued_simple_action<reader_component>(
        make_not_null(&runner), 0);

    // Insert KerrSchild data into the inbox
    auto& inbox = ActionTesting::get_inbox_tag<
        element_array,
        importers::Tags::VolumeData<NumericInitialData::all_vars>,
        Metavariables>(make_not_null(&runner),
                       element_id)[numeric_id.volume_data_id()];
    const auto& selected_vars = numeric_id.selected_variables();
    if (std::holds_alternative<NumericInitialData::GhVars>(selected_vars)) {
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(inbox) =
          get<gr::Tags::SpacetimeMetric<DataVector, 3>>(kerr_gh_vars);
      get<Tags::Pi<DataVector, 3>>(inbox) =
          get<Tags::Pi<DataVector, 3>>(kerr_gh_vars);
    } else if (std::holds_alternative<NumericInitialData::AdmVars>(
                   selected_vars)) {
      const auto kerr_adm_vars = kerr.variables(
          coords, 0.,
          tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
                     gr::Tags::Lapse<DataVector>,
                     gr::Tags::Shift<DataVector, 3>,
                     gr::Tags::ExtrinsicCurvature<DataVector, 3>>{});
      get<gr::Tags::SpatialMetric<DataVector, 3>>(inbox) =
          get<gr::Tags::SpatialMetric<DataVector, 3>>(kerr_adm_vars);
      get<gr::Tags::Lapse<DataVector>>(inbox) =
          get<gr::Tags::Lapse<DataVector>>(kerr_adm_vars);
      get<gr::Tags::Shift<DataVector, 3>>(inbox) =
          get<gr::Tags::Shift<DataVector, 3>>(kerr_adm_vars);
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(inbox) =
          get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(kerr_adm_vars);
    } else {
      REQUIRE(false);
    }

    // ReceiveNumericInitialData
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }

  // Check result. These variables are not particularly precise because we are
  // taking numerical derivatives on a fairly coarse wedge-shaped grid.
  Approx custom_approx = Approx::custom().epsilon(1.e-3).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_element_tag(gr::Tags::SpacetimeMetric<DataVector, 3>{}),
      (get<gr::Tags::SpacetimeMetric<DataVector, 3>>(kerr_gh_vars)),
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get_element_tag(Tags::Pi<DataVector, 3>{}),
                               (get<Tags::Pi<DataVector, 3>>(kerr_gh_vars)),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get_element_tag(Tags::Phi<DataVector, 3>{}),
                               (get<Tags::Phi<DataVector, 3>>(kerr_gh_vars)),
                               custom_approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Gh.NumericInitialData",
                  "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();
  test_set_initial_data(
      NumericInitialData{
          "TestInitialData.h5", "VolumeData", 0., false,
          NumericInitialData::GhVars{"CustomSpacetimeMetric", "CustomPi"}},
      "NumericInitialData:\n"
      "  FileGlob: TestInitialData.h5\n"
      "  Subgroup: VolumeData\n"
      "  ObservationValue: 0.\n"
      "  Interpolate: False\n"
      "  Variables:\n"
      "    SpacetimeMetric: CustomSpacetimeMetric\n"
      "    Pi: CustomPi\n",
      true);
  test_set_initial_data(
      NumericInitialData{"TestInitialData.h5", "VolumeData", 0., false,
                         NumericInitialData::AdmVars{
                             "CustomSpatialMetric", "CustomLapse",
                             "CustomShift", "CustomExtrinsicCurvature"}},
      "NumericInitialData:\n"
      "  FileGlob: TestInitialData.h5\n"
      "  Subgroup: VolumeData\n"
      "  ObservationValue: 0.\n"
      "  Interpolate: False\n"
      "  Variables:\n"
      "    SpatialMetric: CustomSpatialMetric\n"
      "    Lapse: CustomLapse\n"
      "    Shift: CustomShift\n"
      "    ExtrinsicCurvature: CustomExtrinsicCurvature",
      true);
  test_set_initial_data(
      gh::Solutions::WrappedGr<gr::Solutions::KerrSchild>{
          1., {{0., 0., 0.}}, {{0., 0., 0.}}},
      "KerrSchild:\n"
      "  Mass: 1.\n"
      "  Spin: [0, 0, 0]\n"
      "  Center: [0, 0, 0]",
      false);
}

}  // namespace gh

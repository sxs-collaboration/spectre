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
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Actions/SetInitialData.hpp"
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
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::GhValenciaDivClean::Actions {
namespace {

using TovStar = gh::Solutions::WrappedGr<RelativisticEuler::Solutions::TovStar>;

using gh_system_vars = gh::System<3>::variables_tag::tags_list;
using all_ghmhd_vars =
    tmpl::append<hydro::grmhd_tags<DataVector>, gh_system_vars>;

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
              ::Tags::Variables<all_ghmhd_vars>,
              hydro::Tags::EquationOfState<
                  std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>>,
              domain::Tags::Mesh<3>,
              domain::Tags::Coordinates<3, Frame::Inertial>,
              domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                            Frame::Inertial>,
              ::Tags::Time>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<
              grmhd::GhValenciaDivClean::Actions::SetInitialData,
              grmhd::GhValenciaDivClean::Actions::ReceiveNumericInitialData>>>;
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
    CHECK(get<importers::Tags::Selected<
              hydro::Tags::RestMassDensity<DataVector>>>(selected_fields) ==
          "CustomRho");
    CHECK(get<importers::Tags::Selected<
              hydro::Tags::LowerSpatialFourVelocity<DataVector, 3>>>(
              selected_fields) == "CustomUi");
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
                             tmpl::list<NumericInitialData, TovStar>>>;
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
      CHECK(dynamic_cast<const TovStar&>(*created) ==
            dynamic_cast<const TovStar&>(initial_data));
    }
  }

  using reader_component = MockVolumeDataReader<Metavariables>;
  using element_array = MockElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      initial_data.get_clone()};

  // We get test data from a TOV solution
  TovStar tov_star{
      1.e-3,
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(100., 2.)};
  const double star_radius = tov_star.radial_solution().outer_radius();
  const auto& eos = tov_star.equation_of_state();

  // Setup mock data file reader
  ActionTesting::emplace_nodegroup_component<reader_component>(
      make_not_null(&runner));

  // Setup element
  const ElementId<3> element_id{0, {{{1, 0}, {1, 0}, {1, 0}}}};
  const Mesh<3> mesh{8, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  const auto map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          domain::CoordinateMaps::Wedge<3>{
              0.75 * star_radius, star_radius * 1.25, 1., 1., {}, true});
  const auto logical_coords = logical_coordinates(mesh);
  const auto coords = map(logical_coords);
  auto inv_jacobian = map.inv_jacobian(logical_coords);
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), element_id,
      {Variables<all_ghmhd_vars>{num_points}, eos.get_clone(), mesh, coords,
       std::move(inv_jacobian), 0.});
  auto tov_vars = tov_star.variables(
      coords, 0.,
      tmpl::append<hydro::grmhd_tags<DataVector>, gh_system_vars,
                   tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>>>{});

  const auto get_element_tag = [&runner,
                                &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

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

    // Insert TOV data into the inbox
    auto& inbox = ActionTesting::get_inbox_tag<
        element_array,
        importers::Tags::VolumeData<NumericInitialData::all_vars>,
        Metavariables>(make_not_null(&runner),
                       element_id)[numeric_id.volume_data_id()];
    const auto& gh_selected_vars =
        numeric_id.gh_numeric_id().selected_variables();
    if (std::holds_alternative<gh::NumericInitialData::GhVars>(
            gh_selected_vars)) {
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(inbox) =
          get<gr::Tags::SpacetimeMetric<DataVector, 3>>(tov_vars);
      get<gh::Tags::Pi<DataVector, 3>>(inbox) =
          get<gh::Tags::Pi<DataVector, 3>>(tov_vars);
    } else if (std::holds_alternative<gh::NumericInitialData::AdmVars>(
                   gh_selected_vars)) {
      const auto tov_adm_vars = tov_star.variables(
          coords, 0.,
          tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
                     gr::Tags::Lapse<DataVector>,
                     gr::Tags::Shift<DataVector, 3>,
                     gr::Tags::ExtrinsicCurvature<DataVector, 3>>{});
      get<gr::Tags::SpatialMetric<DataVector, 3>>(inbox) =
          get<gr::Tags::SpatialMetric<DataVector, 3>>(tov_adm_vars);
      get<gr::Tags::Lapse<DataVector>>(inbox) =
          get<gr::Tags::Lapse<DataVector>>(tov_adm_vars);
      get<gr::Tags::Shift<DataVector, 3>>(inbox) =
          get<gr::Tags::Shift<DataVector, 3>>(tov_adm_vars);
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(inbox) =
          get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(tov_adm_vars);
    } else {
      REQUIRE(false);
    }
    const auto& hydro_selected_vars =
        numeric_id.hydro_numeric_id().selected_variables();
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<DataVector, 3>>(tov_vars);
    get<hydro::Tags::RestMassDensity<DataVector>>(inbox) =
        get<hydro::Tags::RestMassDensity<DataVector>>(tov_vars);
    const auto& W = get<hydro::Tags::LorentzFactor<DataVector>>(tov_vars);
    auto& u_i =
        get<hydro::Tags::LowerSpatialFourVelocity<DataVector, 3>>(inbox);
    u_i = raise_or_lower_index(
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(tov_vars),
        spatial_metric);
    for (size_t d = 0; d < 3; ++d) {
      u_i.get(d) *= get(W);
    }
    get<hydro::Tags::ElectronFraction<DataVector>>(inbox) =
        get<hydro::Tags::ElectronFraction<DataVector>>(tov_vars);
    get<hydro::Tags::MagneticField<DataVector, 3>>(inbox) =
        get<hydro::Tags::MagneticField<DataVector, 3>>(tov_vars);

    // Override hydro variables if constant value is specified in options
    const auto selected_electron_fraction =
        get<ValenciaDivClean::NumericInitialData::VarName<
            hydro::Tags::ElectronFraction<DataVector>,
            std::bool_constant<false>>>(hydro_selected_vars);
    if (std::holds_alternative<double>(selected_electron_fraction)) {
      get<hydro::Tags::ElectronFraction<DataVector>>(tov_vars) =
          make_with_value<Scalar<DataVector>>(
              W, std::get<double>(selected_electron_fraction));
    }
    const auto selected_magnetic_field =
        get<ValenciaDivClean::NumericInitialData::VarName<
            hydro::Tags::MagneticField<DataVector, 3>,
            std::bool_constant<false>>>(hydro_selected_vars);
    if (std::holds_alternative<double>(selected_magnetic_field)) {
      get<hydro::Tags::MagneticField<DataVector, 3>>(tov_vars) =
          make_with_value<tnsr::I<DataVector, 3>>(
              W, std::get<double>(selected_magnetic_field));
    }

    // ReceiveNumericInitialData
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }

  // Check result. The GH variables are not particularly precise because we
  // are taking numerical derivatives on a fairly coarse wedge-shaped grid.
  Approx custom_approx = Approx::custom().epsilon(1.e-2).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_element_tag(gr::Tags::SpacetimeMetric<DataVector, 3>{}),
      (get<gr::Tags::SpacetimeMetric<DataVector, 3>>(tov_vars)), custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get_element_tag(gh::Tags::Pi<DataVector, 3>{}),
                               (get<gh::Tags::Pi<DataVector, 3>>(tov_vars)),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get_element_tag(gh::Tags::Phi<DataVector, 3>{}),
                               (get<gh::Tags::Phi<DataVector, 3>>(tov_vars)),
                               custom_approx);
  tmpl::for_each<hydro::grmhd_tags<DataVector>>(
      [&get_element_tag, &tov_vars](const auto tag_v) {
        using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
        CAPTURE(pretty_type::name<tag>());
        CHECK_ITERABLE_APPROX(get_element_tag(tag{}), (get<tag>(tov_vars)));
      });
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GhValenciaDivClean.SetInitialData",
                  "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();
  EquationsOfState::register_derived_with_charm();
  test_set_initial_data(
      NumericInitialData{
          "TestInitialData.h5",
          "VolumeData",
          0.,
          false,
          gh::NumericInitialData::GhVars{"CustomSpacetimeMetric", "CustomPi"},
          {"CustomRho", "CustomUi", "CustomYe", "CustomB"},
          1.e-14},
      "NumericInitialData:\n"
      "  FileGlob: TestInitialData.h5\n"
      "  Subgroup: VolumeData\n"
      "  ObservationValue: 0.\n"
      "  Interpolate: False\n"
      "  GhVariables:\n"
      "    SpacetimeMetric: CustomSpacetimeMetric\n"
      "    Pi: CustomPi\n"
      "  HydroVariables:\n"
      "    RestMassDensity: CustomRho\n"
      "    LowerSpatialFourVelocity: CustomUi\n"
      "    ElectronFraction: CustomYe\n"
      "    MagneticField: CustomB\n"
      "  DensityCutoff: 1.e-14",
      true);
  test_set_initial_data(
      NumericInitialData{"TestInitialData.h5",
                         "VolumeData",
                         0.,
                         false,
                         gh::NumericInitialData::AdmVars{
                             "CustomSpatialMetric", "CustomLapse",
                             "CustomShift", "CustomExtrinsicCurvature"},
                         {"CustomRho", "CustomUi", 0.15, 0.},
                         1.e-14},
      "NumericInitialData:\n"
      "  FileGlob: TestInitialData.h5\n"
      "  Subgroup: VolumeData\n"
      "  ObservationValue: 0.\n"
      "  Interpolate: False\n"
      "  GhVariables:\n"
      "    SpatialMetric: CustomSpatialMetric\n"
      "    Lapse: CustomLapse\n"
      "    Shift: CustomShift\n"
      "    ExtrinsicCurvature: CustomExtrinsicCurvature\n"
      "  HydroVariables:\n"
      "    RestMassDensity: CustomRho\n"
      "    LowerSpatialFourVelocity: CustomUi\n"
      "    ElectronFraction: 0.15\n"
      "    MagneticField: 0.\n"
      "  DensityCutoff: 1.e-14",
      true);
  test_set_initial_data(
      TovStar{1.e-3, std::make_unique<EquationsOfState::PolytropicFluid<true>>(
                         100., 2.)},
      "TovStar:\n"
      "  CentralDensity: 1.e-3\n"
      "  EquationOfState:\n"
      "    PolytropicFluid:\n"
      "      PolytropicConstant: 100.\n"
      "      PolytropicExponent: 2.\n"
      "  Coordinates: Schwarzschild\n",
      false);
}

}  // namespace
}  // namespace grmhd::GhValenciaDivClean::Actions

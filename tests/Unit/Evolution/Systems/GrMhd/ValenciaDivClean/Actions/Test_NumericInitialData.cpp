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
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Actions/NumericInitialData.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::ValenciaDivClean::Actions {
namespace {

struct TestOptionGroup {
  using group = importers::OptionTags::Group;
  static constexpr Options::String help = "halp";
};

using all_hydro_vars = hydro::grmhd_tags<DataVector>;

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
              ::Tags::Variables<all_hydro_vars>,
              hydro::Tags::EquationOfState<
                  std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>>,
              gr::Tags::InverseSpatialMetric<3, Frame::Inertial,
                                             DataVector>>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<grmhd::ValenciaDivClean::Actions::ReadNumericInitialData<
                         TestOptionGroup>,
                     grmhd::ValenciaDivClean::Actions::SetNumericInitialData<
                         TestOptionGroup>>>>;
};

struct MockReadVolumeData {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      DataBox& /*box*/, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      tuples::tagged_tuple_from_typelist<
          db::wrap_tags_in<importers::Tags::Selected, detail::all_numeric_vars>>
          selected_fields) {
    const auto selected_vars =
        get<detail::Tags::NumericInitialDataVariables<TestOptionGroup>>(cache);
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
          metavariables::volume_dim, TestOptionGroup, detail::all_numeric_vars,
          MockElementArray<Metavariables>>>;
  using with_these_simple_actions = tmpl::list<MockReadVolumeData>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 3;
  using component_list = tmpl::list<MockElementArray<Metavariables>,
                                    MockVolumeDataReader<Metavariables>>;
};

void test_numeric_initial_data(
    const detail::PrimitiveVarsOptions& selected_vars,
    const std::string& option_string) {
  CHECK(TestHelpers::test_option_tag<
            detail::OptionTags::NumericInitialDataVariables<TestOptionGroup>>(
            option_string) == selected_vars);

  using reader_component = MockVolumeDataReader<Metavariables>;
  using element_array = MockElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{tuples::TaggedTuple<
      importers::Tags::FileGlob<TestOptionGroup>,
      importers::Tags::Subgroup<TestOptionGroup>,
      importers::Tags::ObservationValue<TestOptionGroup>,
      importers::Tags::EnableInterpolation<TestOptionGroup>,
      detail::Tags::NumericInitialDataVariables<TestOptionGroup>,
      detail::Tags::DensityCutoff<TestOptionGroup>>{
      "TestInitialData.h5", "VolumeData", 0., false, selected_vars, 1.e-14}};

  // We get test data from a TOV solution
  RelativisticEuler::Solutions::TovStar tov_star{
      1.e-3,
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(100., 2.)};
  const double star_radius = tov_star.radial_solution().outer_radius();
  const auto& eos = tov_star.equation_of_state();

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
              star_radius / 2., star_radius * 2., 1., 1., {}, true});
  const auto logical_coords = logical_coordinates(mesh);
  const auto coords = map(logical_coords);
  using spatial_metric_tag =
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>;
  const auto spatial_metric = get<spatial_metric_tag>(
      tov_star.variables(coords, 0., tmpl::list<spatial_metric_tag>{}));
  const auto inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), element_id,
      {Variables<all_hydro_vars>{num_points}, eos.get_clone(),
       inv_spatial_metric});

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

  // Insert TOV data into the inbox
  auto& inbox = ActionTesting::get_inbox_tag<
      element_array,
      importers::Tags::VolumeData<TestOptionGroup, detail::all_numeric_vars>,
      Metavariables>(make_not_null(&runner), element_id)[0];
  auto tov_vars = tov_star.variables(coords, 0., all_hydro_vars{});
  get<hydro::Tags::RestMassDensity<DataVector>>(inbox) =
      get<hydro::Tags::RestMassDensity<DataVector>>(tov_vars);
  const auto& W = get<hydro::Tags::LorentzFactor<DataVector>>(tov_vars);
  auto& u_i = get<hydro::Tags::LowerSpatialFourVelocity<DataVector, 3>>(inbox);
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

  // Override variables if constant value is specified in options
  const auto selected_electron_fraction =
      get<detail::OptionTags::VarName<hydro::Tags::ElectronFraction<DataVector>,
                                      std::bool_constant<false>>>(
          selected_vars);
  if (std::holds_alternative<double>(selected_electron_fraction)) {
    get<hydro::Tags::ElectronFraction<DataVector>>(tov_vars) =
        make_with_value<Scalar<DataVector>>(
            W, std::get<double>(selected_electron_fraction));
  }
  const auto selected_magnetic_field =
      get<detail::OptionTags::VarName<hydro::Tags::MagneticField<DataVector, 3>,
                                      std::bool_constant<false>>>(
          selected_vars);
  if (std::holds_alternative<double>(selected_magnetic_field)) {
    get<hydro::Tags::MagneticField<DataVector, 3>>(tov_vars) =
        make_with_value<tnsr::I<DataVector, 3>>(
            W, std::get<double>(selected_magnetic_field));
  }

  // SetNumericInitialData
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  // Check result
  tmpl::for_each<all_hydro_vars>(
      [&get_element_tag, &tov_vars](const auto tag_v) {
        using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
        CAPTURE(pretty_type::name<tag>());
        CHECK_ITERABLE_APPROX(get_element_tag(tag{}), (get<tag>(tov_vars)));
      });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ValenciaDivClean.NumericInitialData",
                  "[Unit][Evolution]") {
  EquationsOfState::register_derived_with_charm();
  test_numeric_initial_data(
      detail::PrimitiveVarsOptions{"CustomRho", "CustomUi", "CustomYe",
                                   "CustomB"},
      "RestMassDensity: CustomRho\n"
      "LowerSpatialFourVelocity: CustomUi\n"
      "ElectronFraction: CustomYe\n"
      "MagneticField: CustomB");
  test_numeric_initial_data(
      detail::PrimitiveVarsOptions{"CustomRho", "CustomUi", 0.15, 0.},
      "RestMassDensity: CustomRho\n"
      "LowerSpatialFourVelocity: CustomUi\n"
      "ElectronFraction: 0.15\n"
      "MagneticField: 0.");
}

}  // namespace grmhd::ValenciaDivClean::Actions

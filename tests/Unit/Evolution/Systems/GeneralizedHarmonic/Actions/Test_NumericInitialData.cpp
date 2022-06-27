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
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/NumericInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.tpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace GeneralizedHarmonic {
namespace {

struct TestOptionGroup {
  using group = importers::OptionTags::Group;
  static constexpr Options::String help = "halp";
};

using gh_system_vars = GeneralizedHarmonic::System<3>::variables_tag::tags_list;

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
                         domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                                       Frame::Inertial>>>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<GeneralizedHarmonic::Actions::ReadNumericInitialData<
                         TestOptionGroup>,
                     GeneralizedHarmonic::Actions::SetNumericInitialData<
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
    if (std::holds_alternative<detail::GeneralizedHarmonic>(selected_vars)) {
      CHECK(get<importers::Tags::Selected<
                gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>>(
                selected_fields) == "CustomSpacetimeMetric");
      CHECK(get<importers::Tags::Selected<Tags::Pi<3, Frame::Inertial>>>(
                selected_fields) == "CustomPi");
      CHECK(get<importers::Tags::Selected<Tags::Phi<3, Frame::Inertial>>>(
                selected_fields) == "CustomPhi");
      CHECK_FALSE(get<importers::Tags::Selected<
                      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
          selected_fields));
      CHECK_FALSE(get<importers::Tags::Selected<gr::Tags::Lapse<DataVector>>>(
          selected_fields));
      CHECK_FALSE(get<importers::Tags::Selected<
                      gr::Tags::Shift<3, Frame::Inertial, DataVector>>>(
          selected_fields));
      CHECK_FALSE(
          get<importers::Tags::Selected<
              gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>>(
              selected_fields));
    } else if (std::holds_alternative<detail::Adm>(selected_vars)) {
      CHECK(get<importers::Tags::Selected<
                gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
                selected_fields) == "CustomSpatialMetric");
      CHECK(get<importers::Tags::Selected<gr::Tags::Lapse<DataVector>>>(
                selected_fields) == "CustomLapse");
      CHECK(get<importers::Tags::Selected<
                gr::Tags::Shift<3, Frame::Inertial, DataVector>>>(
                selected_fields) == "CustomShift");
      CHECK(get<importers::Tags::Selected<
                gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>>(
                selected_fields) == "CustomExtrinsicCurvature");
      CHECK_FALSE(
          get<importers::Tags::Selected<
              gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>>(
              selected_fields));
      CHECK_FALSE(get<importers::Tags::Selected<Tags::Pi<3, Frame::Inertial>>>(
          selected_fields));
      CHECK_FALSE(get<importers::Tags::Selected<Tags::Phi<3, Frame::Inertial>>>(
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
          TestOptionGroup, detail::all_numeric_vars,
          MockElementArray<Metavariables>>>;
  using with_these_simple_actions = tmpl::list<MockReadVolumeData>;
};

struct Metavariables {
  using component_list = tmpl::list<MockElementArray<Metavariables>,
                                    MockVolumeDataReader<Metavariables>>;
};

}  // namespace

void test_numeric_initial_data(
    const typename detail::Tags::NumericInitialDataVariables<
        TestOptionGroup>::type& selected_vars,
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
      detail::Tags::NumericInitialDataVariables<TestOptionGroup>>{
      "TestInitialData.h5", "VolumeData", 0., selected_vars}};

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
       tnsr::iaa<DataVector, 3>{}, mesh, inv_jacobian});

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

  // Insert KerrSchild data into the inbox
  auto& inbox = ActionTesting::get_inbox_tag<
      element_array,
      importers::Tags::VolumeData<TestOptionGroup, detail::all_numeric_vars>,
      Metavariables>(make_not_null(&runner), element_id)[0];
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild> kerr{
      1., {{0., 0., 0.}}, {{0., 0., 0.}}};
  const auto kerr_gh_vars = kerr.variables(coords, 0., gh_system_vars{});
  if (std::holds_alternative<detail::GeneralizedHarmonic>(selected_vars)) {
    get<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>(inbox) =
        get<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>(
            kerr_gh_vars);
    get<Tags::Pi<3, Frame::Inertial>>(inbox) =
        get<Tags::Pi<3, Frame::Inertial>>(kerr_gh_vars);
    get<Tags::Phi<3, Frame::Inertial>>(inbox) =
        get<Tags::Phi<3, Frame::Inertial>>(kerr_gh_vars);
  } else if (std::holds_alternative<detail::Adm>(selected_vars)) {
    const auto kerr_adm_vars = kerr.variables(
        coords, 0.,
        tmpl::list<
            gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
            gr::Tags::Lapse<DataVector>,
            gr::Tags::Shift<3, Frame::Inertial, DataVector>,
            gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>{});
    get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(inbox) =
        get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
            kerr_adm_vars);
    get<gr::Tags::Lapse<DataVector>>(inbox) =
        get<gr::Tags::Lapse<DataVector>>(kerr_adm_vars);
    get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(inbox) =
        get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(kerr_adm_vars);
    get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>(inbox) =
        get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>(
            kerr_adm_vars);
  } else {
    REQUIRE(false);
  }

  // SetNumericInitialData
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  // Check result. These variables are not particularly precise because we are
  // taking numerical derivatives on a fairly coarse wedge-shaped grid.
  Approx custom_approx = Approx::custom().epsilon(1.e-3).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_element_tag(
          gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>{}),
      (get<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>(
          kerr_gh_vars)),
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_element_tag(Tags::Pi<3, Frame::Inertial>{}),
      (get<Tags::Pi<3, Frame::Inertial>>(kerr_gh_vars)), custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_element_tag(Tags::Phi<3, Frame::Inertial>{}),
      (get<Tags::Phi<3, Frame::Inertial>>(kerr_gh_vars)), custom_approx);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Gh.NumericInitialData",
                  "[Unit][Evolution]") {
  test_numeric_initial_data(
      detail::GeneralizedHarmonic{"CustomSpacetimeMetric", "CustomPi",
                                  "CustomPhi"},
      "SpacetimeMetric: CustomSpacetimeMetric\n"
      "Pi: CustomPi\n"
      "Phi: CustomPhi");
  test_numeric_initial_data(
      detail::Adm{"CustomSpatialMetric", "CustomLapse", "CustomShift",
                  "CustomExtrinsicCurvature"},
      "SpatialMetric: CustomSpatialMetric\n"
      "Lapse: CustomLapse\n"
      "Shift: CustomShift\n"
      "ExtrinsicCurvature: CustomExtrinsicCurvature");
}

}  // namespace GeneralizedHarmonic

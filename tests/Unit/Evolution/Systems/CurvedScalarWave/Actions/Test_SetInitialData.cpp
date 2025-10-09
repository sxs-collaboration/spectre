// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Actions/SetInitialData.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Factory.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/PureSphericalHarmonic.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace CurvedScalarWave::Actions {
namespace {

using all_scalar_vars =
    tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
               CurvedScalarWave::Tags::Phi<3>>;

template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<
              ::Tags::Variables<all_scalar_vars>,
              domain::Tags::Coordinates<3, Frame::Inertial>, ::Tags::Time,
              gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<CurvedScalarWave::Actions::SetInitialData,
                     CurvedScalarWave::Actions::ReceiveNumericInitialData>>>;
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
    CHECK(get<importers::Tags::Selected<CurvedScalarWave::Tags::Phi<3>>>(
              selected_fields) == "CustomPhi");
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
    using factory_classes = tmpl::map<
        tmpl::pair<
            evolution::initial_data::InitialData,
            tmpl::list<NumericInitialData,
                       CurvedScalarWave::AnalyticData::PureSphericalHarmonic,
                       ScalarWave::Solutions::PlaneWave<3>>>,
        tmpl::pair<::MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>>;
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
    } else if (const auto* const pure_spherical_harmonic =
                   dynamic_cast<const CurvedScalarWave::AnalyticData::
                                    PureSphericalHarmonic*>(&initial_data)) {
      CHECK(dynamic_cast<
                const CurvedScalarWave::AnalyticData::PureSphericalHarmonic&>(
                *created) == *pure_spherical_harmonic);
    } else if (const auto* const plane_wave =
                   dynamic_cast<const ScalarWave::Solutions::PlaneWave<3>*>(
                       &initial_data)) {
      CHECK(dynamic_cast<const ScalarWave::Solutions::PlaneWave<3>&>(
                *created) == *plane_wave);
    } else {
      FAIL("Unexpected initial data type under test.");
    }
  }

  using reader_component = MockVolumeDataReader<Metavariables>;
  using element_array = MockElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      initial_data.get_clone()};

  ActionTesting::emplace_nodegroup_component<reader_component>(
      make_not_null(&runner));

  const ElementId<3> element_id{0};
  const double initial_time = 0.;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 10.0);
  const size_t num_points = 100;
  const auto shift = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&gen), make_not_null(&dist), DataVector{num_points});
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), make_not_null(&dist), DataVector{num_points});
  const auto coords = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&gen), make_not_null(&dist), DataVector{num_points});

  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), element_id,
      {Variables<all_scalar_vars>{num_points}, coords, initial_time, lapse,
       shift});

  const auto get_element_tag = [&runner,
                                &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  if (is_numeric) {
    REQUIRE_FALSE(ActionTesting::next_action_if_ready<element_array>(
        make_not_null(&runner), element_id));

    ActionTesting::invoke_queued_simple_action<reader_component>(
        make_not_null(&runner), 0);

    const auto& numeric_initial_data =
        dynamic_cast<const NumericInitialData&>(initial_data);
    auto& inbox = ActionTesting::get_inbox_tag<
        element_array,
        importers::Tags::VolumeData<NumericInitialData::all_vars>,
        Metavariables>(make_not_null(&runner),
                       element_id)[numeric_initial_data.volume_data_id()];

    const CurvedScalarWave::AnalyticData::PureSphericalHarmonic source_data{
        2.0, 1.0, {1, 0}};
    const auto source_vars = source_data.variables(coords, all_scalar_vars{});
    get<CurvedScalarWave::Tags::Psi>(inbox) =
        get<CurvedScalarWave::Tags::Psi>(source_vars);
    get<CurvedScalarWave::Tags::Pi>(inbox) =
        get<CurvedScalarWave::Tags::Pi>(source_vars);
    get<CurvedScalarWave::Tags::Phi<3>>(inbox) =
        get<CurvedScalarWave::Tags::Phi<3>>(source_vars);

    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);

    tmpl::for_each<all_scalar_vars>(
        [&get_element_tag, &source_vars](const auto tag_v) {
          using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
          CHECK_ITERABLE_APPROX(get_element_tag(tag{}), get<tag>(source_vars));
        });
    return;
  }

  const auto& lapse_in_box = get_element_tag(gr::Tags::Lapse<DataVector>{});
  const auto& shift_in_box = get_element_tag(gr::Tags::Shift<DataVector, 3>{});

  const auto analytic_verifier = [&coords, &initial_time, &get_element_tag,
                                  &lapse_in_box,
                                  &shift_in_box](const auto& specific_data) {
    using data_type = std::decay_t<decltype(specific_data)>;
    if constexpr (tmpl::list_contains_v<typename data_type::tags,
                                        CurvedScalarWave::Tags::Psi>) {
      const auto curved_initial_data = evolution::Initialization::initial_data(
          specific_data, coords, initial_time,
          typename CurvedScalarWave::System<3>::variables_tag::tags_list{});
      CHECK_ITERABLE_APPROX(
          get_element_tag(CurvedScalarWave::Tags::Psi{}),
          get<CurvedScalarWave::Tags::Psi>(curved_initial_data));
      CHECK_ITERABLE_APPROX(
          get_element_tag(CurvedScalarWave::Tags::Pi{}),
          get<CurvedScalarWave::Tags::Pi>(curved_initial_data));
      CHECK_ITERABLE_APPROX(
          get_element_tag(CurvedScalarWave::Tags::Phi<3>{}),
          get<CurvedScalarWave::Tags::Phi<3>>(curved_initial_data));
    } else {
      const auto flat_initial_data = evolution::Initialization::initial_data(
          specific_data, coords, initial_time,
          typename ScalarWave::System<3>::variables_tag::tags_list{});
      const auto shift_dot_phi = dot_product(
          shift_in_box, get<ScalarWave::Tags::Phi<3>>(flat_initial_data));
      Scalar<DataVector> expected_pi{};
      get(expected_pi) = (get(shift_dot_phi) +
                          get(get<ScalarWave::Tags::Pi>(flat_initial_data))) /
                         get(lapse_in_box);
      CHECK_ITERABLE_APPROX(get_element_tag(CurvedScalarWave::Tags::Psi{}),
                            get<ScalarWave::Tags::Psi>(flat_initial_data));
      CHECK_ITERABLE_APPROX(get_element_tag(CurvedScalarWave::Tags::Phi<3>{}),
                            get<ScalarWave::Tags::Phi<3>>(flat_initial_data));
      CHECK_ITERABLE_APPROX(get_element_tag(CurvedScalarWave::Tags::Pi{}),
                            expected_pi);
    }
  };

  call_with_dynamic_type<
      void, tmpl::list<CurvedScalarWave::AnalyticData::PureSphericalHarmonic,
                       ScalarWave::Solutions::PlaneWave<3>>>(
      &initial_data, [&analytic_verifier](const auto* const analytic_ptr) {
        analytic_verifier(*analytic_ptr);
      });
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.SetInitialData",
                  "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();
  test_set_initial_data(
      NumericInitialData{"TestInitialData.h5",
                         "VolumeData",
                         0.,
                         {1.0e-9},
                         false,
                         {"CustomPsi", "CustomPi", "CustomPhi"}},
      "NumericInitialData:\n"
      "  FileGlob: TestInitialData.h5\n"
      "  Subgroup: VolumeData\n"
      "  ObservationValue: 0.\n"
      "  ObservationValueEpsilon: 1e-9\n"
      "  ElementsAreIdentical: False\n"
      "  Variables:\n"
      "    Psi: CustomPsi\n"
      "    Pi: CustomPi\n"
      "    Phi: CustomPhi",
      true);
  test_set_initial_data(
      CurvedScalarWave::AnalyticData::PureSphericalHarmonic{2.0, 1.0, {2, -1}},
      "PureSphericalHarmonic:\n"
      "  Radius: 2.0 \n"
      "  Width: 1.0 \n"
      "  Mode: [2, -1]",
      false);
  test_set_initial_data(
      ScalarWave::Solutions::PlaneWave<3>{
          {{1.5, -7.2, 2.7}},
          {{2.4, -4.8, 8.4}},
          std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(3)},
      "PlaneWave:\n"
      "  WaveVector: [1.5, -7.2, 2.7]\n"
      "  Center: [2.4, -4.8, 8.4]\n"
      "  Profile:\n"
      "    PowX:\n"
      "      Power: 3",
      false);
}

}  // namespace CurvedScalarWave::Actions

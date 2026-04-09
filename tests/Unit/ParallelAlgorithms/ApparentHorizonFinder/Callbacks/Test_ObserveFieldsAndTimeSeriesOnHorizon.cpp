// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/IO/VolumeData.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/Surfaces/TestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ObserveFieldsOnHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ObserveTimeSeriesOnHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Component.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Initialization.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t low_l_max = 3;
constexpr size_t high_l_max = 16;

template <typename Fr>
Variables<ah::vars_to_interpolate_to_target<3, Fr>> make_vars(
    const ylm::Strahlkorper<Fr>& horizon, const LinkedMessageId<double>& time,
    const double mass) {
  const gr::Solutions::KerrSchild solution{mass, std::array{0.0, 0.0, 0.0},
                                           std::array{0.0, 0.0, 0.0}};
  const auto coords = ylm::cartesian_coords(horizon);
  const auto vars = solution.variables(
      coords, time.id,
      tmpl::pop_back<ah::vars_to_interpolate_to_target<3, Fr>>{});
  const size_t size =
      get<0, 0>(get<gr::Tags::SpatialMetric<DataVector, 3, Fr>>(vars)).size();
  Variables<ah::vars_to_interpolate_to_target<3, Fr>> result{size, 0.0};
  result.assign_subset(vars);
  get<gr::Tags::SpatialRicci<DataVector, 3, Fr>>(result) =
      TestHelpers::Schwarzschild::spatial_ricci(coords, mass);

  return result;
}

void check_ylm_data(
    const std::string& h5_file_name,
    const ylm::Strahlkorper<Frame::Inertial>& expected_surface) {
  const std::vector<std::string> ylm_expected_legend{
      "Time",
      "InertialExpansionCenter_x",
      "InertialExpansionCenter_y",
      "InertialExpansionCenter_z",
      "Lmax",
      "coef(0,0)",
      "coef(1,-1)",
      "coef(1,0)",
      "coef(1,1)",
      "coef(2,-2)",
      "coef(2,-1)",
      "coef(2,0)",
      "coef(2,1)",
      "coef(2,2)",
      "coef(3,-3)",
      "coef(3,-2)",
      "coef(3,-1)",
      "coef(3,0)",
      "coef(3,1)",
      "coef(3,2)",
      "coef(3,3)"};
  const size_t expected_num_columns = ylm_expected_legend.size();

  // Check that the H5 file was written correctly.
  const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);
  const auto& ylm_dat_file = file.get<h5::Dat>("/HorizonD_Ylm");
  const Matrix ylm_written_data = ylm_dat_file.get_data();
  const auto& ylm_written_legend = ylm_dat_file.get_legend();

  CHECK(ylm_written_legend.size() == expected_num_columns);
  CHECK(ylm_written_data.columns() == expected_num_columns);
  CHECK(ylm_written_legend == ylm_expected_legend);

  // Center is origin
  std::vector<double> ylm_expected_data{2.0, 0.0, 0.0, 0.0, low_l_max};

  ylm::SpherepackIterator iter(low_l_max, low_l_max);
  for (size_t l = 0; l <= low_l_max; l++) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
      iter.set(l, m);
      ylm_expected_data.push_back(expected_surface.coefficients()[iter()]);
    }
  }

  ASSERT(ylm_expected_data.size() == expected_num_columns,
         "The size of the constructed test Ylm legend ("
             << expected_num_columns
             << ") and the number of columns in the constructed test Ylm data ("
             << ylm_expected_data.size() << ") do not match.");

  for (size_t i = 0; i < expected_num_columns; i++) {
    CHECK(ylm_written_data(0, i) == ylm_expected_data[i]);
  }
}

void check_surface_volume_data(
    const std::string& surfaces_file_name,
    const ylm::Strahlkorper<Frame::Inertial>& strahlkorper,
    const LinkedMessageId<double>& time, const double mass) {
  const ylm::Spherepack& ylm = strahlkorper.ylm_spherepack();
  const std::vector<size_t> extents{
      {ylm.physical_extents()[0], ylm.physical_extents()[1]}};
  const std::string grid_name{"HorizonD"};

  const auto coords = ylm::cartesian_coords(strahlkorper);
  const auto all_vars = make_vars(strahlkorper, time, mass);
  const std::vector<DataVector> tensor_and_coord_data{
      get<0>(coords), get<1>(coords), get<2>(coords),
      DataVector{get<0>(coords).size(), 0.5 / square(mass)}};
  const std::vector<TensorComponent> tensor_components{
      {grid_name + "/InertialCoordinates_x", tensor_and_coord_data[0]},
      {grid_name + "/InertialCoordinates_y", tensor_and_coord_data[1]},
      {grid_name + "/InertialCoordinates_z", tensor_and_coord_data[2]},
      {grid_name + "/RicciScalar", tensor_and_coord_data[3]}};

  const std::vector<Spectral::Basis> bases{2,
                                           Spectral::Basis::SphericalHarmonic};
  const std::vector<Spectral::Quadrature> quadratures{
      {Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}};
  const observers::ObservationId observation_id{2., "/HorizonD.vol"};
  TestHelpers::io::VolumeData::check_volume_data(
      surfaces_file_name, 0, grid_name, observation_id.hash(),
      observation_id.value(), std::nullopt, tensor_and_coord_data, {grid_name},
      {bases}, {quadratures}, {extents},
      {"InertialCoordinates_x"s, "InertialCoordinates_y"s,
       "InertialCoordinates_z"s, "RicciScalar"s},
      {{0, 1, 2, 3}}, std::optional{1.0e-10});
}

template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<observers::Tags::ReductionFileName,
                                             observers::Tags::SurfaceFileName>;
  using simple_tags =
      typename observers::Actions::InitializeWriter<Metavariables>::simple_tags;
  using compute_tags = typename observers::Actions::InitializeWriter<
      Metavariables>::compute_tags;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<observers::Actions::InitializeWriter<Metavariables>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked = observers::ObserverWriter<Metavariables>;
};

template <typename Metavariables, typename HorizonMetavars>
struct MockComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked = ah::Component<Metavariables, HorizonMetavars>;

  using frame = typename HorizonMetavars::frame;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<
          typename ah::Initialize<HorizonMetavars>::simple_tags,
          typename ah::Initialize<HorizonMetavars>::compute_tags>>>>;
};

struct MockMetavariables {
  template <typename Fr>
  struct HorizonAB : tt::ConformsTo<ah::protocols::HorizonMetavars> {
    using temporal_id_tag = ::Tags::TimeAndPrevious<0>;
    using frame = Fr;

    using horizon_find_callbacks =
        tmpl::list<ah::callbacks::ObserveTimeSeriesOnHorizon<
            tmpl::list<gr::surfaces::Tags::AreaCompute<frame>>, HorizonAB>>;
    using horizon_find_failure_callbacks = tmpl::list<>;

    using compute_tags_on_element = tmpl::list<>;

    static constexpr ah::Destination destination =
        ah::Destination::ControlSystem;

    static std::string name() {
      return "Horizon"s + (std::is_same_v<Fr, Frame::Inertial> ? "A"s : "B"s);
    }
  };

  using HorizonA = HorizonAB<Frame::Inertial>;
  using HorizonB = HorizonAB<Frame::Grid>;

  struct HorizonC : tt::ConformsTo<ah::protocols::HorizonMetavars> {
    using temporal_id_tag = ::Tags::TimeAndPrevious<0>;
    using frame = Frame::Inertial;

    using horizon_find_callbacks =
        tmpl::list<ah::callbacks::ObserveTimeSeriesOnHorizon<
            tmpl::list<ylm::Tags::MaxRicciScalarCompute>, HorizonC>>;
    using horizon_find_failure_callbacks = tmpl::list<>;

    using compute_tags_on_element = tmpl::list<>;

    static constexpr ah::Destination destination =
        ah::Destination::ControlSystem;

    static std::string name() { return "HorizonC"; }
  };

  struct HorizonD : tt::ConformsTo<ah::protocols::HorizonMetavars> {
    using temporal_id_tag = ::Tags::TimeAndPrevious<0>;
    using frame = Frame::Inertial;

    using horizon_find_callbacks =
        tmpl::list<ah::callbacks::ObserveFieldsOnHorizon<
            tmpl::list<ylm::Tags::RicciScalar>, HorizonD>>;
    using horizon_find_failure_callbacks = tmpl::list<>;

    using compute_tags_on_element = tmpl::list<>;

    static constexpr ah::Destination destination = ah::Destination::Observation;

    static std::string name() { return "HorizonD"; }
  };

  using observed_reduction_data_tags = tmpl::list<>;

  using component_list = tmpl::list<MockObserverWriter<MockMetavariables>,
                                    MockComponent<MockMetavariables, HorizonA>,
                                    MockComponent<MockMetavariables, HorizonB>,
                                    MockComponent<MockMetavariables, HorizonC>,
                                    MockComponent<MockMetavariables, HorizonD>>;
};

void run_test() {
  const auto remove_file_if_exists = [](const std::string& file_name) {
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
  };

  // Check if either file generated by this test exists and remove them
  // if so. Check for both files existing before the test runs, since
  // both files get written when evaluating the list of post interpolation
  // callbacks below.
  const std::string h5_file_prefix = "Test_ObserveFieldsAndTimeSeriesOnHorizon";
  const std::string h5_file_name = h5_file_prefix + ".h5";
  remove_file_if_exists(h5_file_name);
  const std::string surfaces_file_prefix = "Surfaces";
  const std::string surfaces_file_name = surfaces_file_prefix + ".h5";
  remove_file_if_exists(surfaces_file_name);

  // Test That ObserveTimeSeriesOnSurface indeed does conform to its protocol
  using metavars = MockMetavariables;
  (void)metavars::HorizonA::destination;
  (void)metavars::HorizonB::destination;
  (void)metavars::HorizonC::destination;
  (void)metavars::HorizonD::destination;
  using horizon_a_callback =
      tmpl::front<typename metavars::HorizonA::horizon_find_callbacks>;
  using horizon_b_callback =
      tmpl::front<typename metavars::HorizonB::horizon_find_callbacks>;
  using horizon_c_callback =
      tmpl::front<typename metavars::HorizonC::horizon_find_callbacks>;
  using horizon_d_callback =
      tmpl::front<typename metavars::HorizonD::horizon_find_callbacks>;
  using protocol = ah::protocols::Callback;
  static_assert(tt::assert_conforms_to_v<horizon_a_callback, protocol>);
  static_assert(tt::assert_conforms_to_v<horizon_b_callback, protocol>);
  static_assert(tt::assert_conforms_to_v<horizon_c_callback, protocol>);
  static_assert(tt::assert_conforms_to_v<horizon_d_callback, protocol>);

  using horizon_a_component = MockComponent<metavars, metavars::HorizonA>;
  using horizon_b_component = MockComponent<metavars, metavars::HorizonB>;
  using horizon_c_component = MockComponent<metavars, metavars::HorizonC>;
  using horizon_d_component = MockComponent<metavars, metavars::HorizonD>;
  using obs_writer = MockObserverWriter<metavars>;

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {h5_file_prefix, surfaces_file_prefix}};

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);

  const LinkedMessageId<double> time{2.0, {1.0}};
  const double mass = 0.5;
  const double radius = 2.0 * mass;
  const std::array initial_center{0.0, 0.0, 0.0};

  const auto initialize_component =
      [&]<typename Component>(Component /*component_v*/, const size_t l_max) {
        using Fr = typename Component::frame;

        const ylm::Strahlkorper<Fr> horizon{l_max, radius, initial_center};

        ActionTesting::emplace_array_component<Component>(
            make_not_null(&runner), ActionTesting::NodeId{0},
            ActionTesting::LocalCoreId{0}, 0);

        auto& box =
            ActionTesting::get_databox<Component>(make_not_null(&runner), 0);

        db::mutate<ylm::Tags::Strahlkorper<Fr>, ah::Tags::CurrentTime,
                   ::Tags::Variables<ah::vars_to_interpolate_to_target<3, Fr>>>(
            [&](const gsl::not_null<ylm::Strahlkorper<Fr>*> strahlkorper,
                const gsl::not_null<std::optional<LinkedMessageId<double>>*>
                    current_time,
                const gsl::not_null<
                    ::Variables<ah::vars_to_interpolate_to_target<3, Fr>>*>
                    vars) {
              (*strahlkorper) = horizon;
              (*current_time) = time;
              (*vars) = make_vars(horizon, time, mass);
            },
            make_not_null(&box));
      };

  const ylm::Strahlkorper<Frame::Inertial> low_l_horizon{low_l_max, radius,
                                                         initial_center};
  const ylm::Strahlkorper<Frame::Inertial> high_l_horizon{high_l_max, radius,
                                                          initial_center};

  initialize_component(horizon_a_component{}, low_l_max);
  initialize_component(horizon_b_component{}, low_l_max);
  initialize_component(horizon_c_component{}, high_l_max);
  initialize_component(horizon_d_component{}, low_l_max);

  ActionTesting::emplace_nodegroup_component<obs_writer>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<obs_writer>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const FastFlow::Status status = FastFlow::Status::AbsTol;
  auto& cache = ActionTesting::cache<obs_writer>(runner, 0_st);

  const auto& horizon_a_box =
      ActionTesting::get_databox<horizon_a_component>(runner, 0);
  const auto& horizon_b_box =
      ActionTesting::get_databox<horizon_b_component>(runner, 0);
  const auto& horizon_c_box =
      ActionTesting::get_databox<horizon_c_component>(runner, 0);
  const auto& horizon_d_box =
      ActionTesting::get_databox<horizon_d_component>(runner, 0);

  // Call callbacks
  horizon_a_callback::apply(horizon_a_box, cache, status);
  horizon_b_callback::apply(horizon_b_box, cache, status);
  horizon_c_callback::apply(horizon_c_box, cache, status);
  horizon_d_callback::apply(horizon_d_box, cache, status);

  // One for each ObserveTimeSeries and two for the ObserveFields
  REQUIRE(ActionTesting::number_of_queued_threaded_actions<obs_writer>(runner,
                                                                       0) == 5);

  for (size_t i = 0; i < 5; i++) {
    ActionTesting::invoke_queued_threaded_action<obs_writer>(
        make_not_null(&runner), 0);
  }

  const std::vector<double> expected_integral_ab{4.0 * M_PI};
  const std::vector<double> expected_integral_c{2.0 / square(radius)};
  const std::vector<std::string> expected_legend_ab{"Time", "Area"};
  const std::vector<std::string> expected_legend_c{"Time", "MaxRicciScalar"};

  // Check that the H5 file was written correctly.
  const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);
  auto check_file_contents =
      [&file](const std::vector<double>& expected_integral,
              const std::vector<std::string>& expected_legend,
              const std::string& group_name) {
        CAPTURE(group_name);
        CAPTURE(expected_legend);
        file.close_current_object();
        const auto& dat_file = file.get<h5::Dat>(group_name);
        const Matrix written_data = dat_file.get_data();
        const auto& written_legend = dat_file.get_legend();
        CHECK(written_legend == expected_legend);
        CHECK(2.0 == written_data(0, 0));
        // The interpolation is not perfect because we use too few grid points.
        const Approx custom_approx = Approx::custom().epsilon(1.e-4).scale(1.0);
        for (size_t i = 0; i < expected_integral.size(); ++i) {
          CAPTURE(i);
          CHECK(expected_integral[i] == custom_approx(written_data(0, i + 1)));
        }
      };
  check_file_contents(expected_integral_ab, expected_legend_ab, "/HorizonA");
  check_file_contents(expected_integral_ab, expected_legend_ab, "/HorizonB");
  check_file_contents(expected_integral_c, expected_legend_c, "/HorizonC");

  // Check that the Ylm data were written correctly
  // As this data depends only on the known target (a KerrHorizon) it
  // uses no interpolated data
  check_ylm_data(h5_file_name, low_l_horizon);

  remove_file_if_exists(h5_file_name);

  // Check that the Surfaces file contains the correct surface data
  check_surface_volume_data(surfaces_file_name, low_l_horizon, time, mass);

  remove_file_if_exists(surfaces_file_name);

  // Verify adaptive resolution writes all requested coefficients
  const std::string adaptive_horizon_reduction_file_prefix =
      "AdaptiveHorizonReduction";
  const std::string adaptive_horizon_surface_file_prefix =
      "AdaptiveHorizonSurfaceData";
  const std::string adaptive_horizon_reduction_file_name =
      adaptive_horizon_reduction_file_prefix + ".h5";
  const std::string adaptive_horizon_surface_file_name =
      adaptive_horizon_surface_file_prefix + ".h5";
  remove_file_if_exists(adaptive_horizon_reduction_file_name);
  remove_file_if_exists(adaptive_horizon_surface_file_name);

  constexpr size_t max_resolution_and_output_l = 6;
  constexpr size_t low_l = 4;
  constexpr size_t high_l = max_resolution_and_output_l;
  const double base_radius = 1.7;
  const std::array<double, 3> adaptive_center{{0.05, -0.01, 0.08}};

  MAKE_GENERATOR(adaptive_generator);
  const std::uniform_real_distribution<> radius_distribution{0.9 * base_radius,
                                                             1.1 * base_radius};
  const ylm::Spherepack spherepack_high(high_l, high_l);
  const auto radius_high = make_with_random_values<DataVector>(
      make_not_null(&adaptive_generator), radius_distribution,
      DataVector{spherepack_high.physical_size(), 0.0});
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper_high{
      high_l, high_l, radius_high, adaptive_center};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper_low{low_l, low_l,
                                                            strahlkorper_high};

  struct AdaptiveHorizonMetavariables {
    using observed_reduction_data_tags = tmpl::list<>;
    using const_global_cache_tags = tmpl::list<ah::Tags::LMax>;
    using component_list =
        tmpl::list<MockObserverWriter<AdaptiveHorizonMetavariables>>;
  };

  using ObsWriter = MockObserverWriter<AdaptiveHorizonMetavariables>;
  tuples::TaggedTuple<observers::Tags::ReductionFileName,
                      observers::Tags::SurfaceFileName, ah::Tags::LMax>
      adaptive_opts{adaptive_horizon_reduction_file_prefix,
                    adaptive_horizon_surface_file_prefix,
                    max_resolution_and_output_l};
  ActionTesting::MockRuntimeSystem<AdaptiveHorizonMetavariables>
      adaptive_runner{std::move(adaptive_opts)};

  ActionTesting::set_phase(make_not_null(&adaptive_runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_nodegroup_component<ObsWriter>(&adaptive_runner);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<ObsWriter>(make_not_null(&adaptive_runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&adaptive_runner),
                           Parallel::Phase::Testing);
  auto& adaptive_cache = ActionTesting::cache<ObsWriter>(adaptive_runner, 0_st);

  using HorizonMetavars = MockMetavariables::HorizonD;
  using Callback =
      ah::callbacks::ObserveFieldsOnHorizon<tmpl::list<>, HorizonMetavars>;

  const auto make_box =
      [](const ylm::Strahlkorper<Frame::Inertial>& strahlkorper,
         const LinkedMessageId<double>& time_id) {
        const auto coords = ylm::cartesian_coords(strahlkorper);
        return db::create<tmpl::list<
            ah::Tags::CurrentTime, ylm::Tags::Strahlkorper<Frame::Inertial>,
            ylm::Tags::CartesianCoords<Frame::Inertial>>>(
            std::optional<LinkedMessageId<double>>{time_id}, strahlkorper,
            coords);
      };

  const LinkedMessageId<double> time1{1.5, std::nullopt};
  const LinkedMessageId<double> time2{2.5, std::optional{time1.id}};

  auto low_box = make_box(strahlkorper_low, time1);
  auto high_box = make_box(strahlkorper_high, time2);

  Callback::apply(low_box, adaptive_cache, FastFlow::Status::AbsTol);
  Callback::apply(high_box, adaptive_cache, FastFlow::Status::AbsTol);

  while (ActionTesting::number_of_queued_threaded_actions<ObsWriter>(
             adaptive_runner, 0) > 0) {
    ActionTesting::invoke_queued_threaded_action<ObsWriter>(
        make_not_null(&adaptive_runner), 0);
  }

  const auto adaptive_horizon_reduction_file =
      h5::H5File<h5::AccessType::ReadOnly>(
          adaptive_horizon_reduction_file_name);
  const std::string surface_name = pretty_type::name<HorizonMetavars>();
  adaptive_horizon_reduction_file.close_current_object();
  const auto& ylm_dat = adaptive_horizon_reduction_file.get<h5::Dat>(
      std::string{"/"} + surface_name + "_Ylm");
  const Matrix ylm_data = ylm_dat.get_data();
  const auto& legend = ylm_dat.get_legend();
  const size_t expected_columns = 5 + square(max_resolution_and_output_l + 1);

  CHECK(ylm_data.rows() == 2);
  CHECK(ylm_data.columns() == expected_columns);
  CHECK(legend.size() == expected_columns);

  const auto check_row =
      [](const Matrix& data, const size_t row,
         const LinkedMessageId<double>& time_id,
         const ylm::Strahlkorper<Frame::Inertial>& strahlkorper,
         const size_t expected_l_max) {
        CHECK(data(row, 0) == time_id.id);
        const auto& expansion_center = strahlkorper.expansion_center();
        CHECK(data(row, 1) == expansion_center[0]);
        CHECK(data(row, 2) == expansion_center[1]);
        CHECK(data(row, 3) == expansion_center[2]);
        CHECK(data(row, 4) == expected_l_max);
        size_t column = 5;
        for (size_t l = 0; l <= max_resolution_and_output_l; ++l) {
          for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
            double expected_value = 0.0;
            if (l <= strahlkorper.l_max()) {
              ylm::SpherepackIterator iterator{strahlkorper.l_max(),
                                               strahlkorper.m_max()};
              iterator.set(l, m);
              expected_value = strahlkorper.coefficients()[iterator()];
            }
            CHECK(data(row, column) == expected_value);
            ++column;
          }
        }
      };

  check_row(ylm_data, 0, time1, strahlkorper_low, low_l);
  check_row(ylm_data, 1, time2, strahlkorper_high, high_l);

  remove_file_if_exists(adaptive_horizon_reduction_file_name);
  remove_file_if_exists(adaptive_horizon_surface_file_name);
}

SPECTRE_TEST_CASE(
    "Unit.ApparentHorizonFinder.ObserveFieldsAndTimeSeriesOnHorizon",
    "[ApparentHorizonFinder][Unit]") {
  run_test();
}
}  // namespace

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/Actions/WorldtubeBoundaryMocking.hpp"
#include "Helpers/Evolution/Systems/Cce/KleinGordonBoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {

namespace {

struct KleinGordonH5Metavariables {
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using klein_gordon_boundary_communication_tags =
      Tags::klein_gordon_worldtube_boundary_tags;
  using component_list = tmpl::list<
      mock_klein_gordon_h5_worldtube_boundary<KleinGordonH5Metavariables>>;

  static constexpr bool evolve_ccm = false;
};

// This function tests the action
// `InitializeWorldtubeBoundary<KleinGordonH5WorldtubeBoundary<Metavariables>>`.
// The action is responsible for initializing the tags
// `Metavariables::cce_boundary_communication_tags` and
// `Metavariables::klein_gordon_boundary_communication_tags` for the boundary
// component. The initialization process involves reading tensor and scalar data
// separately from an HDF5 file. This is handled by two managers:
// `Tags::H5WorldtubeBoundaryDataManager` and
// `Tags::KleinGordonH5WorldtubeBoundaryDataManager`.
//
// The function begins by generating and storing some scalar and tensor data
// into an HDF5 file named `filename` using `write_scalar_tensor_test_file`.
// Subsequently, it initializes a mocked worldtube boundary component that
// contains two optional tags: `Tags::H5WorldtubeBoundaryDataManager` and
// `Tags::KleinGordonH5WorldtubeBoundaryDataManager`.
// The function then invokes the action under examination within the
// initialize-action list to initialize the specified tags. Finally, it tests
// whether the tags are in the expected state.
template <typename Generator>
void test_klein_gordon_h5_initialization(const gsl::not_null<Generator*> gen) {
  using component =
      mock_klein_gordon_h5_worldtube_boundary<KleinGordonH5Metavariables>;
  const size_t l_max = 8;
  const size_t end_time = 100.0;
  const size_t start_time = 0.0;

  const size_t buffer_size = 8;
  const std::string filename =
      "InitializeKleinGordonWorldtubeBoundaryTest_CceR0100.h5";

  // create the test file, because on initialization the manager will need to
  // get basic data out of the file
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0;
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);
  TestHelpers::write_scalar_tensor_test_file(solution, filename, target_time,
                                             extraction_radius, frequency,
                                             amplitude, l_max);

  // tests start here
  ActionTesting::MockRuntimeSystem<KleinGordonH5Metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<KleinGordonH5Metavariables>>{
          l_max, extraction_radius, end_time, start_time}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0,
      Tags::H5WorldtubeBoundaryDataManager::create_from_options(
          l_max, filename, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u),
          false, false, std::optional<double>{}),
      Tags::KleinGordonH5WorldtubeBoundaryDataManager::create_from_options(
          l_max, filename, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u),
          std::optional<double>{}));

  // go through the initialization list
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);

  // The tensor part:
  //
  // check that the tensor data manager copied out of the databox has the
  // correct properties
  const auto& tensor_data_manager =
      ActionTesting::get_databox_tag<component,
                                     Tags::H5WorldtubeBoundaryDataManager>(
          runner, 0);
  CHECK(tensor_data_manager.get_l_max() == l_max);
  const auto tensor_time_span = tensor_data_manager.get_time_span();
  CHECK(tensor_time_span.first == 0);
  CHECK(tensor_time_span.second == 0);

  // check that the tensor Variables is in the expected state (here we just make
  // sure it has the right size - it shouldn't have been written to yet)
  const auto& tensor_variables = ActionTesting::get_databox_tag<
      component, ::Tags::Variables<typename KleinGordonH5Metavariables::
                                       cce_boundary_communication_tags>>(runner,
                                                                         0);
  CHECK(
      get(get<Tags::BoundaryValue<Tags::BondiBeta>>(tensor_variables)).size() ==
      Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  // then we repeat the tests for the scalar (Klein-Gordon) part
  const auto& scalar_data_manager = ActionTesting::get_databox_tag<
      component, Tags::KleinGordonH5WorldtubeBoundaryDataManager>(runner, 0);
  CHECK(scalar_data_manager.get_l_max() == l_max);
  const auto scalar_time_span = scalar_data_manager.get_time_span();
  CHECK(scalar_time_span.first == 0);
  CHECK(scalar_time_span.second == 0);

  // check that the scalar Variables is in the expected state (here we just make
  // sure it has the right size - it shouldn't have been written to yet)
  const auto& scalar_variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<typename KleinGordonH5Metavariables::
                            klein_gordon_boundary_communication_tags>>(runner,
                                                                       0);
  CHECK(get(get<Tags::BoundaryValue<Tags::KleinGordonPsi>>(scalar_variables))
            .size() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.InitializeKleinGordonWorldtubeBoundary",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_klein_gordon_h5_initialization(make_not_null(&gen));
}
}  // namespace Cce

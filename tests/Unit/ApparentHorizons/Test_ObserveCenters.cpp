// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>

#include "ApparentHorizons/ObserveCenters.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Matrix.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/IO/Observers/MockWriteReductionDataRow.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace {

// This is needed to template ObserveCenters, but it doesn't have to be an
// actual InterpolationTargetTag. This is only used for the name of the subfile
// written
struct AhA {};

struct TestMetavars {
  using const_global_cache_tags = tmpl::list<ah::Tags::ObserveCenters>;
  using component_list =
      tmpl::list<TestHelpers::observers::MockObserverWriter<TestMetavars>>;
};

using FoTPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;

template <typename Frame>
void test() {
  using observer_writer =
      TestHelpers::observers::MockObserverWriter<TestMetavars>;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> center_dist{-10.0, 10.0};

  // set up runner and stuff
  ActionTesting::MockRuntimeSystem<TestMetavars> runner{{true}};
  runner.set_phase(Parallel::Phase::Initialization);
  ActionTesting::emplace_nodegroup_component_and_initialize<observer_writer>(
      make_not_null(&runner), {});
  auto& cache = ActionTesting::cache<observer_writer>(runner, 0);

  runner.set_phase(Parallel::Phase::Execute);

  const auto make_center = [&gen, &center_dist]() -> std::array<double, 3> {
    return make_with_random_values<std::array<double, 3>>(
        make_not_null(&gen), center_dist, std::array<double, 3>{});
  };

  // Lists of centers so we can check that the correct centers were written
  std::vector<std::array<double, 3>> grid_centers{};
  std::vector<std::array<double, 3>> inertial_centers{};

  db::DataBox<tmpl::list<StrahlkorperTags::Strahlkorper<Frame>,
                         StrahlkorperTags::CartesianCoords<::Frame::Inertial>,
                         StrahlkorperTags::EuclideanAreaElement<Frame>>>
      box{};

  const auto update_stored_centers = [&make_center, &grid_centers,
                                      &inertial_centers, &box]() {
    const auto grid_center = make_center();
    const auto inertial_center = make_center();

    grid_centers.push_back(grid_center);
    inertial_centers.push_back(inertial_center);

    db::mutate<StrahlkorperTags::Strahlkorper<Frame>,
               StrahlkorperTags::CartesianCoords<::Frame::Inertial>,
               StrahlkorperTags::EuclideanAreaElement<Frame>>(
        [&grid_center, &inertial_center](
            gsl::not_null<Strahlkorper<Frame>*> box_grid_horizon,
            gsl::not_null<tnsr::I<DataVector, 3, ::Frame::Inertial>*>
                box_inertial_coords,
            gsl::not_null<Scalar<DataVector>*> box_area_element) {
          *box_grid_horizon = Strahlkorper<Frame>{10, 1.0, grid_center};
          const auto theta_phi =
              StrahlkorperFunctions::theta_phi(*box_grid_horizon);
          const auto radius = StrahlkorperFunctions::radius(*box_grid_horizon);
          const auto rhat = StrahlkorperFunctions::rhat(theta_phi);

          // Set area element to the Euclidean area element for the test.
          *box_area_element = StrahlkorperGr::euclidean_area_element(
              StrahlkorperFunctions::jacobian(theta_phi),
              StrahlkorperFunctions::normal_one_form(
                  StrahlkorperFunctions::cartesian_derivs_of_scalar(
                      radius, *box_grid_horizon, radius,
                      StrahlkorperFunctions::inv_jacobian(theta_phi)),
                  rhat),
              radius, rhat);

          // Simply offset the inertial coords from the grid coords.
          const auto grid_coords = StrahlkorperFunctions::cartesian_coords(
              *box_grid_horizon, radius, rhat);
          for (size_t i = 0; i < 3; ++i) {
            box_inertial_coords->get(i) = grid_coords.get(i) +
                                          gsl::at(inertial_center, i) -
                                          gsl::at(grid_center, i);
          }
        },
        make_not_null(&box));
  };

  // times to write
  const std::vector<double> times{0.0, 0.1, 0.2, 0.3, 0.4, 0.5};

  // write some data
  for (size_t i = 0; i < times.size(); i++) {
    update_stored_centers();

    ah::callbacks::ObserveCenters<AhA, Frame>::apply(box, cache, times[i]);

    size_t num_threaded_actions =
        ActionTesting::number_of_queued_threaded_actions<observer_writer>(
            runner, 0);
    CHECK(num_threaded_actions == 1);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
  }

  // These must be the same as in ObserveCenters
  const std::vector<std::string> compare_legend{
      {"Time", "GridCenter_x", "GridCenter_y", "GridCenter_z",
       "InertialCenter_x", "InertialCenter_y", "InertialCenter_z"}};
  const std::string subfile_name =
      "/ApparentHorizons/" + pretty_type::name<AhA>() + "_Centers";

  auto& h5_file = ActionTesting::get_databox_tag<
      observer_writer, TestHelpers::observers::MockReductionFileTag>(runner, 0);
  const auto& dataset = h5_file.get_dat(subfile_name);
  const Matrix data = dataset.get_data();
  const std::vector<std::string>& legend = dataset.get_legend();

  // Check legend is correct
  for (size_t i = 0; i < legend.size(); i++) {
    CHECK(legend[i] == compare_legend[i]);
  }

  // Check proper number of times were written
  CHECK(data.rows() == times.size());

  // Check centers
  for (size_t i = 0; i < times.size(); i++) {
    CHECK(data(i, 0) == times[i]);

    const std::array<double, 3>& grid_center = grid_centers[i];
    const std::array<double, 3>& inertial_center = inertial_centers[i];
    for (size_t j = 0; j < grid_center.size(); j++) {
      // Grid center is columns 2-4
      CHECK(data(i, j + 1) == approx(gsl::at(grid_center, j)));
      // Inertial center is columns 5-7
      CHECK(data(i, j + 4) == approx(gsl::at(inertial_center, j)));
    }
  }
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.ObserveCenters",
                  "[Unit][ApparentHorizons]") {
  test<::Frame::Grid>();
  test<::Frame::Distorted>();
}
}  // namespace

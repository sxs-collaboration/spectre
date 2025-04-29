// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestingFramework.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <deque>
#include <unordered_set>
#include <vector>

#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputePoints.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {
namespace {
void test_compute_points() {
  // Only test grid frame because the only difference is a different code path
  // is taken within block logical coordinates which we aren't testing here
  using Fr = ::Frame::Grid;

  const size_t l_max = 4;
  ah::Storage::Iteration<Fr> current_iteration{};

  const domain::creators::Sphere sphere{
      1.0,          3.0,  domain::creators::Sphere::Excision{nullptr},
      0_st,         2_st, true,
      std::nullopt, {2.0}};
  const Domain<3> domain = sphere.create_domain();
  const domain::FunctionsOfTimeMap functions_of_time =
      sphere.functions_of_time();

  const size_t max_interpolation_retries = 3;
  const LinkedMessageId<double> time{4.0, {3.0}};
  ylm::Strahlkorper<Fr> initial_guess{l_max, 2.5, std::array{0.0, 0.0, 0.0}};
  ylm::Strahlkorper<Fr> previous_iteration_surface =
      current_iteration.strahlkorper;
  std::deque<ah::Storage::PreviousSurface<Fr>> previous_surfaces{};
  FastFlow fast_flow{
      FastFlow::FlowType::Fast, 1.0, 0.5, 1.e-12, 1.e-2, 1.2, 5, 100};

  // We should find points
  {
    const bool coords_set_successfully = set_current_iteration_coords(
        make_not_null(&current_iteration), time, fast_flow, initial_guess,
        previous_iteration_surface, previous_surfaces,
        max_interpolation_retries, domain, functions_of_time);

    // We don't check the actual points because the function basically just
    // wraps block_logical_coordinates and that's already tested. Here we check
    // the wrapping logic
    CHECK(coords_set_successfully);
    REQUIRE(current_iteration.block_coord_holders.has_value());
    const size_t l_mesh =
        fast_flow.current_l_mesh(current_iteration.strahlkorper);
    REQUIRE(current_iteration.block_coord_holders.value().size() ==
            ylm::Spherepack::physical_size(l_mesh, l_mesh));
    CHECK(alg::all_of(current_iteration.block_coord_holders.value(),
                      [](const auto& holder) { return holder.has_value(); }));
    CHECK(current_iteration.interpolation_retries == 0);
  }

  // This will succeed after failing once
  {
    current_iteration.block_coord_holders.reset();
    current_iteration.strahlkorper = ylm::Strahlkorper<Fr>{};
    current_iteration.interpolation_retries = 0;
    initial_guess =
        ylm::Strahlkorper<Fr>{l_max, 0.75, std::array{0.0, 0.0, 0.0}};

    const bool coords_set_successfully = set_current_iteration_coords(
        make_not_null(&current_iteration), time, fast_flow, initial_guess,
        previous_iteration_surface, previous_surfaces,
        max_interpolation_retries, domain, functions_of_time);

    CHECK(coords_set_successfully);
    REQUIRE(current_iteration.block_coord_holders.has_value());
    const size_t l_mesh =
        fast_flow.current_l_mesh(current_iteration.strahlkorper);
    REQUIRE(current_iteration.block_coord_holders.value().size() ==
            ylm::Spherepack::physical_size(l_mesh, l_mesh));
    CHECK(alg::all_of(current_iteration.block_coord_holders.value(),
                      [](const auto& holder) { return holder.has_value(); }));
    CHECK(current_iteration.interpolation_retries == 1);
  }

  // This will fail to find points and will increase the radius of the
  // strahlkorper by 50% 3 times.
  {
    current_iteration.block_coord_holders.reset();
    current_iteration.strahlkorper = ylm::Strahlkorper<Fr>{};
    current_iteration.interpolation_retries = 0;
    initial_guess =
        ylm::Strahlkorper<Fr>{l_max, 1.e-6, std::array{0.0, 0.0, 0.0}};

    const bool coords_set_successfully = set_current_iteration_coords(
        make_not_null(&current_iteration), time, fast_flow, initial_guess,
        previous_iteration_surface, previous_surfaces,
        max_interpolation_retries, domain, functions_of_time);

    CHECK_FALSE(coords_set_successfully);
    REQUIRE(current_iteration.block_coord_holders.has_value());
    const size_t l_mesh =
        fast_flow.current_l_mesh(current_iteration.strahlkorper);
    REQUIRE(current_iteration.block_coord_holders.value().size() ==
            ylm::Spherepack::physical_size(l_mesh, l_mesh));
    CHECK(alg::none_of(current_iteration.block_coord_holders.value(),
                       [](const auto& holder) { return holder.has_value(); }));
    CHECK(current_iteration.interpolation_retries == 3);
  }

  // Force linear extrapolation of previous surfaces that will succeed
  {
    current_iteration.block_coord_holders.reset();
    current_iteration.strahlkorper = ylm::Strahlkorper<Fr>{};
    current_iteration.interpolation_retries = 0;
    initial_guess =
        ylm::Strahlkorper<Fr>{l_max, 1.0, std::array{0.0, 0.0, 0.0}};

    const LinkedMessageId<double> prev_time_1{3.0, {2.0}};
    const LinkedMessageId<double> prev_time_2{2.0, {1.0}};
    previous_surfaces.emplace_front(
        prev_time_2,
        ylm::Strahlkorper<Fr>{l_max, 1.1, std::array{0.0, 0.0, 0.0}});
    previous_surfaces.emplace_front(
        prev_time_1,
        ylm::Strahlkorper<Fr>{l_max, 1.3, std::array{0.0, 0.0, 0.0}});

    const bool coords_set_successfully = set_current_iteration_coords(
        make_not_null(&current_iteration), time, fast_flow, initial_guess,
        previous_iteration_surface, previous_surfaces,
        max_interpolation_retries, domain, functions_of_time);

    CHECK(coords_set_successfully);
    REQUIRE(current_iteration.block_coord_holders.has_value());
    const size_t l_mesh =
        fast_flow.current_l_mesh(current_iteration.strahlkorper);
    REQUIRE(current_iteration.block_coord_holders.value().size() ==
            ylm::Spherepack::physical_size(l_mesh, l_mesh));
    CHECK(alg::all_of(current_iteration.block_coord_holders.value(),
                      [](const auto& holder) { return holder.has_value(); }));
    CHECK(current_iteration.interpolation_retries == 0);
  }

  // Force quadratic extrapolation of previous surfaces that will succeed
  {
    current_iteration.block_coord_holders.reset();
    current_iteration.strahlkorper = ylm::Strahlkorper<Fr>{};
    current_iteration.interpolation_retries = 0;

    const LinkedMessageId<double> prev_time{1.0, std::nullopt};
    previous_surfaces.emplace_back(
        prev_time,
        ylm::Strahlkorper<Fr>{l_max, 1.0, std::array{0.0, 0.0, 0.0}});

    const bool coords_set_successfully = set_current_iteration_coords(
        make_not_null(&current_iteration), time, fast_flow, initial_guess,
        previous_iteration_surface, previous_surfaces,
        max_interpolation_retries, domain, functions_of_time);

    CHECK(coords_set_successfully);
    REQUIRE(current_iteration.block_coord_holders.has_value());
    const size_t l_mesh =
        fast_flow.current_l_mesh(current_iteration.strahlkorper);
    REQUIRE(current_iteration.block_coord_holders.value().size() ==
            ylm::Spherepack::physical_size(l_mesh, l_mesh));
    CHECK(alg::all_of(current_iteration.block_coord_holders.value(),
                      [](const auto& holder) { return holder.has_value(); }));
    CHECK(current_iteration.interpolation_retries == 0);
  }

  // Succeed on second iteration of fast flow, but we had to reinterpolate once
  {
    previous_surfaces.clear();
    current_iteration.block_coord_holders.reset();
    current_iteration.strahlkorper =
        ylm::Strahlkorper<Fr>{l_max, 1.0, std::array{0.0, 0.0, 0.0}};
    current_iteration.interpolation_retries = 0;

    // Use Schwarzschild solution to get valid tensors for a fast flow iteration
    const size_t l_mesh =
        fast_flow.current_l_mesh(current_iteration.strahlkorper);
    const gr::Solutions::KerrSchild analytic_solution{
        1.0, std::array{0.0, 0.0, 0.0}, std::array{0.0, 0.0, 0.0}};
    Variables<
        tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3, Fr>,
                   gr::Tags::ExtrinsicCurvature<DataVector, 3, Fr>,
                   gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, Fr>>>
        vars{ylm::Spherepack::physical_size(l_mesh, l_mesh)};
    vars.assign_subset(analytic_solution.variables(
        ylm::cartesian_coords(
            ylm::Strahlkorper<Fr>{l_mesh, 1.0, std::array{0.0, 0.0, 0.0}}),
        time.id, typename std::decay_t<decltype(vars)>::tags_list{}));

    fast_flow.iterate_horizon_finder(
        make_not_null(&current_iteration.strahlkorper),
        get<gr::Tags::InverseSpatialMetric<DataVector, 3, Fr>>(vars),
        get<gr::Tags::ExtrinsicCurvature<DataVector, 3, Fr>>(vars),
        get<gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, Fr>>(vars));

    // Reset strahlkorpers so that the average of these two coefficients are
    // used
    current_iteration.strahlkorper =
        ylm::Strahlkorper<Fr>{l_max, 0.9, std::array{0.0, 0.0, 0.0}};
    previous_iteration_surface =
        ylm::Strahlkorper<Fr>{l_max, 2.0, std::array{0.0, 0.0, 0.0}};

    const bool coords_set_successfully = set_current_iteration_coords(
        make_not_null(&current_iteration), time, fast_flow, initial_guess,
        previous_iteration_surface, previous_surfaces,
        max_interpolation_retries, domain, functions_of_time);

    CHECK(coords_set_successfully);
    REQUIRE(current_iteration.block_coord_holders.has_value());
    REQUIRE(current_iteration.block_coord_holders.value().size() ==
            ylm::Spherepack::physical_size(l_mesh, l_mesh));
    CHECK(alg::all_of(current_iteration.block_coord_holders.value(),
                      [](const auto& holder) { return holder.has_value(); }));
    CHECK(current_iteration.interpolation_retries == 1);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.ComputePoints",
                  "[ApparentHorizonFinder][Unit]") {
  test_compute_points();
}
}  // namespace ah

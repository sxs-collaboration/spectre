// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <string>
#include <utility>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/WrappedGr.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/ParallelRays.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"

namespace ray_tracing {

namespace {

ParallelRays test_parallel_rays(const std::string& options_string) {
  // Test factory-creation
  const auto created =
      TestHelpers::test_factory_creation<RaySource, ParallelRays>(
          options_string);
  REQUIRE(dynamic_cast<const ParallelRays*>(created.get()) != nullptr);
  const auto& camera = dynamic_cast<const ParallelRays&>(*created);
  {
    INFO("Semantics");
    test_serialization(camera);
    test_copy_semantics(camera);
    auto move_camera = camera;
    test_move_semantics(std::move(move_camera), camera);
    const auto clone = camera.get_clone();
    REQUIRE(dynamic_cast<const ParallelRays*>(clone.get()) != nullptr);
    CHECK(dynamic_cast<const ParallelRays&>(*clone) == camera);
  }
  return camera;
}

}  // namespace

SPECTRE_TEST_CASE("Unit.RayTracer.RaySources.ParallelRays",
                  "[Unit][ParallelAlgorithms]") {
  auto camera = test_parallel_rays(
      "ParallelRays:\n"
      "  Position: [30, 0, 0]\n"
      "  Focus: [0, 0, 0]\n"
      "  Up: [0, 0, 1]\n"
      "  Extent: [1.0, 2.0]\n"
      "  Resolution: [3, 5]\n"
      "  CenterRays: false\n"
      "  StartTime: 1.5\n"
      "  Interval: 2.0\n"
      "  NumFrames: 10\n"
      "  IntegrationTime: 100.0\n"
      "  OnlyUpperHalf: false\n");
  const WrappedGr<gr::Solutions::Minkowski<3>> background_spacetime{};
  {
    INFO("Ray geometry");
    camera.initialize(/* frame */ 0, background_spacetime);
    CHECK(camera.four_velocity() == tnsr::A<double, 3>{{1.0, 0.0, 0.0, 0.0}});
    CHECK(camera.direction() == tnsr::A<double, 3>{{0.0, -1.0, 0.0, 0.0}});
    CHECK(camera.up() == tnsr::A<double, 3>{{0.0, 0.0, 0.0, 1.0}});
    CHECK(camera.right() == tnsr::A<double, 3>{{0.0, 0.0, 1.0, 0.0}});
    const std::array<std::pair<size_t, tnsr::I<double, 3>>, 5> expected_pos{
        {// Central ray
         {7_st, tnsr::I<double, 3>{{30.0, 0.0, 0.0}}},
         // Top-middle ray
         {1_st, tnsr::I<double, 3>{{30.0, 0.0, 2.0}}},
         // Bottom-middle ray
         {13_st, tnsr::I<double, 3>{{30.0, 0.0, -2.0}}},
         // Left-middle ray
         {6_st, tnsr::I<double, 3>{{30.0, -1.0, 0.0}}},
         // Right-middle ray
         {8_st, tnsr::I<double, 3>{{30.0, 1.0, 0.0}}}}};
    for (const auto& [ray_index, expected_position] : expected_pos) {
      CAPTURE(ray_index);
      const auto ray = camera(ray_index, background_spacetime);
      CHECK_ITERABLE_APPROX(get<Tags::Position<double>>(ray),
                            expected_position);
      CHECK_ITERABLE_APPROX(get<Tags::Momentum<double>>(ray),
                            SINGLE_ARG(tnsr::i<double, 3>{{1.0, 0.0, 0.0}}));
    }
  }
}

}  // namespace ray_tracing

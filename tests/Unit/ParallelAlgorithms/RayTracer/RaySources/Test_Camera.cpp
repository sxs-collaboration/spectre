// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <string>
#include <utility>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/WrappedGr.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/Camera.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"

namespace ray_tracing {

namespace {

Camera test_camera(const std::string& options_string) {
  // Test factory-creation
  const auto created =
      TestHelpers::test_factory_creation<RaySource, Camera>(options_string);
  REQUIRE(dynamic_cast<const Camera*>(created.get()) != nullptr);
  const auto& camera = dynamic_cast<const Camera&>(*created);
  {
    INFO("Semantics");
    test_serialization(camera);
    test_copy_semantics(camera);
    auto move_camera = camera;
    test_move_semantics(std::move(move_camera), camera);
    const auto clone = camera.get_clone();
    REQUIRE(dynamic_cast<const Camera*>(clone.get()) != nullptr);
    CHECK(dynamic_cast<const Camera&>(*clone) == camera);
  }
  return camera;
}

}  // namespace

SPECTRE_TEST_CASE("Unit.RayTracer.RaySources.Camera",
                  "[Unit][ParallelAlgorithms]") {
  auto camera = test_camera(
      "Camera:\n"
      "  Position: [30, 0, 0]\n"
      "  Focus: [0, 0, 0]\n"
      "  Up: [0, 0, 1]\n"
      "  OpeningAngle: [180.0, 90.0]\n"
      "  Resolution: [3, 5]\n"
      "  CenterRays: false\n"
      "  StartTime: 1.5\n"
      "  Interval: 2.0\n"
      "  NumFrames: 10\n"
      "  IntegrationTime: 100.0\n"
      "  OnlyUpperHalf: false\n");
  const WrappedGr<gr::Solutions::Minkowski<3>> background_spacetime{};
  {
    INFO("Frames");
    CHECK(camera.num_frames() == 10);
    CHECK(camera.start_time() == 1.5);
    CHECK(camera.interval() == 2.0);
    for (size_t frame = 0; frame < camera.num_frames(); ++frame) {
      CAPTURE(frame);
      camera.initialize(frame, background_spacetime);
      CHECK(camera.time() == 1.5 + 2.0 * static_cast<double>(frame));
    }
  }
  {
    INFO("Ray geometry");
    camera.initialize(/* frame */ 0, background_spacetime);
    CHECK(camera.four_velocity() == tnsr::A<double, 3>{{1.0, 0.0, 0.0, 0.0}});
    CHECK(camera.direction() == tnsr::A<double, 3>{{0.0, -1.0, 0.0, 0.0}});
    CHECK(camera.up() == tnsr::A<double, 3>{{0.0, 0.0, 0.0, 1.0}});
    CHECK(camera.right() == tnsr::A<double, 3>{{0.0, 0.0, 1.0, 0.0}});
    const std::array<std::pair<size_t, tnsr::i<double, 3>>, 5> expected_momenta{
        {// Central ray should pierce camera along x-axis
         {7_st, tnsr::i<double, 3>{{1.0, 0.0, 0.0}}},
         // Top-middle ray should pierce camera at 45 degrees down
         {1_st, tnsr::i<double, 3>{{1.0 / sqrt(2.0), 0.0, -1.0 / sqrt(2.0)}}},
         // Bottom-middle ray should pierce camera at 45 degrees up
         {13_st, tnsr::i<double, 3>{{1.0 / sqrt(2.0), 0.0, 1.0 / sqrt(2.0)}}},
         // Left-middle ray should pierce camera at 90 degrees left
         {6_st, tnsr::i<double, 3>{{0.0, 1.0, 0.0}}},
         // Right-middle ray should pierce camera at 90 degrees right
         {8_st, tnsr::i<double, 3>{{0.0, -1.0, 0.0}}}}};
    for (const auto& [ray_index, expected_momentum] : expected_momenta) {
      CAPTURE(ray_index);
      const auto ray = camera(ray_index, background_spacetime);
      CHECK(get<Tags::Position<double>>(ray) ==
            tnsr::I<double, 3>{camera.position()});
      CHECK_ITERABLE_APPROX(get<Tags::Momentum<double>>(ray),
                            expected_momentum);
    }
  }
}

}  // namespace ray_tracing

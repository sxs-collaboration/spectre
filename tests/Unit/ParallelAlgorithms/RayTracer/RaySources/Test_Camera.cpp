// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <variant>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/WrappedGr.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/Camera.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "Utilities/Literals.hpp"

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

std::string camera_options(const std::string& projection_options) {
  return "Camera:\n"
         "  Position: [30, 0, 0]\n"
         "  Focus: [0, 0, 0]\n"
         "  Up: [0, 0, 1]\n"
         "  Projection:\n" +
         projection_options +
         "  Resolution: [3, 5]\n"
         "  CenterRays: false\n"
         "  StartTime: 1.5\n"
         "  Interval: 2.0\n"
         "  NumFrames: 10\n"
         "  IntegrationTime: 100.0\n"
         "  OnlyUpperHalf: false\n";
}

void test_projection_options() {
  INFO("Projection is chosen by name");
  CHECK(std::holds_alternative<PerspectiveProjection>(
      TestHelpers::test_creation<Camera::Projection::type>(
          "Perspective:\n"
          "  OpeningAngle: [60.0, 40.0]")));
  CHECK(std::holds_alternative<OrthographicProjection>(
      TestHelpers::test_creation<Camera::Projection::type>(
          "Orthographic:\n"
          "  Extent: [2.0, 4.0]")));
  // The projection can't be inferred from its options alone
  CHECK_THROWS_WITH(TestHelpers::test_creation<Camera::Projection::type>(
                        "OpeningAngle: [60.0, 40.0]"),
                    Catch::Matchers::ContainsSubstring(
                        "Failed to convert value to type "
                        "PerspectiveProjection or OrthographicProjection"));
}

void check_flat_tetrad(const Camera& camera) {
  CHECK(camera.four_velocity() == tnsr::A<double, 3>{{1.0, 0.0, 0.0, 0.0}});
  CHECK(camera.direction() == tnsr::A<double, 3>{{0.0, -1.0, 0.0, 0.0}});
  CHECK(camera.up() == tnsr::A<double, 3>{{0.0, 0.0, 0.0, 1.0}});
  CHECK(camera.right() == tnsr::A<double, 3>{{0.0, 0.0, 1.0, 0.0}});
}

void test_perspective_flat() {
  INFO("Perspective projection in flat spacetime");
  auto camera =
      test_camera(camera_options("    Perspective:\n"
                                 "      OpeningAngle: [180.0, 90.0]\n"));
  CHECK(std::holds_alternative<PerspectiveProjection>(camera.projection()));
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
    check_flat_tetrad(camera);
    const std::array<std::pair<size_t, tnsr::i<double, 3>>, 5> expected_momenta{
        {// Central ray should pierce camera along x-axis
         {7_st, tnsr::i<double, 3>{{1.0, 0.0, 0.0}}},
         // Top-middle ray should pierce camera at 45 degrees down
         {1_st, tnsr::i<double, 3>{{M_SQRT1_2, 0.0, -M_SQRT1_2}}},
         // Bottom-middle ray should pierce camera at 45 degrees up
         {13_st, tnsr::i<double, 3>{{M_SQRT1_2, 0.0, M_SQRT1_2}}},
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

void test_orthographic_flat() {
  INFO("Orthographic projection in flat spacetime");
  auto camera =
      test_camera(camera_options("    Orthographic:\n"
                                 "      Extent: [2.0, 4.0]\n"));
  CHECK(std::holds_alternative<OrthographicProjection>(camera.projection()));
  const WrappedGr<gr::Solutions::Minkowski<3>> background_spacetime{};
  camera.initialize(/* frame */ 0, background_spacetime);
  check_flat_tetrad(camera);
  const std::array<std::pair<size_t, tnsr::I<double, 3>>, 5> expected_positions{
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
  for (const auto& [ray_index, expected_position] : expected_positions) {
    CAPTURE(ray_index);
    const auto ray = camera(ray_index, background_spacetime);
    CHECK_ITERABLE_APPROX(get<Tags::Position<double>>(ray), expected_position);
    CHECK_ITERABLE_APPROX(get<Tags::Momentum<double>>(ray),
                          SINGLE_ARG(tnsr::i<double, 3>{{1.0, 0.0, 0.0}}));
  }
}

// Check identities that hold in any spacetime. The Kerr-Schild spacetime has a
// nonzero shift and a lapse different from one, so it catches errors in the
// normalization of the momentum.
void test_curved(const Camera::Projection::type& projection) {
  const WrappedGr<gr::Solutions::KerrSchild> background_spacetime{
      {/* mass */ 1., /* spin */ {{0.1, 0.2, 0.5}},
       /* center */ {{0., 0., 0.}}}};
  Camera camera{/* position */ {{8.0, 3.0, 2.0}},
                /* focus */ {{0.0, 0.0, 0.0}},
                /* up */ {{0.0, 0.0, 1.0}},
                projection,
                /* resolution */ {{4, 5}},
                /* center_rays */ true,
                /* start_time */ 0.0,
                /* interval */ 0.0,
                /* num_frames */ 1,
                /* integration_time */ 100.0};
  camera.initialize(/* frame */ 0, background_spacetime);
  {
    INFO("Initializing again gives identical rays");
    Camera reinitialized_camera = camera;
    reinitialized_camera.initialize(/* frame */ 0, background_spacetime);
    CHECK(reinitialized_camera == camera);
  }
  const auto background_vars =
      background_spacetime.variables(camera.position(), camera.time());
  const auto& lapse = get<gr::Tags::Lapse<double>>(background_vars);
  const auto& shift = get<gr::Tags::Shift<double, 3>>(background_vars);
  const auto& inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<double, 3>>(background_vars);
  const auto inv_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inv_spatial_metric);
  const auto spacetime_metric =
      determinant_and_inverse(inv_spacetime_metric).second;
  {
    INFO("Tetrad is orthonormal");
    const std::array<tnsr::A<double, 3>, 4> tetrad{
        {camera.four_velocity(), camera.direction(), camera.up(),
         camera.right()}};
    for (size_t a = 0; a < 4; ++a) {
      for (size_t b = 0; b < 4; ++b) {
        CAPTURE(a);
        CAPTURE(b);
        CHECK(get(dot_product(gsl::at(tetrad, a), gsl::at(tetrad, b),
                              spacetime_metric)) ==
              approx(a == b ? (a == 0 ? -1.0 : 1.0) : 0.0));
      }
    }
  }
  for (size_t ray_index = 0; ray_index < camera.num_rays(0); ++ray_index) {
    CAPTURE(ray_index);
    const auto ray = camera(ray_index, background_spacetime);
    const auto& pi = get<Tags::Momentum<double>>(ray);
    // Pi_i = p_i / (alpha p^0) has unit norm for null rays
    CHECK(get(dot_product(pi, pi, inv_spatial_metric)) == approx(1.0));
    // Reconstruct the four-momentum (up to normalization) from Pi_i using
    // p_0 = -alpha^2 p^0 + beta^i p_i
    tnsr::a<double, 3> p{};
    get<0>(p) = -get(lapse) + get(dot_product(shift, pi));
    for (size_t i = 0; i < 3; ++i) {
      p.get(i + 1) = pi.get(i);
    }
    CHECK(get(dot_product(p, p, inv_spacetime_metric)) ==
          approx(0.0).margin(1.e-12));
    const double energy = -get(dot_product(p, camera.four_velocity()));
    CHECK(energy > 0.0);
    if (std::holds_alternative<PerspectiveProjection>(projection)) {
      // The camera sees the ray at the angle that corresponds to its screen
      // coordinates
      const auto& opening_angle =
          std::get<PerspectiveProjection>(projection).opening_angle;
      const auto [x_frac, y_frac] = camera.screen_coordinates(ray_index);
      const double x_angle =
          (2.0 * x_frac - 1.0) * tan(opening_angle[0] * M_PI / 360.0);
      const double y_angle =
          (2.0 * y_frac - 1.0) * tan(opening_angle[1] * M_PI / 360.0);
      CHECK(get(dot_product(p, camera.direction())) / energy ==
            approx(-1.0 / sqrt(1.0 + square(x_angle) + square(y_angle))));
      CHECK(get(dot_product(p, camera.right())) / energy ==
            approx(-x_angle / sqrt(1.0 + square(x_angle) + square(y_angle))));
      CHECK(get<Tags::Position<double>>(ray) ==
            tnsr::I<double, 3>{camera.position()});
    } else {
      // All rays travel in the viewing direction
      CHECK(get(dot_product(p, camera.direction())) / energy == approx(-1.0));
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.RayTracer.RaySources.Camera",
                  "[Unit][ParallelAlgorithms]") {
  test_projection_options();
  test_perspective_flat();
  test_orthographic_flat();
  {
    INFO("Perspective projection in Kerr-Schild spacetime");
    test_curved(PerspectiveProjection{{{60.0, 40.0}}});
  }
  {
    INFO("Orthographic projection in Kerr-Schild spacetime");
    test_curved(OrthographicProjection{{{2.0, 4.0}}});
  }
}

}  // namespace ray_tracing

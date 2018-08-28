// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <random>

#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/OrientationMap.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"

namespace {
void test_suite_for_frustum(const bool with_equiangular_map) {
  // Set up random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> lower_bound_dis(-14, -2);
  std::uniform_real_distribution<> upper_bound_dis(2, 14);

  const double lower_x_lower_base = lower_bound_dis(gen);
  CAPTURE_PRECISE(lower_x_lower_base);
  const double lower_y_lower_base = lower_bound_dis(gen);
  CAPTURE_PRECISE(lower_y_lower_base);
  const double upper_x_lower_base = upper_bound_dis(gen);
  CAPTURE_PRECISE(upper_x_lower_base);
  const double upper_y_lower_base = upper_bound_dis(gen);
  CAPTURE_PRECISE(upper_y_lower_base);
  const double lower_x_upper_base = lower_bound_dis(gen);
  CAPTURE_PRECISE(lower_x_upper_base);
  const double lower_y_upper_base = lower_bound_dis(gen);
  CAPTURE_PRECISE(lower_y_upper_base);
  const double upper_x_upper_base = upper_bound_dis(gen);
  CAPTURE_PRECISE(upper_x_upper_base);
  const double upper_y_upper_base = upper_bound_dis(gen);
  CAPTURE_PRECISE(upper_y_upper_base);

  for (OrientationMapIterator<3> map_i{}; map_i; ++map_i) {
    const std::array<std::array<double, 2>, 4> face_vertices{
        {{{lower_x_lower_base, lower_y_lower_base}},
         {{upper_x_lower_base, upper_y_lower_base}},
         {{lower_x_upper_base, lower_y_upper_base}},
         {{upper_x_upper_base, upper_y_upper_base}}}};
    const domain::CoordinateMaps::Frustum frustum_map(
        face_vertices, -1.0, 2.0, map_i(), with_equiangular_map);
    test_suite_for_map_on_unit_cube(frustum_map);
  }
}

void test_frustum_fail() noexcept {
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const domain::CoordinateMaps::Frustum map(face_vertices, 2.0, 5.0,
                                            domain::OrientationMap<3>{});

  // For the choice of params above, any point with z<=-1 should fail.
  const std::array<double, 3> test_mapped_point1{{3.0, 3.0, -1.0}};
  const std::array<double, 3> test_mapped_point2{{6.0, -7.0, -1.0}};
  const std::array<double, 3> test_mapped_point3{{6.0, -7.0, -3.0}};

  // This is outside the mapped frustum, so inverse should either
  // return the correct inverse (which happens to be computable for
  // this point) or it should return boost::none.
  const std::array<double, 3> test_mapped_point4{{0.0, 0.0, 9.0}};

  CHECK_FALSE(static_cast<bool>(map.inverse(test_mapped_point1)));
  CHECK_FALSE(static_cast<bool>(map.inverse(test_mapped_point2)));
  CHECK_FALSE(static_cast<bool>(map.inverse(test_mapped_point3)));
  if(map.inverse(test_mapped_point4)) {
    CHECK_ITERABLE_APPROX(map(map.inverse(test_mapped_point4).get()),
                          test_mapped_point4);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Frustum.Fail", "[Domain][Unit]") {
  test_frustum_fail();
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Frustum.Equidistant",
                  "[Domain][Unit]") {
  test_suite_for_frustum(false);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Frustum.Equiangular",
                  "[Domain][Unit]") {
  test_suite_for_frustum(true);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Frustum.Alignment",
                  "[Domain][Unit]") {
  // This test tests that the logical axes point along the expected directions
  // in physical space

  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const double lower_bound = 2.0;
  const double upper_bound = 5.0;

  const auto wedge_directions = all_wedge_directions();
  const domain::CoordinateMaps::Frustum map_upper_zeta(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[0]);  // Upper Z frustum
  const domain::CoordinateMaps::Frustum map_upper_eta(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[2]);  // Upper Y frustum
  const domain::CoordinateMaps::Frustum map_upper_xi(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[4]);  // Upper X Frustum
  const domain::CoordinateMaps::Frustum map_lower_zeta(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[1]);  // Lower Z frustum
  const domain::CoordinateMaps::Frustum map_lower_eta(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[3]);  // Lower Y frustum
  const domain::CoordinateMaps::Frustum map_lower_xi(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[5]);  // Lower X frustum
  const std::array<double, 3> lowest_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> along_xi{{1.0, -1.0, -1.0}};
  const std::array<double, 3> along_eta{{-1.0, 1.0, -1.0}};
  const std::array<double, 3> along_zeta{{-1.0, -1.0, 1.0}};

  const std::array<double, 3> lowest_physical_corner_in_map_upper_zeta{
      {-2.0, -2.0, 2.0}};
  const std::array<double, 3> lowest_physical_corner_in_map_upper_eta{
      {-2.0, 2.0, -2.0}};
  const std::array<double, 3> lowest_physical_corner_in_map_upper_xi{
      {2.0, -2.0, -2.0}};
  const std::array<double, 3> lowest_physical_corner_in_map_lower_zeta{
      {-2.0, 2.0, -2.0}};
  const std::array<double, 3> lowest_physical_corner_in_map_lower_eta{
      {-2.0, -2.0, 2.0}};
  const std::array<double, 3> lowest_physical_corner_in_map_lower_xi{
      {-2.0, 2.0, -2.0}};

  // Test that this map's logical axes point along +X, +Y, +Z:
  CHECK(map_upper_zeta(along_xi)[0] == approx(2.0));
  CHECK(map_upper_zeta(along_eta)[1] == approx(2.0));
  CHECK(map_upper_zeta(along_zeta)[2] == approx(5.0));
  CHECK_ITERABLE_APPROX(map_upper_zeta(lowest_corner),
                        lowest_physical_corner_in_map_upper_zeta);

  // Test that this map's logical axes point along +Z, +X, +Y:
  CHECK(map_upper_eta(along_xi)[2] == approx(2.0));
  CHECK(map_upper_eta(along_eta)[0] == approx(2.0));
  CHECK(map_upper_eta(along_zeta)[1] == approx(5.0));
  CHECK_ITERABLE_APPROX(map_upper_eta(lowest_corner),
                        lowest_physical_corner_in_map_upper_eta);

  // Test that this map's logical axes point along +Y, +Z, +X:
  CHECK(map_upper_xi(along_xi)[1] == approx(2.0));
  CHECK(map_upper_xi(along_eta)[2] == approx(2.0));
  CHECK(map_upper_xi(along_zeta)[0] == approx(5.0));
  CHECK_ITERABLE_APPROX(map_upper_xi(lowest_corner),
                        lowest_physical_corner_in_map_upper_xi);

  // Test that this map's logical axes point along +X, -Y, -Z:
  CHECK(map_lower_zeta(along_xi)[0] == approx(2.0));
  CHECK(map_lower_zeta(along_eta)[1] == approx(-2.0));
  CHECK(map_lower_zeta(along_zeta)[2] == approx(-5.0));
  CHECK_ITERABLE_APPROX(map_lower_zeta(lowest_corner),
                        lowest_physical_corner_in_map_lower_zeta);

  // Test that this map's logical axes point along -Z, +X, -Y:
  CHECK(map_lower_eta(along_xi)[2] == approx(-2.0));
  CHECK(map_lower_eta(along_eta)[0] == approx(2.0));
  CHECK(map_lower_eta(along_zeta)[1] == approx(-5.0));
  CHECK_ITERABLE_APPROX(map_lower_eta(lowest_corner),
                        lowest_physical_corner_in_map_lower_eta);

  // Test that this map's logical axes point along -Y, +Z, -X:
  CHECK(map_lower_xi(along_xi)[1] == approx(-2.0));
  CHECK(map_lower_xi(along_eta)[2] == approx(2.0));
  CHECK(map_lower_xi(along_zeta)[0] == approx(-5.0));
  CHECK_ITERABLE_APPROX(map_lower_xi(lowest_corner),
                        lowest_physical_corner_in_map_lower_xi);
}

// [[OutputRegex, The lower bound for a coordinate must be numerically less
// than the upper bound for that coordinate.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Frustum.Assert1",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{-3.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const double lower_bound = 2.0;
  const double upper_bound = 5.0;

  auto failed_frustum = domain::CoordinateMaps::Frustum(
      face_vertices, lower_bound, upper_bound, domain::OrientationMap<3>{});
  static_cast<void>(failed_frustum);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The lower bound for a coordinate must be numerically less
// than the upper bound for that coordinate.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Frustum.Assert2",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, -3.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const double lower_bound = 2.0;
  const double upper_bound = 5.0;

  auto failed_frustum = domain::CoordinateMaps::Frustum(
      face_vertices, lower_bound, upper_bound, domain::OrientationMap<3>{});
  static_cast<void>(failed_frustum);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The lower bound for a coordinate must be numerically less
// than the upper bound for that coordinate.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Frustum.Assert3",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{-5.0, 4.0}}}};
  const double lower_bound = 2.0;
  const double upper_bound = 5.0;

  auto failed_frustum = domain::CoordinateMaps::Frustum(
      face_vertices, lower_bound, upper_bound, domain::OrientationMap<3>{});
  static_cast<void>(failed_frustum);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The lower bound for a coordinate must be numerically less
// than the upper bound for that coordinate.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Frustum.Assert4",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, -5.0}}}};
  const double lower_bound = 2.0;
  const double upper_bound = 5.0;

  auto failed_frustum = domain::CoordinateMaps::Frustum(
      face_vertices, lower_bound, upper_bound, domain::OrientationMap<3>{});
  static_cast<void>(failed_frustum);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The lower bound for a coordinate must be numerically less
// than the upper bound for that coordinate.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Frustum.Assert5",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const double lower_bound = 2.0;
  const double upper_bound = -2.0;

  auto failed_frustum = domain::CoordinateMaps::Frustum(
      face_vertices, lower_bound, upper_bound, domain::OrientationMap<3>{});
  static_cast<void>(failed_frustum);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

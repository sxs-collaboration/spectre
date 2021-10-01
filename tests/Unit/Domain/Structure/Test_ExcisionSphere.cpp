// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>

#include "Domain/Structure/ExcisionSphere.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

namespace {
template <size_t VolumeDim>
void check_excision_sphere_work(const double radius,
                                const std::array<double, VolumeDim> center) {
  const ExcisionSphere<VolumeDim> excision_sphere(radius, center);

  CHECK(excision_sphere.radius() == radius);
  CHECK(excision_sphere.center() == center);
  CHECK(excision_sphere == excision_sphere);
  CHECK_FALSE(excision_sphere != excision_sphere);

  const double diff_radius = 0.001;
  const ExcisionSphere<VolumeDim> excision_sphere_diff_radius(diff_radius,
                                                              center);
  CHECK(excision_sphere != excision_sphere_diff_radius);
  CHECK_FALSE(excision_sphere == excision_sphere_diff_radius);

  CHECK(get_output(excision_sphere) ==
        "ExcisionSphere:\n"
        "  Radius: " +
            get_output(excision_sphere.radius()) +
            "\n"
            "  Center: " +
            get_output(excision_sphere.center()) + "\n");

  test_serialization(excision_sphere);
}

void check_excision_sphere_1d() {
  const double radius = 1.2;
  const std::array<double, 1> center = {{5.4}};
  check_excision_sphere_work<1>(radius, center);
}

void check_excision_sphere_2d() {
  const double radius = 4.2;
  const std::array<double, 2> center = {{5.4, -2.3}};
  check_excision_sphere_work<2>(radius, center);
}

void check_excision_sphere_3d() {
  const double radius = 5.2;
  const std::array<double, 3> center = {{5.4, -2.3, 9.0}};
  check_excision_sphere_work<3>(radius, center);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.ExcisionSphere", "[Domain][Unit]") {
  check_excision_sphere_1d();
  check_excision_sphere_2d();
  check_excision_sphere_3d();
}

// [[OutputRegex, The ExcisionSphere must have a radius greater than zero.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Structure.ExcisionSphereAssert",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_excision_sphere = ExcisionSphere<3>(-2.0, {{3.4, 1.2, -0.9}});
  static_cast<void>(failed_excision_sphere);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

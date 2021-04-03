// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <random>

#include "ApparentHorizons/ChangeCenterOfStrahlkorper.hpp"
#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace Frame {
struct Inertial;
}  // namespace Frame

namespace {

Strahlkorper<Frame::Inertial> make_strahlkorper() noexcept {
  const std::array<double, 3> center{{0.1, 0.2, 0.3}};
  const size_t l_max = 18;
  const double avg_radius = 2.0;
  const double delta_radius = 0.1;

  std::uniform_real_distribution<double> interval_dis(-1.0, 1.0);
  MAKE_GENERATOR(gen);

  // First make a sphere.
  Strahlkorper<Frame::Inertial> sk(l_max, l_max, avg_radius, center);
  // Now adjust the coefficients randomly, but in a manner so that
  // coefficients decay like exp(-l).
  auto coefs = sk.coefficients();
  for (SpherepackIterator it(l_max, l_max); it; ++it) {
    coefs[it()] +=
        interval_dis(gen) * delta_radius * exp(-static_cast<double>(it.l()));
  }
  return Strahlkorper<Frame::Inertial>(coefs, sk);
}

void test_change_center_of_strahlkorper() noexcept {
  auto strahlkorper = make_strahlkorper();
  const auto original_strahlkorper = strahlkorper;
  const auto original_physical_center = strahlkorper.physical_center();

  const std::array<double, 3> new_center{{0.12, 0.21, 0.33}};
  change_expansion_center_of_strahlkorper(make_not_null(&strahlkorper),
                                          new_center);

  // Center should have changed
  for (size_t i = 0; i < 3; ++i) {
    CHECK(gsl::at(new_center, i) == gsl::at(strahlkorper.center(), i));
  }

  // Physical center should not change (to lowest order only, since physical
  // center is only computed to l=1.  Therefore we use a custom approx here.)
  const auto new_physical_center = strahlkorper.physical_center();
  Approx custom_approx_three = Approx::custom().epsilon(1.e-3).scale(1.);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(custom_approx_three(gsl::at(original_physical_center, i)) ==
          gsl::at(new_physical_center, i));
  }

  // Now transform back
  change_expansion_center_of_strahlkorper(make_not_null(&strahlkorper),
                                          original_strahlkorper.center());

  // The original center and radii should have been restored.
  for (size_t i = 0; i < 3; ++i) {
    CHECK(gsl::at(original_strahlkorper.center(), i) ==
          gsl::at(strahlkorper.center(), i));
  }
  // Radii are not the same to machine precision, but only to the
  // specified l_max.  If l_max is changed at the top of this file, then the
  // approx below may need to change.
  Approx custom_approx_nine = Approx::custom().epsilon(1.e-9).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(
      strahlkorper.ylm_spherepack().spec_to_phys(strahlkorper.coefficients()),
      original_strahlkorper.ylm_spherepack().spec_to_phys(
          original_strahlkorper.coefficients()),
      custom_approx_nine);
}

void test_change_center_of_strahlkorper_to_physical() {
  auto strahlkorper = make_strahlkorper();

  change_expansion_center_of_strahlkorper_to_physical(
      make_not_null(&strahlkorper));

  const auto new_physical_center = strahlkorper.physical_center();
  const auto new_center = strahlkorper.center();
  for (size_t i = 0; i < 3; ++i) {
    CHECK(approx(gsl::at(new_physical_center, i)) == gsl::at(new_center, i));
  }
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.Strahlkorper.ChangeCenter",
                  "[ApparentHorizons][Unit]") {
  test_change_center_of_strahlkorper();
  test_change_center_of_strahlkorper_to_physical();
}
}  // namespace

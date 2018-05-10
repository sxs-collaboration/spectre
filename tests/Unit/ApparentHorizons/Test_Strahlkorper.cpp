// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace {

void test_invert_spec_phys_transform() {
  const double avg_radius = 1.0;
  const double delta_radius = 0.1;
  const size_t l_grid = 33;
  const auto l_grid_high_res = static_cast<size_t>(l_grid * 1.5);
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};

  // Create radius as a function of angle
  DataVector radius(YlmSpherepack::physical_size(l_grid, l_grid), avg_radius);
  {
    std::uniform_real_distribution<double> ran(0.0, 1.0);
    std::mt19937 gen;
    for (auto& r : radius) {
      r += delta_radius * ran(gen);
    }
  }
  CAPTURE_PRECISE(radius);

  // Initialize a strahlkorper of l_max=l_grid
  const Strahlkorper<Frame::Inertial> sk(l_grid, l_grid, radius, center);

  // Put that Strahlkorper onto a larger grid
  const Strahlkorper<Frame::Inertial> sk_high_res(l_grid_high_res,
                                                  l_grid_high_res, sk);

  // Compare coefficients
  SpherepackIterator iter(sk.l_max(), sk.m_max());
  SpherepackIterator iter_high_res(sk_high_res.l_max(), sk_high_res.m_max());
  const auto& init_coefs = sk.coefficients();
  const auto& final_coefs = sk_high_res.coefficients();

  for (size_t l = 0; l <= sk.ylm_spherepack().l_max(); ++l) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
      CHECK(init_coefs[iter.set(l, m)()] ==
            approx(final_coefs[iter_high_res.set(l, m)()]));
    }
  }

  for (size_t l = sk.ylm_spherepack().l_max() + 1;
       l <= sk_high_res.ylm_spherepack().l_max(); ++l) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
      CHECK(final_coefs[iter_high_res.set(l, m)()] == approx(0.0));
    }
  }
}

void test_average_radius() {
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const double r = 3.0;
  Strahlkorper<Frame::Inertial> s(4, 4, r, center);
  CHECK(s.average_radius() == approx(r));
}

void test_copy_and_move() {
  Strahlkorper<Frame::Inertial> s(4, 4, 3.0, {{0.1, 0.2, 0.3}});

  test_copy_semantics(s);
  auto s_copy = s;
  test_move_semantics(std::move(s), s_copy);
}

void test_physical_center() {
  const std::array<double, 3> physical_center = {{1.5, 0.5, 1.0}};
  const std::array<double, 3> expansion_center = {{0.0, 0.0, 0.0}};
  const double radius = 5.0;
  const int l_max = 9;

  Strahlkorper<Frame::Inertial> sk(l_max, l_max, radius, expansion_center);
  DataVector r(sk.ylm_spherepack().physical_size(), 0.);

  for (size_t s = 0; s < r.size(); ++s) {
    const double theta = sk.ylm_spherepack().theta_phi_points()[0][s];
    const double phi = sk.ylm_spherepack().theta_phi_points()[1][s];
    // Compute the distance (radius as a function of theta,phi) from
    // the expansion_center to a spherical surface of radius `radius`
    // centered at physical_center.
    const double a = 1.0;
    const double b = -2.0 * cos(phi) * sin(theta) * physical_center[0] -
                     2.0 * sin(phi) * sin(theta) * physical_center[1] -
                     2.0 * cos(theta) * physical_center[2];
    const double c = square(physical_center[0]) + square(physical_center[1]) +
                     square(physical_center[2]) - square(radius);
    auto roots = real_roots(a, b, c);
    r[s] = std::max(roots[0], roots[1]);
  }
  // Construct a new Strahlkorper sk_test with the radius computed
  // above, centered at expansion_center, so that
  // sk_test.physical_center() should recover the physical center of
  // this surface.
  Strahlkorper<Frame::Inertial> sk_test(l_max, l_max, r, expansion_center);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(approx(gsl::at(physical_center, i)) ==
          gsl::at(sk_test.physical_center(), i));
  }
}

void test_point_is_contained() {
  // Construct a spherical Strahlkorper
  const double radius = 2.;
  const std::array<double, 3> center = {{-1.2, 3., 4.}};
  const Strahlkorper<Frame::Inertial> sphere(3, 2, radius, center);

  // Check whether two known points are contained.
  const std::array<double, 3> point_inside = {{-1.2, 1.01, 4.}};
  const std::array<double, 3> point_outside = {{-1.2, 3., 6.01}};
  CHECK(sphere.point_is_contained(point_inside));
  CHECK_FALSE(sphere.point_is_contained(point_outside));
}

template <typename Func>
void test_constructor_with_different_coefs(Func function) {
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const double r = 3.0;
  const double add_to_r = 1.34;
  Strahlkorper<Frame::Inertial> strahlkorper(4, 4, r, center);
  const Strahlkorper<Frame::Inertial> strahlkorper_test1(4, 4, r + add_to_r,
                                                         center);

  // Modify the 0,0 coefficient to add a constant to the radius.
  const auto strahlkorper_test2 = function(strahlkorper, add_to_r);

  CHECK_ITERABLE_APPROX(strahlkorper_test1.coefficients(),
                        strahlkorper_test2.coefficients());
}

void test_construct_from_options() {
  Options<tmpl::list<OptionTags::Strahlkorper<Frame::Inertial>>> opts("");
  opts.parse(
      "Strahlkorper:\n"
      " Lmax : 6\n"
      " Center: [1.,2.,3.]\n"
      " Radius: 4.5\n");
  CHECK(opts.get<OptionTags::Strahlkorper<Frame::Inertial>>() ==
        Strahlkorper<Frame::Inertial>(6, 6, 4.5, {{1., 2., 3.}}));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizons.Strahlkorper",
                  "[ApparentHorizons][Unit]") {
  test_invert_spec_phys_transform();
  test_copy_and_move();
  test_average_radius();
  test_physical_center();
  test_point_is_contained();
  test_constructor_with_different_coefs([](Strahlkorper<Frame::Inertial> & sk,
                                           double add_to_r) noexcept {
    auto coefs = sk.coefficients();  // make a copy
    coefs[0] += sqrt(8.0) * add_to_r;
    return Strahlkorper<Frame::Inertial>(coefs, std::move(sk));
  });
  test_constructor_with_different_coefs([](
      const Strahlkorper<Frame::Inertial>& sk, double add_to_r) noexcept {
    auto coefs = sk.coefficients();  // make a copy
    coefs[0] += sqrt(8.0) * add_to_r;
    return Strahlkorper<Frame::Inertial>(coefs, sk);
  });
  test_constructor_with_different_coefs([](Strahlkorper<Frame::Inertial> & sk,
                                           double add_to_r) noexcept {
    auto& coefs = sk.coefficients();  // no copy
    coefs[0] += sqrt(8.0) * add_to_r;
    return Strahlkorper<Frame::Inertial>(std::move(sk));
  });
  test_construct_from_options();
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.Strahlkorper.Serialization",
                  "[ApparentHorizons][Unit]") {
  Strahlkorper<Frame::Inertial> s(4, 4, 2.0, {{1.0, 2.0, 3.0}});
  test_serialization(s);
}

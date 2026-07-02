// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/FilteringB3.hpp"
#include "NumericalAlgorithms/Spectral/FilteringB3.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// Build the physical-space DataVector for the pure B3 mode (n_jacobi, l) using
// SPHEREPACK offset s (which encodes the (l, m, cos/sin) combination).
DataVector pure_b3_mode_phys(const Mesh<3>& mesh, const size_t n_jacobi,
                             const size_t l, const size_t s) {
  const size_t n_r = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;
  const size_t n_phys = mesh.number_of_grid_points();

  const auto& ylm = ylm::get_spherepack_cache(l_max);
  const size_t n_spectral = ylm.spectral_size();

  const DataVector& radial_pts =
      Spectral::collocation_points<Spectral::Basis::ZernikeB3,
                                   Spectral::Quadrature::GaussRadauUpper>(n_r);
  const DataVector radial_profile =
      Spectral::compute_basis_function_value<Spectral::Basis::ZernikeB3>(
          l + 2 * n_jacobi, l, radial_pts);

  // Construct a SH spectral buffer with just mode s active.
  // Layout: spec_buf[s * n_r + i_r]
  DataVector spec_buf(n_spectral * n_r, 0.0);
  for (size_t i_r = 0; i_r < n_r; ++i_r) {
    spec_buf[s * n_r + i_r] = radial_profile[i_r];
  }

  // Synthesize to physical space.
  DataVector phys(n_phys);
  ylm.spec_to_phys_all_offsets(make_not_null(phys.data()),
                               make_not_null(spec_buf.data()), n_r);
  return phys;
}

double expected_b3_radial_weight(const double alpha, const unsigned half_power,
                                 const size_t n_r, const size_t n_jacobi,
                                 const size_t l) {
  const size_t n_order = n_r - 1;
  const auto n_i =
      static_cast<double>(static_cast<size_t>((l + 2 * n_jacobi) / 2));
  return std::exp(-alpha *
                  std::pow(n_i / static_cast<double>(n_order), 2 * half_power));
}

void test_ball_radial_filter() {
  using TagsList = tmpl::list<::Tags::TempScalar<0>, ::Tags::TempI<1, 3>>;

  // Mesh sizes: (n_r, l_max), satisfying l_max >= 2 and l_max <= 2*n_r-2
  const std::vector<std::pair<size_t, size_t>> mesh_sizes{
      {2, 2}, {3, 2}, {3, 3}, {3, 4}, {4, 5}, {4, 6}};

  for (const auto& [n_r, l_max] : mesh_sizes) {
    CAPTURE(n_r);
    CAPTURE(l_max);
    const Mesh<3> mesh{
        {n_r, l_max + 1, 2 * l_max + 1},
        {Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
         Spectral::Basis::ZernikeB3},
        {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::Equiangular}};
    const size_t n_phys = mesh.number_of_grid_points();

    Variables<TagsList> u{n_phys};
    Variables<TagsList> expected{n_phys};

    // Find the SPHEREPACK offset for (l=0, m=0)
    size_t s_l0_m0 = 0;
    {
      ylm::SpherepackIterator iter{l_max, l_max};
      while (iter) {
        if (iter.l() == 0) {
          s_l0_m0 = iter();
          break;
        }
        ++iter;
      }
    }

    const DataVector top_radial = pure_b3_mode_phys(mesh, n_r - 1, 0, s_l0_m0);

    // Test 1: alpha=0 is the identity.
    {
      get(get<::Tags::TempScalar<0>>(u)) = top_radial;
      get<0>(get<::Tags::TempI<1, 3>>(u)) = top_radial;
      get<1>(get<::Tags::TempI<1, 3>>(u)) = top_radial;
      get<2>(get<::Tags::TempI<1, 3>>(u)) = top_radial;
      expected = u;
      Spectral::filtering::zernike_b3_ball_radial_exponential_filter(
          make_not_null(&u), mesh, 0.0, 2);
      CHECK_VARIABLES_APPROX(u, expected);
    }

    // Test 2: the constant function f=1 is unaffected by any filter.
    // It occupies only the (l=0, n_jacobi=0) mode whose radial weight is
    // always 1 (n_i=0 so w_r = exp(0) = 1).
    {
      get(get<::Tags::TempScalar<0>>(u)) = 1.0;
      get<0>(get<::Tags::TempI<1, 3>>(u)) = 1.0;
      get<1>(get<::Tags::TempI<1, 3>>(u)) = 1.0;
      get<2>(get<::Tags::TempI<1, 3>>(u)) = 1.0;
      expected = u;
      Spectral::filtering::zernike_b3_ball_radial_exponential_filter(
          make_not_null(&u), mesh, 36.0, 32);
      CHECK_VARIABLES_APPROX(u, expected);
    }

    // Test 3: the top radial mode for l=0 (n_jacobi = n_r-1) is killed by
    // heavy filtering.  n_i = n_r-1 = n_order, so
    // w_r = exp(-36 * 1^64) = exp(-36) ~ 2e-16 ~ 0 to machine precision.
    {
      get(get<::Tags::TempScalar<0>>(u)) = top_radial;
      get<0>(get<::Tags::TempI<1, 3>>(u)) = top_radial;
      get<1>(get<::Tags::TempI<1, 3>>(u)) = top_radial;
      get<2>(get<::Tags::TempI<1, 3>>(u)) = top_radial;
      get(get<::Tags::TempScalar<0>>(expected)) = 0.0;
      get<0>(get<::Tags::TempI<1, 3>>(expected)) = 0.0;
      get<1>(get<::Tags::TempI<1, 3>>(expected)) = 0.0;
      get<2>(get<::Tags::TempI<1, 3>>(expected)) = 0.0;
      Spectral::filtering::zernike_b3_ball_radial_exponential_filter(
          make_not_null(&u), mesh, 36.0, 32);
      CHECK_VARIABLES_APPROX(u, expected);
    }

    // Test 4: constant + top radial mode: constant survives, top vanishes.
    {
      get(get<::Tags::TempScalar<0>>(u)) = 1.0 + top_radial;
      get<0>(get<::Tags::TempI<1, 3>>(u)) = 1.0 + top_radial;
      get<1>(get<::Tags::TempI<1, 3>>(u)) = 1.0 + top_radial;
      get<2>(get<::Tags::TempI<1, 3>>(u)) = 1.0 + top_radial;
      get(get<::Tags::TempScalar<0>>(expected)) = 1.0;
      get<0>(get<::Tags::TempI<1, 3>>(expected)) = 1.0;
      get<1>(get<::Tags::TempI<1, 3>>(expected)) = 1.0;
      get<2>(get<::Tags::TempI<1, 3>>(expected)) = 1.0;
      Spectral::filtering::zernike_b3_ball_radial_exponential_filter(
          make_not_null(&u), mesh, 36.0, 32);
      CHECK_VARIABLES_APPROX(u, expected);
    }
  }
}

// For each pure B3 spectral mode (n_jacobi, l, s), verify that
// zernike_b3_ball_radial_exponential_filter scales the mode by w_r only
// (no angular factor).
void test_ball_radial_filter_weights() {
  using TagsList = tmpl::list<::Tags::TempScalar<0>>;

  const std::vector<std::pair<size_t, size_t>> mesh_sizes{
      {2, 2}, {3, 3}, {3, 4}, {4, 5}};
  const std::vector<std::pair<double, unsigned>> params{
      {0.0, 1}, {10.0, 2}, {20.0, 4}, {36.0, 8}};

  for (const auto& [n_r, l_max] : mesh_sizes) {
    CAPTURE(n_r);
    CAPTURE(l_max);
    const Mesh<3> mesh{
        {n_r, l_max + 1, 2 * l_max + 1},
        {Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
         Spectral::Basis::ZernikeB3},
        {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::Equiangular}};
    const size_t n_phys = mesh.number_of_grid_points();
    const size_t n_r_max = 2 * n_r - 2;
    Variables<TagsList> u{n_phys};
    Variables<TagsList> expected_result{n_phys};

    for (const auto& [alpha, half_power] : params) {
      CAPTURE(alpha);
      CAPTURE(half_power);

      ylm::SpherepackIterator iter{l_max, l_max};
      while (iter) {
        const size_t l = iter.l();
        const size_t s = iter();
        const size_t spectral_size_l = (n_r_max - l + 2) / 2;
        CAPTURE(l);
        CAPTURE(s);

        for (size_t n_jacobi = 0; n_jacobi < spectral_size_l; ++n_jacobi) {
          CAPTURE(n_jacobi);
          const DataVector phys = pure_b3_mode_phys(mesh, n_jacobi, l, s);
          get(get<::Tags::TempScalar<0>>(u)) = phys;

          Spectral::filtering::zernike_b3_ball_radial_exponential_filter(
              make_not_null(&u), mesh, alpha, half_power);

          const double weight =
              expected_b3_radial_weight(alpha, half_power, n_r, n_jacobi, l);
          get(get<::Tags::TempScalar<0>>(expected_result)) = weight * phys;
          CHECK_VARIABLES_APPROX(u, expected_result);
        }
        ++iter;
      }
    }
  }
}

#ifdef SPECTRE_DEBUG
void test_asserts() {
  using TagsList = tmpl::list<::Tags::TempScalar<0>>;

  {
    INFO("n_r=1 triggers the min-radial-points assert");
    const Mesh<3> mesh{
        {1, 3, 5},
        {Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
         Spectral::Basis::ZernikeB3},
        {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::Equiangular}};
    Variables<TagsList> u{mesh.number_of_grid_points()};
    CHECK_THROWS_WITH(
        Spectral::filtering::zernike_b3_ball_radial_exponential_filter(
            make_not_null(&u), mesh, 1.0, 2),
        Catch::Matchers::ContainsSubstring(
            "At least 2 radial grid points are required to filter ZernikeB3"));
  }
  {
    INFO("l_max=1 triggers the min-l_max assert");
    const Mesh<3> mesh{
        {3, 2, 3},
        {Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
         Spectral::Basis::ZernikeB3},
        {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::Equiangular}};
    Variables<TagsList> u{mesh.number_of_grid_points()};
    CHECK_THROWS_WITH(
        Spectral::filtering::zernike_b3_ball_radial_exponential_filter(
            make_not_null(&u), mesh, 1.0, 2),
        Catch::Matchers::ContainsSubstring(
            "At least l_max=2 (3 latitudinal points) is required"));
  }
  {
    INFO("l_max > 2*n_r-2 triggers the angular-resolution assert");
    const Mesh<3> mesh{
        {2, 5, 9},
        {Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
         Spectral::Basis::ZernikeB3},
        {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::Equiangular}};
    Variables<TagsList> u{mesh.number_of_grid_points()};
    CHECK_THROWS_WITH(
        Spectral::filtering::zernike_b3_ball_radial_exponential_filter(
            make_not_null(&u), mesh, 1.0, 2),
        Catch::Matchers::ContainsSubstring(
            "ZernikeB3 radial resolution is insufficient for the requested "
            "angular resolution"));
  }
}
#endif  // SPECTRE_DEBUG

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.B3Filter",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_ball_radial_filter();
  test_ball_radial_filter_weights();
#ifdef SPECTRE_DEBUG
  test_asserts();
#endif  // SPECTRE_DEBUG
}
}  // namespace

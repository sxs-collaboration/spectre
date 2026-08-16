// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/FilledSpherePowerMonitor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/SphericalShellPowerMonitor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {

// Number of SH-mode slots contributed to one angular-degree bin l by the
// metric (or pi) TensorYlm group after spin-weight filtering.
//
// Components with |sw| > l are skipped, giving:
//   l=0: real(1) + 3 sw=0 complex(2 each) = 1 + 6 = 7
//   l=1: real(3) + 7 |sw|<=1 complex(4 each) = 3 + 28 = 31
//   l>=2: real(2l+1) + 9 complex(2(l+1) each) = (2l+1) + 18(l+1) = 20l+19
size_t metric_sh_modes_at_l(const size_t l) {
  if (l == 0) {
    return 7;
  } else if (l == 1) {
    return 31;
  } else {
    return 20 * l + 19;
  }
}

// Expected angular power monitor for a single unit-coefficient B3 mode at
// degree l when one metric (or pi) group is excited.  The angular slot count
// = spectral_size_l * metric_sh_modes_at_l(l), where the spectral_size_l
// Jacobi levels each contribute metric_sh_modes_at_l(l) SH offsets.
double expected_b3_angular_pm(const size_t l, const size_t n_r_max) {
  const size_t spectral_size_l = (n_r_max - l + 2) / 2;
  return 1.0 /
         sqrt(static_cast<double>(spectral_size_l * metric_sh_modes_at_l(l)));
}

// Expected radial power monitor for a single unit-coefficient B3 mode landing
// at radial level r when one metric (or pi) group is excited.  For each l'
// that maps to r (even l' via n_total=2r, odd l' via n_total=2r-1), exactly
// one Jacobi coefficient k_spec contributes metric_sh_modes_at_l(l') slots.
// The sum over those l' gives the total count.
double expected_b3_radial_pm(const size_t r, const size_t l_max) {
  size_t count = 0;
  for (size_t l = 0; l <= std::min(2 * r, l_max); l += 2) {
    count += metric_sh_modes_at_l(l);
  }
  if (r >= 1) {
    for (size_t l = 1; l <= std::min(2 * r - 1, l_max); l += 2) {
      count += metric_sh_modes_at_l(l);
    }
  }
  return 1.0 / sqrt(static_cast<double>(count));
}

Mesh<3> b3_mesh(const size_t n_r, const size_t l_max) {
  return Mesh<3>{
      {n_r, l_max + 1, 2 * l_max + 1},
      {Spectral::Basis::ZernikeB3, Spectral::Basis::ZernikeB3,
       Spectral::Basis::ZernikeB3},
      {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
       Spectral::Quadrature::Equiangular}};
}

// Build the physical-space DataVector for the pure B3 mode (n_jacobi, l_ang)
// with SPHEREPACK offset s (encoding the (l, m, cos/sin) SH mode).
DataVector pure_b3_mode_phys(const Mesh<3>& mesh, const size_t n_jacobi,
                             const size_t l_ang, const size_t s) {
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
          l_ang + 2 * n_jacobi, l_ang, radial_pts);

  // Spectral buffer: only mode s active.
  DataVector spec_buf(n_spectral * n_r, 0.0);
  for (size_t i_r = 0; i_r < n_r; ++i_r) {
    spec_buf[s * n_r + i_r] = radial_profile[i_r];
  }

  DataVector phys(n_phys);
  ylm.spec_to_phys_all_offsets(make_not_null(phys.data()),
                               make_not_null(spec_buf.data()), n_r);
  return phys;
}

// Returns the SPHEREPACK offset for the (l_ang, m, cos) SH mode.
size_t spherepack_offset(const size_t l_max, const size_t l_ang,
                         const size_t m = 0) {
  ylm::SpherepackIterator iter{l_max, l_max};
  iter.set(l_ang, m, ylm::SpherepackIterator::CoefficientArray::a);
  return iter();
}

// Identity inverse Jacobian (inertial == grid frame).
InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid> identity_jacobian(
    const size_t size) {
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid> jacobian{size,
                                                                        0.0};
  for (size_t i = 0; i < 3; ++i) {
    jacobian.get(i, i) = 1.0;
  }
  return jacobian;
}

// Returns zeroed GH spacetime variables on the given mesh.
Variables<ylm::TensorYlm::filter_detail::gh_spacetime_vars_list> zero_gh_vars(
    const Mesh<3>& mesh) {
  return {mesh.number_of_grid_points(), 0.0};
}

// Test that zero GH variables give zero power monitors.
void test_zero_input() {
  const Mesh<3> mesh = b3_mesh(3, 2);
  auto gh_vars = zero_gh_vars(mesh);
  gh::power_monitor::CartToSphereMatrices cart_to_sphere{};
  const auto monitors = gh::power_monitor::gh_filled_sphere_power_monitors(
      make_not_null(&cart_to_sphere), gh_vars, mesh,
      identity_jacobian(mesh.number_of_grid_points()));

  CHECK(max(abs(monitors.spacetime_metric[0])) == 0.0);
  CHECK(max(abs(monitors.spacetime_metric[1])) == 0.0);
  CHECK(max(abs(monitors.pi[0])) == 0.0);
  CHECK(max(abs(monitors.pi[1])) == 0.0);
  CHECK(max(abs(monitors.phi[0])) == 0.0);
  CHECK(max(abs(monitors.phi[1])) == 0.0);
}

// Test angular and radial mode isolation: a pure B3 mode in metric_00 (the
// scalar time-time piece) produces power only at the expected angular degree
// and radial mode, and leaves pi and phi monitors at zero.
void test_mode_isolation() {
  // Mesh sizes satisfying l_max >= 1 and l_max <= 2*n_r - 2.
  const std::vector<std::pair<size_t, size_t>> mesh_sizes{
      {2, 2}, {3, 2}, {3, 4}, {4, 4}};

  for (const auto& [n_r, l_max] : mesh_sizes) {
    CAPTURE(n_r);
    CAPTURE(l_max);
    const Mesh<3> mesh = b3_mesh(n_r, l_max);
    const size_t n_r_max = 2 * n_r - 2;

    gh::power_monitor::CartToSphereMatrices cart_to_sphere{};

    for (size_t l_test = 0; l_test <= l_max; ++l_test) {
      CAPTURE(l_test);
      const size_t spectral_size_l = (n_r_max - l_test + 2) / 2;
      // Test m=0 and (when l_test >= 1) m=1 to exercise non-zero azimuthal
      // modes. The expected PM values are independent of m since the monitor
      // bins by angular degree l only.
      const size_t m_max_test = std::min(l_test, size_t{1});

      for (size_t m_test = 0; m_test <= m_max_test; ++m_test) {
        CAPTURE(m_test);
        const size_t s = spherepack_offset(l_max, l_test, m_test);

        for (size_t k = 0; k < spectral_size_l; ++k) {
          CAPTURE(k);
          const size_t expected_radial_mode = (l_test + 2 * k + 1) / 2;

          // Set metric_00 to pure B3 mode (n_jacobi=k, l=l_test, m=m_test,
          // cos).
          auto gh_vars = zero_gh_vars(mesh);
          get<0, 0>(
              get<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(
                  gh_vars)) = pure_b3_mode_phys(mesh, k, l_test, s);

          const auto monitors =
              gh::power_monitor::gh_filled_sphere_power_monitors(
                  make_not_null(&cart_to_sphere), gh_vars, mesh,
                  identity_jacobian(mesh.number_of_grid_points()));

          // Metric angular power: non-zero only at l_test, with exact value.
          CHECK(monitors.spacetime_metric[1].size() == l_max + 1_st);
          CHECK(monitors.spacetime_metric[1][l_test] ==
                approx(expected_b3_angular_pm(l_test, n_r_max)));
          for (size_t l = 0; l <= l_max; ++l) {
            if (l != l_test) {
              CHECK(monitors.spacetime_metric[1][l] == approx(0.0));
            }
          }

          // Metric radial power: non-zero only at expected_radial_mode, with
          // exact value.
          CHECK(monitors.spacetime_metric[0].size() == n_r);
          CHECK(monitors.spacetime_metric[0][expected_radial_mode] ==
                approx(expected_b3_radial_pm(expected_radial_mode, l_max)));
          for (size_t mode = 0; mode < n_r; ++mode) {
            if (mode != expected_radial_mode) {
              CHECK(monitors.spacetime_metric[0][mode] == approx(0.0));
            }
          }

          // Pi and Phi monitors must be zero (no data was set there).
          CHECK(max(abs(monitors.pi[0])) == 0.0);
          CHECK(max(abs(monitors.pi[1])) == 0.0);
          CHECK(max(abs(monitors.phi[0])) == 0.0);
          CHECK(max(abs(monitors.phi[1])) == 0.0);
        }
      }
    }
  }
}

// Test that setting pi_00 or phi_k00 produces power in the corresponding GH
// variable monitor and zero in the others.
void test_variable_isolation() {
  const Mesh<3> mesh = b3_mesh(3, 2);
  const size_t l_max = 2;
  gh::power_monitor::CartToSphereMatrices cart_to_sphere{};

  // Pi_00 is a scalar (spin weight 0); excite with an l=0, k=0 mode.
  // Pi00 accumulates identically to Metric00: exact values follow from
  // expected_b3_angular_pm / expected_b3_radial_pm with l_test=0, r=0.
  {
    const size_t n_r_max = 2 * mesh.extents(0) - 2;
    const DataVector mode =
        pure_b3_mode_phys(mesh, 0, 0, spherepack_offset(l_max, 0));
    auto gh_vars = zero_gh_vars(mesh);
    get<0, 0>(get<gh::Tags::Pi<DataVector, 3, Frame::Inertial>>(gh_vars)) =
        mode;
    const auto monitors = gh::power_monitor::gh_filled_sphere_power_monitors(
        make_not_null(&cart_to_sphere), gh_vars, mesh,
        identity_jacobian(mesh.number_of_grid_points()));
    CHECK(max(abs(monitors.spacetime_metric[0])) == 0.0);
    CHECK(max(abs(monitors.spacetime_metric[1])) == 0.0);
    CHECK(monitors.pi[0][0] == approx(expected_b3_radial_pm(0, l_max)));
    CHECK(monitors.pi[1][0] == approx(expected_b3_angular_pm(0, n_r_max)));
    CHECK(max(abs(monitors.phi[0])) == 0.0);
    CHECK(max(abs(monitors.phi[1])) == 0.0);
  }

  // Phi_x00 maps to Phik00 (spin weight 1, so minimum angular degree l=1).
  // Use an l=1 physical mode so it survives the spin-weight filter.
  // The cart-to-sphere transform may distribute power across l=1 and l=2, so
  // check that the total phi monitor is nonzero and that metric/pi are zero.
  {
    const DataVector mode =
        pure_b3_mode_phys(mesh, 0, 1, spherepack_offset(l_max, 1));
    auto gh_vars = zero_gh_vars(mesh);
    get<0, 0, 0>(get<gh::Tags::Phi<DataVector, 3, Frame::Inertial>>(gh_vars)) =
        mode;
    const auto monitors = gh::power_monitor::gh_filled_sphere_power_monitors(
        make_not_null(&cart_to_sphere), gh_vars, mesh,
        identity_jacobian(mesh.number_of_grid_points()));
    CHECK(max(abs(monitors.spacetime_metric[0])) == 0.0);
    CHECK(max(abs(monitors.spacetime_metric[1])) == 0.0);
    CHECK(max(abs(monitors.pi[0])) == 0.0);
    CHECK(max(abs(monitors.pi[1])) == 0.0);
    CHECK(max(abs(monitors.phi[0])) > 1.0e-3);
    CHECK(max(abs(monitors.phi[1])) > 1.0e-3);
  }
}

// Test that angular monitor bins with l < |spin_weight| are exactly zero,
// verifying the spin-weighted spherical harmonic suppression convention.
void test_spin_weight_suppression() {
  const Mesh<3> mesh = b3_mesh(3, 2);
  const size_t l_max = 2;
  gh::power_monitor::CartToSphereMatrices cart_to_sphere{};

  // Excite a traceless combination g_xx = -g_yy with a pure l=0 physical mode.
  // The traceless part of a symmetric rank-2 spatial tensor is spin weight 2,
  // so angular bins l=0 and l=1 must be zero. The trace (Metric00, spin
  // weight 0) is zero by construction, so it cannot contaminate the l=0 bin.
  // The TensorYlm rank-2 coupling maps l_in=0 to l_out=2 (the only value
  // satisfying l_out >= |s|=2), so angular bin l=2 must be nonzero.
  {
    const DataVector mode_l0 =
        pure_b3_mode_phys(mesh, 0, 0, spherepack_offset(l_max, 0));
    auto gh_vars = zero_gh_vars(mesh);
    get<1, 1>(get<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(
        gh_vars)) = mode_l0;
    get<2, 2>(get<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(
        gh_vars)) = -mode_l0;
    const auto monitors = gh::power_monitor::gh_filled_sphere_power_monitors(
        make_not_null(&cart_to_sphere), gh_vars, mesh,
        identity_jacobian(mesh.number_of_grid_points()));
    CHECK(monitors.spacetime_metric[1][0] == approx(0.0));
    CHECK(monitors.spacetime_metric[1][1] == approx(0.0));
    CHECK(monitors.spacetime_metric[1][2] > 1.0e-3);
  }

  // Excite Phi_x00 (maps to Phik00, spin weight 1) with a pure l=0 physical
  // mode.  The l=0 TensorYlm mode does not exist for spin weight 1, so the
  // angular bin at l=0 must be zero regardless of the radial content.
  {
    const DataVector mode_l0 =
        pure_b3_mode_phys(mesh, 0, 0, spherepack_offset(l_max, 0));
    auto gh_vars = zero_gh_vars(mesh);
    get<0, 0, 0>(get<gh::Tags::Phi<DataVector, 3, Frame::Inertial>>(gh_vars)) =
        mode_l0;
    const auto monitors = gh::power_monitor::gh_filled_sphere_power_monitors(
        make_not_null(&cart_to_sphere), gh_vars, mesh,
        identity_jacobian(mesh.number_of_grid_points()));
    CHECK(monitors.phi[1][0] == approx(0.0));
    CHECK(max(abs(monitors.phi[1])) > 1.0e-3);
  }
}

// Test Pythagorean property: two orthogonal B3 modes add in quadrature in both
// the angular and radial monitors.
//   - Angular: modes at l=0 and l=1 are in disjoint angular bins.
//   - Radial:  modes at (l=0,k=0) -> r=0 and (l=0,k=1) -> r=1 are in disjoint
//     radial bins.
// In both cases, combining the two modes leaves each individual bin unchanged.
void test_pythagorean() {
  const Mesh<3> mesh = b3_mesh(3, 2);
  const size_t l_max = 2;
  const size_t s0 = spherepack_offset(l_max, 0);
  gh::power_monitor::CartToSphereMatrices cart_to_sphere{};

  // Angular-orthogonal pair: l=0 (k=0 -> r=0) and l=1 (k=0 -> r=1).
  const DataVector mode_l0 = pure_b3_mode_phys(mesh, 0, 0, s0);
  const DataVector mode_l1 =
      pure_b3_mode_phys(mesh, 0, 1, spherepack_offset(l_max, 1));
  // Radial-orthogonal pair: both l=0 but k=0 -> r=0 and k=1 -> r=1.
  const DataVector mode_r1 = pure_b3_mode_phys(mesh, 1, 0, s0);

  auto set_metric00 = [&](const DataVector& v) {
    auto vars = zero_gh_vars(mesh);
    get<0, 0>(get<gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(
        vars)) = v;
    return gh::power_monitor::gh_filled_sphere_power_monitors(
        make_not_null(&cart_to_sphere), vars, mesh,
        identity_jacobian(mesh.number_of_grid_points()));
  };

  const auto pm_l0 = set_metric00(mode_l0);
  const auto pm_l1 = set_metric00(mode_l1);
  const auto pm_l01 = set_metric00(mode_l0 + mode_l1);
  const auto pm_r1 = set_metric00(mode_r1);
  const auto pm_r01 = set_metric00(mode_l0 + mode_r1);

  // Angular monitor: l=0 and l=1 are orthogonal bins.
  CHECK(square(pm_l01.spacetime_metric[1][0]) ==
        approx(square(pm_l0.spacetime_metric[1][0])));
  CHECK(square(pm_l01.spacetime_metric[1][1]) ==
        approx(square(pm_l1.spacetime_metric[1][1])));

  // Radial monitor: r=0 and r=1 are orthogonal bins.
  CHECK(square(pm_r01.spacetime_metric[0][0]) ==
        approx(square(pm_l0.spacetime_metric[0][0])));
  CHECK(square(pm_r01.spacetime_metric[0][1]) ==
        approx(square(pm_r1.spacetime_metric[0][1])));
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.FilledSpherePowerMonitor",
    "[Unit][Evolution]") {
  test_zero_input();
  test_mode_isolation();
  test_spin_weight_suppression();
  test_variable_isolation();
  test_pythagorean();
}

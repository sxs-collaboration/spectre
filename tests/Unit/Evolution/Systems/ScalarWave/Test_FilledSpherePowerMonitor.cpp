// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarWave/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/ScalarWave/FilledSpherePowerMonitor.hpp"
#include "Evolution/Systems/ScalarWave/SphericalShellPowerMonitor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {

// Number of SH-mode slots at angular degree l for a real scalar field
// (zero_m_is_real = true): m=0 contributes 1 (cosine only), m=1..l each
// contribute 2 (cosine + sine), for a total of 2l+1.
size_t scalar_sh_modes_at_l(const size_t l) { return 2 * l + 1; }

// Expected angular power monitor at degree l for a single unit-coefficient
// B3 mode in a 1-component scalar field (Psi or Pi).
double expected_scalar_angular_pm(const size_t l, const size_t n_r_max) {
  const size_t spectral_size_l = (n_r_max - l + 2) / 2;
  return 1.0 /
         sqrt(static_cast<double>(spectral_size_l * scalar_sh_modes_at_l(l)));
}

// Expected radial power monitor at bin r for a single unit-coefficient B3
// mode in a 1-component scalar field.  Every (l', k') pair that maps to the
// same radial bin contributes scalar_sh_modes_at_l(l') weight.
double expected_scalar_radial_pm(const size_t r, const size_t l_max) {
  size_t count = 0;
  for (size_t l = 0; l <= std::min(2 * r, l_max); l += 2) {
    count += scalar_sh_modes_at_l(l);
  }
  if (r >= 1) {
    for (size_t l = 1; l <= std::min(2 * r - 1, l_max); l += 2) {
      count += scalar_sh_modes_at_l(l);
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

  DataVector spec_buf(n_spectral * n_r, 0.0);
  for (size_t i_r = 0; i_r < n_r; ++i_r) {
    spec_buf[s * n_r + i_r] = radial_profile[i_r];
  }

  DataVector phys(n_phys);
  ylm.spec_to_phys_all_offsets(make_not_null(phys.data()),
                               make_not_null(spec_buf.data()), n_r);
  return phys;
}

// Returns the flat SPHEREPACK buffer offset for the (l_ang, m, cosine) mode.
size_t spherepack_offset(const size_t l_max, const size_t l_ang,
                         const size_t m = 0) {
  ylm::SpherepackIterator iter{l_max, l_max};
  iter.set(l_ang, m, ylm::SpherepackIterator::CoefficientArray::a);
  return iter();
}

InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid> identity_jacobian(
    const size_t size) {
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid> jacobian{size,
                                                                        0.0};
  for (size_t i = 0; i < 3; ++i) {
    jacobian.get(i, i) = 1.0;
  }
  return jacobian;
}

Variables<ylm::TensorYlm::filter_detail::sw_vars_list<Frame::Inertial>>
zero_sw_vars(const Mesh<3>& mesh) {
  return {mesh.number_of_grid_points(), 0.0};
}

void test_zero_input() {
  const Mesh<3> mesh = b3_mesh(3, 2);
  auto sw_vars = zero_sw_vars(mesh);
  ScalarWave::power_monitor::SwCartToSphereMatrix cart_to_sphere{};
  const auto monitors =
      ScalarWave::power_monitor::sw_filled_sphere_power_monitors(
          make_not_null(&cart_to_sphere), sw_vars, mesh,
          identity_jacobian(mesh.number_of_grid_points()));

  CHECK(max(abs(monitors.psi[0])) == 0.0);
  CHECK(max(abs(monitors.psi[1])) == 0.0);
  CHECK(max(abs(monitors.pi[0])) == 0.0);
  CHECK(max(abs(monitors.pi[1])) == 0.0);
  CHECK(max(abs(monitors.phi[0])) == 0.0);
  CHECK(max(abs(monitors.phi[1])) == 0.0);
}

// Test mode isolation for Psi: a pure B3 mode produces power only at the
// expected angular degree and radial bin, with exact normalisation values.
// Pi and Phi monitors remain zero.
void test_psi_mode_isolation() {
  const std::vector<std::pair<size_t, size_t>> mesh_sizes{
      {2, 2}, {3, 2}, {3, 4}, {4, 4}};

  for (const auto& [n_r, l_max] : mesh_sizes) {
    CAPTURE(n_r);
    CAPTURE(l_max);
    const Mesh<3> mesh = b3_mesh(n_r, l_max);
    const size_t n_r_max = 2 * n_r - 2;

    ScalarWave::power_monitor::SwCartToSphereMatrix cart_to_sphere{};

    for (size_t l_test = 0; l_test <= l_max; ++l_test) {
      CAPTURE(l_test);
      const size_t spectral_size_l = (n_r_max - l_test + 2) / 2;
      // Test m=0 and (when l_test >= 1) m=1 to exercise non-zero azimuthal
      // modes.  Expected values depend only on l, not m.
      const size_t m_max_test = std::min(l_test, size_t{1});

      for (size_t m_test = 0; m_test <= m_max_test; ++m_test) {
        CAPTURE(m_test);
        const size_t s = spherepack_offset(l_max, l_test, m_test);

        for (size_t k = 0; k < spectral_size_l; ++k) {
          CAPTURE(k);
          const size_t expected_radial_mode = (l_test + 2 * k + 1) / 2;

          auto sw_vars = zero_sw_vars(mesh);
          get(get<ScalarWave::Tags::Psi>(sw_vars)) =
              pure_b3_mode_phys(mesh, k, l_test, s);

          const auto monitors =
              ScalarWave::power_monitor::sw_filled_sphere_power_monitors(
                  make_not_null(&cart_to_sphere), sw_vars, mesh,
                  identity_jacobian(mesh.number_of_grid_points()));

          // Angular: non-zero only at l_test.
          CHECK(monitors.psi[1].size() == l_max + 1_st);
          CHECK(monitors.psi[1][l_test] ==
                approx(expected_scalar_angular_pm(l_test, n_r_max)));
          for (size_t l = 0; l <= l_max; ++l) {
            if (l != l_test) {
              CHECK(monitors.psi[1][l] == approx(0.0));
            }
          }

          // Radial: non-zero only at expected_radial_mode.
          CHECK(monitors.psi[0].size() == n_r);
          CHECK(monitors.psi[0][expected_radial_mode] ==
                approx(expected_scalar_radial_pm(expected_radial_mode, l_max)));
          for (size_t mode = 0; mode < n_r; ++mode) {
            if (mode != expected_radial_mode) {
              CHECK(monitors.psi[0][mode] == approx(0.0));
            }
          }

          // Pi and Phi must be zero.
          CHECK(max(abs(monitors.pi[0])) == 0.0);
          CHECK(max(abs(monitors.pi[1])) == 0.0);
          CHECK(max(abs(monitors.phi[0])) == 0.0);
          CHECK(max(abs(monitors.phi[1])) == 0.0);
        }
      }
    }
  }
}

// Test that setting Pi or Phi_x independently produces nonzero power in the
// correct monitor and leaves the other two monitors at zero.
void test_variable_isolation() {
  const Mesh<3> mesh = b3_mesh(3, 2);
  const size_t l_max = 2;
  const size_t n_r_max = 2 * mesh.extents(0) - 2;
  ScalarWave::power_monitor::SwCartToSphereMatrix cart_to_sphere{};

  // Pi (scalar, spin weight 0): same mode structure as Psi.
  {
    const DataVector mode =
        pure_b3_mode_phys(mesh, 0, 0, spherepack_offset(l_max, 0));
    auto sw_vars = zero_sw_vars(mesh);
    get(get<ScalarWave::Tags::Pi>(sw_vars)) = mode;
    const auto monitors =
        ScalarWave::power_monitor::sw_filled_sphere_power_monitors(
            make_not_null(&cart_to_sphere), sw_vars, mesh,
            identity_jacobian(mesh.number_of_grid_points()));
    CHECK(max(abs(monitors.psi[0])) == 0.0);
    CHECK(max(abs(monitors.psi[1])) == 0.0);
    CHECK(monitors.pi[0][0] == approx(expected_scalar_radial_pm(0, l_max)));
    CHECK(monitors.pi[1][0] == approx(expected_scalar_angular_pm(0, n_r_max)));
    CHECK(max(abs(monitors.phi[0])) == 0.0);
    CHECK(max(abs(monitors.phi[1])) == 0.0);
  }

  // Phi_x (rank-1, spin weight 1): excite with an l=1 physical mode so it
  // survives the spin-weight filter.  Only verify that the Phi monitor is
  // nonzero and that Psi and Pi remain zero.
  {
    const DataVector mode =
        pure_b3_mode_phys(mesh, 0, 1, spherepack_offset(l_max, 1));
    auto sw_vars = zero_sw_vars(mesh);
    get<0>(get<ScalarWave::Tags::Phi<3>>(sw_vars)) = mode;
    const auto monitors =
        ScalarWave::power_monitor::sw_filled_sphere_power_monitors(
            make_not_null(&cart_to_sphere), sw_vars, mesh,
            identity_jacobian(mesh.number_of_grid_points()));
    CHECK(max(abs(monitors.psi[0])) == 0.0);
    CHECK(max(abs(monitors.psi[1])) == 0.0);
    CHECK(max(abs(monitors.pi[0])) == 0.0);
    CHECK(max(abs(monitors.pi[1])) == 0.0);
    CHECK(max(abs(monitors.phi[0])) > 1.0e-3);
    CHECK(max(abs(monitors.phi[1])) > 1.0e-3);
  }
}

// Test the Pythagorean property for Psi: orthogonal B3 modes (in disjoint
// angular or radial bins) add in quadrature.
void test_pythagorean() {
  const Mesh<3> mesh = b3_mesh(3, 2);
  const size_t l_max = 2;
  const size_t s0 = spherepack_offset(l_max, 0);
  ScalarWave::power_monitor::SwCartToSphereMatrix cart_to_sphere{};

  // Angular-orthogonal pair: l=0 (k=0 -> r=0) and l=1 (k=0 -> r=1).
  const DataVector mode_l0 = pure_b3_mode_phys(mesh, 0, 0, s0);
  const DataVector mode_l1 =
      pure_b3_mode_phys(mesh, 0, 1, spherepack_offset(l_max, 1));
  // Radial-orthogonal pair: l=0, k=0 -> r=0 and l=0, k=1 -> r=1.
  const DataVector mode_r1 = pure_b3_mode_phys(mesh, 1, 0, s0);

  auto run = [&](const DataVector& v) {
    auto vars = zero_sw_vars(mesh);
    get(get<ScalarWave::Tags::Psi>(vars)) = v;
    return ScalarWave::power_monitor::sw_filled_sphere_power_monitors(
        make_not_null(&cart_to_sphere), vars, mesh,
        identity_jacobian(mesh.number_of_grid_points()));
  };

  const auto pm_l0 = run(mode_l0);
  const auto pm_l1 = run(mode_l1);
  const auto pm_l01 = run(mode_l0 + mode_l1);
  const auto pm_r1 = run(mode_r1);
  const auto pm_r01 = run(mode_l0 + mode_r1);

  // Angular bins l=0 and l=1 are orthogonal.
  CHECK(square(pm_l01.psi[1][0]) == approx(square(pm_l0.psi[1][0])));
  CHECK(square(pm_l01.psi[1][1]) == approx(square(pm_l1.psi[1][1])));

  // Radial bins r=0 and r=1 are orthogonal.
  CHECK(square(pm_r01.psi[0][0]) == approx(square(pm_l0.psi[0][0])));
  CHECK(square(pm_r01.psi[0][1]) == approx(square(pm_r1.psi[0][1])));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.FilledSpherePowerMonitor",
                  "[Unit][Evolution]") {
  test_zero_input();
  test_psi_mode_isolation();
  test_variable_isolation();
  test_pythagorean();
}

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/ZernikeB2.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {

void test_b2_disk_power_monitors() {
  // Test structural properties of both angular and radial power monitors on a
  // 2D disk over several mesh sizes (n_phi odd, M = n_phi/2 <= 2*n_r - 2).
  // Exact single-mode values are verified by
  // test_b2_power_monitor_exact_values; here we check properties that require
  // sin modes or multiple modes:
  //   (a) sin mode: zero at all slots except the excited one,
  //   (b) cos and sin carry equal power (Fourier symmetry),
  //   (c) cos+sin has sqrt(2) times the single-component power,
  //   (d) Pythagorean addition for distinct modes sharing a radial level.
  const std::vector<std::pair<size_t, size_t>> mesh_sizes{
      {2, 3}, {3, 5}, {3, 9}, {4, 5}, {5, 7}};

  for (const auto& [n_r, n_phi] : mesh_sizes) {
    CAPTURE(n_r);
    CAPTURE(n_phi);
    const size_t M = n_phi / 2;
    const Mesh<2> mesh{{n_r, n_phi},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular}};
    const auto xi = logical_coordinates(mesh);
    const DataVector r = 0.5 * (get<0>(xi) + 1.0);
    const DataVector& phi = get<1>(xi);

    for (size_t m_test = 1; m_test <= M; ++m_test) {
      CAPTURE(m_test);
      const size_t ell_test = (m_test + 1) / 2;
      const DataVector r_m = pow(r, static_cast<double>(m_test));
      const DataVector f_cos = r_m * cos(static_cast<double>(m_test) * phi);
      const DataVector f_sin = r_m * sin(static_cast<double>(m_test) * phi);

      DataVector pm_ang_cos;
      DataVector pm_ang_sin;
      DataVector pm_ang_both;
      DataVector pm_rad_cos;
      DataVector pm_rad_sin;
      DataVector pm_rad_both;
      Spectral::b2_power_monitor_angular(make_not_null(&pm_ang_cos), f_cos,
                                         mesh);
      Spectral::b2_power_monitor_angular(make_not_null(&pm_ang_sin), f_sin,
                                         mesh);
      Spectral::b2_power_monitor_angular(make_not_null(&pm_ang_both),
                                         f_cos + f_sin, mesh);
      Spectral::b2_power_monitor_radial(make_not_null(&pm_rad_cos), f_cos,
                                        mesh);
      Spectral::b2_power_monitor_radial(make_not_null(&pm_rad_sin), f_sin,
                                        mesh);
      Spectral::b2_power_monitor_radial(make_not_null(&pm_rad_both),
                                        f_cos + f_sin, mesh);

      // (a) Sin mode: zero at every slot except the excited one.
      for (size_t m = 0; m <= M; ++m) {
        if (m != m_test) {
          CHECK(pm_ang_sin[m] == approx(0.0));
        }
      }
      for (size_t ell = 0; ell < n_r; ++ell) {
        if (ell != ell_test) {
          CHECK(pm_rad_sin[ell] == approx(0.0));
        }
      }

      // (b) Cos and sin carry equal power.
      CHECK(pm_ang_cos[m_test] == approx(pm_ang_sin[m_test]));
      CHECK(pm_rad_cos[ell_test] == approx(pm_rad_sin[ell_test]));

      // (c) Cos+sin has sqrt(2) times the single-component power.
      CHECK(pm_ang_both[m_test] == approx(std::sqrt(2.0) * pm_ang_cos[m_test]));
      CHECK(pm_rad_both[ell_test] ==
            approx(std::sqrt(2.0) * pm_rad_cos[ell_test]));
    }

    // (d) Pythagorean addition for modes at the same radial level.
    // m=1 -> ell=1 and m=2 -> ell=1, so
    //        |pm(f1+f2)|^2 = |pm(f1)|^2 + |pm(f2)|^2.
    if (M >= 2) {
      const DataVector f1 = r * cos(phi);                // (n=1, m=1) -> ell=1
      const DataVector f2 = square(r) * cos(2.0 * phi);  // (n=2, m=2) -> ell=1
      DataVector pm1;
      DataVector pm2;
      DataVector pm12;
      Spectral::b2_power_monitor_radial(make_not_null(&pm1), f1, mesh);
      Spectral::b2_power_monitor_radial(make_not_null(&pm2), f2, mesh);
      Spectral::b2_power_monitor_radial(make_not_null(&pm12), f1 + f2, mesh);
      CHECK(square(pm12[1]) == approx(square(pm1[1]) + square(pm2[1])));
    }
  }
}

void test_b2_cylinder_power_monitors() {
  // Test that the Mesh<3> (cylinder) power monitors agree with the Mesh<2>
  // (disk) monitors when n_z = 1, and that z-stacking two identical disks
  // gives the same angular and radial power as a single disk.
  const std::vector<std::pair<size_t, size_t>> disk_sizes{
      {2, 3}, {3, 5}, {4, 7}};
  const std::vector<size_t> z_sizes{1, 2, 3};

  for (const auto& [n_r, n_phi] : disk_sizes) {
    CAPTURE(n_r);
    CAPTURE(n_phi);
    const size_t M = n_phi / 2;
    const Mesh<2> disk_mesh{
        {n_r, n_phi},
        {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
        {Spectral::Quadrature::GaussRadauUpper,
         Spectral::Quadrature::Equiangular}};
    const auto xi = logical_coordinates(disk_mesh);
    const DataVector r = 0.5 * (get<0>(xi) + 1.0);
    const DataVector& phi = get<1>(xi);
    // Use a non-trivial test function with several modes excited.
    const DataVector u_disk = 1.0 + r * cos(phi) + square(r) * cos(2.0 * phi);
    DataVector pm_disk;
    DataVector pm_radial_disk;
    Spectral::b2_power_monitor_angular(make_not_null(&pm_disk), u_disk,
                                       disk_mesh);
    Spectral::b2_power_monitor_radial(make_not_null(&pm_radial_disk), u_disk,
                                      disk_mesh);

    for (const size_t n_z : z_sizes) {
      CAPTURE(n_z);
      const Mesh<3> cyl_mesh{
          {n_r, n_phi, n_z},
          {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
           Spectral::Basis::Legendre},
          {Spectral::Quadrature::GaussRadauUpper,
           Spectral::Quadrature::Equiangular,
           Spectral::Quadrature::GaussLobatto}};

      // Construct cylinder data by repeating u_disk for every z-slice.
      DataVector u_cyl(cyl_mesh.number_of_grid_points(), 0.0);
      for (size_t k_z = 0; k_z < n_z; ++k_z) {
        for (size_t i = 0; i < n_r * n_phi; ++i) {
          u_cyl[k_z * n_r * n_phi + i] = u_disk[i];
        }
      }

      // Angular monitor: cylinder result must equal disk result (same data
      // repeated in z, so the RMS over z is the same as for one slice).
      DataVector pm_cyl;
      Spectral::b2_power_monitor_angular(make_not_null(&pm_cyl), u_cyl,
                                         cyl_mesh);
      CHECK(pm_cyl.size() == M + 1_st);
      CHECK_ITERABLE_APPROX(pm_cyl, pm_disk);

      // Radial monitor: same reasoning.
      DataVector pm_radial_cyl;
      Spectral::b2_power_monitor_radial(make_not_null(&pm_radial_cyl), u_cyl,
                                        cyl_mesh);
      CHECK(pm_radial_cyl.size() == n_r);
      CHECK_ITERABLE_APPROX(pm_radial_cyl, pm_radial_disk);
    }

    // When two *different* z-slices are stacked, the Pythagorean law holds
    // in z: P_cyl^2 = (P_disk1^2 + P_disk2^2) / 2.
    {
      const size_t n_z2 = 2;
      const Mesh<3> cyl_mesh2{
          {n_r, n_phi, n_z2},
          {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
           Spectral::Basis::Legendre},
          {Spectral::Quadrature::GaussRadauUpper,
           Spectral::Quadrature::Equiangular,
           Spectral::Quadrature::GaussLobatto}};

      const DataVector u2_disk = r * cos(phi);  // different second slice
      DataVector pm2_disk;
      DataVector pm2_radial_disk;
      Spectral::b2_power_monitor_angular(make_not_null(&pm2_disk), u2_disk,
                                         disk_mesh);
      Spectral::b2_power_monitor_radial(make_not_null(&pm2_radial_disk),
                                        u2_disk, disk_mesh);

      DataVector u2_cyl(cyl_mesh2.number_of_grid_points(), 0.0);
      for (size_t i = 0; i < n_r * n_phi; ++i) {
        u2_cyl[i] = u_disk[i];
        u2_cyl[n_r * n_phi + i] = u2_disk[i];
      }

      DataVector pm2_cyl;
      DataVector pm2_radial_cyl;
      Spectral::b2_power_monitor_angular(make_not_null(&pm2_cyl), u2_cyl,
                                         cyl_mesh2);
      Spectral::b2_power_monitor_radial(make_not_null(&pm2_radial_cyl), u2_cyl,
                                        cyl_mesh2);

      for (size_t m = 0; m <= M; ++m) {
        // P_cyl^2 = (P1^2 + P2^2) / 2
        CHECK(square(pm2_cyl[m]) ==
              approx(0.5 * (square(pm_disk[m]) + square(pm2_disk[m]))));
      }
      for (size_t ell = 0; ell < n_r; ++ell) {
        CHECK(square(pm2_radial_cyl[ell]) ==
              approx(0.5 * (square(pm_radial_disk[ell]) +
                            square(pm2_radial_disk[ell]))));
      }
    }
  }
}
// Count spectral slots at radial level ell_0 for a ZernikeB2 disk.
// m=0 modes contribute 1 slot each; m>0 modes contribute 2 (cos + sin).
// Each mode (m, k) maps to ell = (m + 2*k + 1) / 2 (integer division).
size_t count_b2_radial_slots(const size_t ell_0, const size_t n_r_max,
                             const size_t M) {
  const size_t spectral_size_m0 = (n_r_max + 2) / 2;
  size_t slots = 0;
  if (ell_0 < spectral_size_m0) {
    ++slots;  // one m=0 mode at this level
  }
  for (size_t m = 1; m <= M; ++m) {
    const size_t sm = (n_r_max - m + 2) / 2;
    for (size_t k = 0; k < sm; ++k) {
      if ((m + 2 * k + 1) / 2 == ell_0) {
        slots += 2;
      }
    }
  }
  return slots;
}

size_t count_b2_angular_slots(const size_t m, const size_t n_r_max) {
  const size_t spectral_size_m0 = (n_r_max + 2) / 2;
  return m == 0 ? spectral_size_m0 : 2 * ((n_r_max - m + 2) / 2);
}

// Test b2_power_monitor_radial and b2_power_monitor_angular on a 2D disk with
// the pure mode u = phi_n^m(r) * cos(m*phi), where phi_n^m is the orthonormal
// ZernikeB2 basis function.  By orthonormality, the (n,m,cos) spectral
// coefficient equals 1 and all others equal 0, so expected power monitor
// values follow directly from slot counts:
//   pm_radial[ell_0] = 1 / sqrt(slots_rad),
//   pm_angular[m]    = 1 / sqrt(slots_ang).
void test_b2_disk_mode(const size_t n_zernike, const size_t m_zernike,
                       const size_t n_r, const size_t n_phi) {
  CAPTURE(n_zernike);
  CAPTURE(m_zernike);
  CAPTURE(n_r);
  CAPTURE(n_phi);
  const size_t n_r_max = 2 * n_r - 2;
  const size_t M = n_phi / 2;
  const Mesh<2> mesh{{n_r, n_phi},
                     {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                     {Spectral::Quadrature::GaussRadauUpper,
                      Spectral::Quadrature::Equiangular}};
  const DataVector xi_r = Spectral::collocation_points(mesh.slice_through(0));
  const DataVector phi_pts =
      Spectral::collocation_points(mesh.slice_through(1));
  const DataVector phi_nm_at_r =
      Spectral::compute_basis_function_value<Spectral::Basis::ZernikeB2>(
          n_zernike, m_zernike, xi_r);
  // Build u = phi_n^m(r) * cos(m*phi); index layout u[i_r + n_r*i_phi].
  DataVector u(mesh.number_of_grid_points());
  for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
    const double cos_mphi =
        cos(static_cast<double>(m_zernike) * phi_pts[i_phi]);
    for (size_t i_r = 0; i_r < n_r; ++i_r) {
      u[i_r + n_r * i_phi] = phi_nm_at_r[i_r] * cos_mphi;
    }
  }
  const size_t ell_0 = (n_zernike + 1) / 2;
  DataVector expected_radial(n_r, 0.0);
  expected_radial[ell_0] =
      1.0 / sqrt(static_cast<double>(count_b2_radial_slots(ell_0, n_r_max, M)));
  DataVector expected_angular(M + 1, 0.0);
  expected_angular[m_zernike] =
      1.0 /
      sqrt(static_cast<double>(count_b2_angular_slots(m_zernike, n_r_max)));
  DataVector pm_radial;
  DataVector pm_angular;
  Spectral::b2_power_monitor_radial(make_not_null(&pm_radial), u, mesh);
  Spectral::b2_power_monitor_angular(make_not_null(&pm_angular), u, mesh);
  CHECK(pm_radial.size() == n_r);
  CHECK(pm_angular.size() == M + 1);
  CHECK_ITERABLE_APPROX(pm_radial, expected_radial);
  CHECK_ITERABLE_APPROX(pm_angular, expected_angular);
}

// Same test for Mesh<3> (cylinder) overloads with
// u = phi_n^m(r) * cos(m*phi) * P_{k'}(z).
// The disk spectral coefficient equals P_{k'}(z) per z-slice, so the
// z-averaged power monitor uses sum_Pk_sq = sum_i P_{k'}(z_i)^2:
//   pm_radial[ell_0] = sqrt(sum_Pk_sq / (n_z * slots_rad)),
//   pm_angular[m]    = sqrt(sum_Pk_sq / (n_z * slots_ang)).
void test_b2_cylinder_mode(const size_t n_zernike, const size_t m_zernike,
                           const size_t k_legendre, const size_t n_r,
                           const size_t n_phi, const size_t n_z) {
  CAPTURE(n_zernike);
  CAPTURE(m_zernike);
  CAPTURE(k_legendre);
  CAPTURE(n_r);
  CAPTURE(n_phi);
  CAPTURE(n_z);
  const size_t n_r_max = 2 * n_r - 2;
  const size_t M = n_phi / 2;
  const Mesh<3> cyl_mesh{
      {n_r, n_phi, n_z},
      {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
       Spectral::Basis::Legendre},
      {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Equiangular,
       Spectral::Quadrature::GaussLobatto}};
  const DataVector xi_r =
      Spectral::collocation_points(cyl_mesh.slice_through(0));
  const DataVector phi_pts =
      Spectral::collocation_points(cyl_mesh.slice_through(1));
  const DataVector z_pts =
      Spectral::collocation_points(cyl_mesh.slice_through(2));
  const DataVector phi_nm_at_r =
      Spectral::compute_basis_function_value<Spectral::Basis::ZernikeB2>(
          n_zernike, m_zernike, xi_r);
  const DataVector P_k_at_z =
      Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
          k_legendre, z_pts);
  DataVector u(cyl_mesh.number_of_grid_points());
  for (size_t i_z = 0; i_z < n_z; ++i_z) {
    for (size_t i_phi = 0; i_phi < n_phi; ++i_phi) {
      const double cos_mphi =
          cos(static_cast<double>(m_zernike) * phi_pts[i_phi]);
      for (size_t i_r = 0; i_r < n_r; ++i_r) {
        u[i_r + n_r * (i_phi + n_phi * i_z)] =
            phi_nm_at_r[i_r] * cos_mphi * P_k_at_z[i_z];
      }
    }
  }
  double sum_Pk_sq = 0.0;
  for (size_t i = 0; i < n_z; ++i) {
    sum_Pk_sq += square(P_k_at_z[i]);
  }
  const size_t ell_0 = (n_zernike + 1) / 2;
  DataVector expected_radial(n_r, 0.0);
  expected_radial[ell_0] =
      sqrt(sum_Pk_sq /
           (static_cast<double>(n_z) *
            static_cast<double>(count_b2_radial_slots(ell_0, n_r_max, M))));
  DataVector expected_angular(M + 1, 0.0);
  expected_angular[m_zernike] =
      sqrt(sum_Pk_sq /
           (static_cast<double>(n_z) *
            static_cast<double>(count_b2_angular_slots(m_zernike, n_r_max))));
  DataVector pm_radial;
  DataVector pm_angular;
  Spectral::b2_power_monitor_radial(make_not_null(&pm_radial), u, cyl_mesh);
  Spectral::b2_power_monitor_angular(make_not_null(&pm_angular), u, cyl_mesh);
  CHECK(pm_radial.size() == n_r);
  CHECK(pm_angular.size() == M + 1);
  CHECK_ITERABLE_APPROX(pm_radial, expected_radial);
  CHECK_ITERABLE_APPROX(pm_angular, expected_angular);
}

void test_b2_power_monitor_exact_values() {
  // 2D disk: (n_zernike, m_zernike, n_r, n_phi)
  test_b2_disk_mode(0, 0, 2, 3);  // monopole (n=m=0)
  test_b2_disk_mode(1, 1, 2, 3);  // lowest non-trivial mode
  test_b2_disk_mode(2, 0, 3, 5);  // m=0 second radial level
  test_b2_disk_mode(2, 2, 3, 5);  // m=2
  test_b2_disk_mode(3, 1, 3, 5);  // n=3, m=1
  test_b2_disk_mode(3, 3, 4, 7);  // m=3
  test_b2_disk_mode(4, 2, 4, 7);  // n=4, m=2
  // 3D cylinder: (n_zernike, m_zernike, k_legendre, n_r, n_phi, n_z)
  test_b2_cylinder_mode(1, 1, 0, 3, 5, 4);
  test_b2_cylinder_mode(2, 0, 1, 3, 5, 4);
  test_b2_cylinder_mode(2, 2, 2, 3, 5, 4);
  test_b2_cylinder_mode(3, 1, 0, 4, 7, 3);
  test_b2_cylinder_mode(3, 3, 2, 4, 7, 4);
  test_b2_cylinder_mode(4, 2, 1, 4, 7, 5);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ZernikeB2",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_b2_disk_power_monitors();
  test_b2_cylinder_power_monitors();
  test_b2_power_monitor_exact_values();
}

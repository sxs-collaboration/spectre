// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/CartoonGhost.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/ParityFromSymmetry.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// The test fills every tensor component with a function of known parity:
//   Even parity -> f(x) = x^2
//   Odd  parity -> g(x) = x
//
// Ghost layer ig should mirror interior layer ix = ghost_zone_size - 1 - ig
// with the correct parity sign.
constexpr double h = 1.0;
constexpr size_t num_x = 4;
constexpr size_t num_y = 3;
constexpr size_t num_z = 3;

size_t interior_idx(const size_t ix, const size_t iy, const size_t iz) {
  return ix + num_x * (iy + num_y * iz);
}

size_t ghost_idx(const size_t i_ghost, const size_t iy, const size_t iz,
                 const size_t ghost_zone_size) {
  return i_ghost + ghost_zone_size * (iy + num_y * iz);
}

double x_coord(const size_t ix) { return (static_cast<double>(ix) + 0.5) * h; }

DataVector fill(const double power) {
  DataVector result(num_x * num_y * num_z);
  for (size_t iz = 0; iz < num_z; ++iz) {
    for (size_t iy = 0; iy < num_y; ++iy) {
      for (size_t ix = 0; ix < num_x; ++ix) {
        result[interior_idx(ix, iy, iz)] = std::pow(x_coord(ix), power);
      }
    }
  }
  return result;
}

// Returns a tensor with each component filled according to its parity:
// even -> x^2, odd -> x.
template <typename TensorType>
TensorType fill_by_parity() {
  TensorType tensor(num_x * num_y * num_z);
  constexpr auto parities = Spectral::make_component_parity_array<TensorType>();
  for (size_t comp = 0; comp < TensorType::size(); ++comp) {
    tensor[comp] = (gsl::at(parities, comp) == Spectral::Parity::Even)
                       ? fill(2.0)
                       : fill(1.0);
  }
  return tensor;
}

// Checks that every ghost layer ig mirrors interior layer
// ix = ghost_zone_size - 1 - ig with the correct parity sign.
template <typename TensorType>
void check_all_ghost_layers(const TensorType& interior, const TensorType& ghost,
                            const size_t ghost_zone_size,
                            const std::string& name) {
  CAPTURE(name);
  constexpr auto parities = Spectral::make_component_parity_array<TensorType>();
  for (size_t ig = 0; ig < ghost_zone_size; ++ig) {
    const size_t mirror_ix = ghost_zone_size - 1 - ig;
    for (size_t iy = 0; iy < num_y; ++iy) {
      for (size_t iz = 0; iz < num_z; ++iz) {
        const size_t gidx = ghost_idx(ig, iy, iz, ghost_zone_size);
        const size_t iidx = interior_idx(mirror_ix, iy, iz);
        for (size_t comp = 0; comp < TensorType::size(); ++comp) {
          const double sign =
              (gsl::at(parities, comp) == Spectral::Parity::Even) ? 1.0 : -1.0;
          CHECK(ghost[comp][gidx] == approx(sign * interior[comp][iidx]));
        }
      }
    }
  }
}

// Verifies that ghost_inv_metric is the inverse of ghost_metric and that
// ghost_sqrt_det equals sqrt(det(ghost_metric)) at every ghost point.
void check_inv_metric_and_sqrt_det(const tnsr::ii<DataVector, 3>& metric,
                                   const tnsr::II<DataVector, 3>& inv_metric,
                                   const Scalar<DataVector>& sqrt_det) {
  const size_t npts = get<0, 0>(metric).size();
  for (size_t gi = 0; gi < npts; ++gi) {
    const double g00 = get<0, 0>(metric)[gi];
    const double g01 = get<0, 1>(metric)[gi];
    const double g02 = get<0, 2>(metric)[gi];
    const double g11 = get<1, 1>(metric)[gi];
    const double g12 = get<1, 2>(metric)[gi];
    const double g22 = get<2, 2>(metric)[gi];
    const double det = g00 * (g11 * g22 - g12 * g12) -
                       g01 * (g01 * g22 - g12 * g02) +
                       g02 * (g01 * g12 - g11 * g02);
    CHECK(get(sqrt_det)[gi] == approx(std::sqrt(det)));

    for (size_t i = 0; i < 3; ++i) {
      for (size_t k = 0; k < 3; ++k) {
        double product = 0.0;
        for (size_t j = 0; j < 3; ++j) {
          product += inv_metric.get(i, j)[gi] * metric.get(j, k)[gi];
        }
        CHECK(product == approx(i == k ? 1.0 : 0.0));
      }
    }
  }
}

// Test with  need_tags_for_fluxes=true fills fills ghost metric, lapse,
// shift, pressure, eps and then computes LorentzFactor and SpatialVelocity.
void test_fd_ghost_impl(const size_t ghost_zone_size,
                        const bool need_tags_for_fluxes) {
  CAPTURE(ghost_zone_size);
  CAPTURE(need_tags_for_fluxes);

  const Mesh<3> subcell_mesh{{{num_x, num_y, num_z}},
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};

  const size_t npts = num_x * num_y * num_z;
  const auto interior_rho = fill_by_parity<Scalar<DataVector>>();
  const auto interior_ye = fill_by_parity<Scalar<DataVector>>();
  const auto interior_temp = fill_by_parity<Scalar<DataVector>>();
  const auto interior_press = fill_by_parity<Scalar<DataVector>>();
  const auto interior_eps = fill_by_parity<Scalar<DataVector>>();
  const auto interior_div_phi = fill_by_parity<Scalar<DataVector>>();
  const Scalar<DataVector> interior_lorentz{DataVector(npts, 1.56)};
  const auto interior_vel = fill_by_parity<tnsr::I<DataVector, 3>>();
  const auto interior_B = fill_by_parity<tnsr::I<DataVector, 3>>();

  // Positive-definite metric with one non-trivial odd component (gamma_xy)
  // to exercise the parity-negation logic for off-diagonal x-components.
  // Diagonal entries are large constants; off-diagonal x entries are 0.1*x
  // (small enough that the matrix stays positive definite for all test cells).
  tnsr::ii<DataVector, 3> interior_metric(npts);
  get<0, 0>(interior_metric) = DataVector(npts, 3.13);  // even, positive
  get<1, 1>(interior_metric) = DataVector(npts, 2.22);  // even, positive
  get<2, 2>(interior_metric) = DataVector(npts, 1.31);  // even, positive
  get<0, 1>(interior_metric) = 0.1 * fill(1.0);         // odd: 0.1*x
  get<0, 2>(interior_metric) = DataVector(npts, 0.0);   // odd: zero
  get<1, 2>(interior_metric) = DataVector(npts, 0.0);   // even: zero

  const Scalar<DataVector> interior_lapse{DataVector(npts, 1.5)};  // even
  tnsr::I<DataVector, 3> interior_shift(npts);
  get<0>(interior_shift) = 0.2 * fill(1.0);        // odd: 0.2*x
  get<1>(interior_shift) = DataVector(npts, 0.0);  // even: zero
  get<2>(interior_shift) = DataVector(npts, 0.0);  // even: zero

  Scalar<DataVector> ghost_rho{};
  Scalar<DataVector> ghost_ye{};
  Scalar<DataVector> ghost_temp{};
  Scalar<DataVector> ghost_press{};
  Scalar<DataVector> ghost_eps{};
  tnsr::I<DataVector, 3> ghost_lf_vel{};
  tnsr::I<DataVector, 3> ghost_spatial_velocity{};
  Scalar<DataVector> ghost_lorentz_factor{};
  tnsr::I<DataVector, 3> ghost_B{};
  Scalar<DataVector> ghost_phi{};
  tnsr::ii<DataVector, 3> ghost_metric{};
  tnsr::II<DataVector, 3> ghost_inv_metric{};
  Scalar<DataVector> ghost_sqrt_det{};
  Scalar<DataVector> ghost_lapse{};
  tnsr::I<DataVector, 3> ghost_shift{};

  grmhd::ValenciaDivClean::BoundaryConditions::CartoonGhost::fd_ghost_impl(
      make_not_null(&ghost_rho), make_not_null(&ghost_ye),
      make_not_null(&ghost_temp), make_not_null(&ghost_press),
      make_not_null(&ghost_eps), make_not_null(&ghost_lf_vel),
      make_not_null(&ghost_spatial_velocity),
      make_not_null(&ghost_lorentz_factor), make_not_null(&ghost_B),
      make_not_null(&ghost_phi), make_not_null(&ghost_metric),
      make_not_null(&ghost_inv_metric), make_not_null(&ghost_sqrt_det),
      make_not_null(&ghost_lapse), make_not_null(&ghost_shift),

      Direction<3>::lower_xi(), subcell_mesh,

      interior_rho, interior_ye, interior_temp, interior_press, interior_eps,
      interior_lorentz, interior_div_phi, interior_vel, interior_B,
      interior_metric, interior_lapse, interior_shift,

      ghost_zone_size, need_tags_for_fluxes);

  // Hydro prims: every ghost layer mirrors the correct interior layer.
  check_all_ghost_layers(interior_rho, ghost_rho, ghost_zone_size, "rho");
  check_all_ghost_layers(interior_ye, ghost_ye, ghost_zone_size, "ye");
  check_all_ghost_layers(interior_temp, ghost_temp, ghost_zone_size, "temp");
  check_all_ghost_layers(interior_div_phi, ghost_phi, ghost_zone_size, "phi");
  tnsr::I<DataVector, 3> interior_lf_vel{};
  tenex::evaluate<ti::I>(make_not_null(&interior_lf_vel),
                         (interior_lorentz)() * (interior_vel)(ti::I));
  check_all_ghost_layers(interior_lf_vel, ghost_lf_vel, ghost_zone_size,
                         "lorentz_times_vel");
  check_all_ghost_layers(interior_B, ghost_B, ghost_zone_size, "B");

  // Metric variables — only populated when need_tags_for_fluxes=true.
  if (need_tags_for_fluxes) {
    check_all_ghost_layers(interior_press, ghost_press, ghost_zone_size,
                           "pressure");
    check_all_ghost_layers(interior_eps, ghost_eps, ghost_zone_size, "eps");
    // Spatial metric: diagonal even components unchanged, off-diagonal x
    // components (gamma_xy, gamma_xz) negated.
    check_all_ghost_layers(interior_metric, ghost_metric, ghost_zone_size,
                           "metric");
    // Lapse is even (scalar), so ghost = interior at mirror.
    check_all_ghost_layers(interior_lapse, ghost_lapse, ghost_zone_size,
                           "lapse");
    // Shift: x-component is odd (negated), y and z are even.
    check_all_ghost_layers(interior_shift, ghost_shift, ghost_zone_size,
                           "shift");

    // Verify ghost_inv_metric is the inverse of ghost_metric and ghost_sqrt_det
    // equals sqrt(det(ghost_metric)) at every ghost point.
    check_inv_metric_and_sqrt_det(ghost_metric, ghost_inv_metric,
                                  ghost_sqrt_det);

    // LorentzFactor and SpatialVelocity are recomputed from the ghost metric
    // and ghost Wv (not merely mirrored from the interior).  Verify the self-
    // consistent relationships that the code implements:
    //   W = sqrt(1 + gamma_ij * (Wv)^i * (Wv)^j)
    //   v^i = (Wv)^i / W
    const size_t ghost_npts = ghost_zone_size * num_y * num_z;
    for (size_t gi = 0; gi < ghost_npts; ++gi) {
      // gamma_ij * (Wv)^i * (Wv)^j — use upper-triangular storage of tnsr::ii
      double contraction = 0.0;
      for (size_t a = 0; a < 3; ++a) {
        contraction += ghost_metric.get(a, a)[gi] * ghost_lf_vel.get(a)[gi] *
                       ghost_lf_vel.get(a)[gi];
        for (size_t b = a + 1; b < 3; ++b) {
          contraction += 2.0 * ghost_metric.get(a, b)[gi] *
                         ghost_lf_vel.get(a)[gi] * ghost_lf_vel.get(b)[gi];
        }
      }
      CHECK(get(ghost_lorentz_factor)[gi] ==
            approx(std::sqrt(1.0 + contraction)));
      for (size_t a = 0; a < 3; ++a) {
        CHECK(ghost_spatial_velocity.get(a)[gi] ==
              approx(ghost_lf_vel.get(a)[gi] / get(ghost_lorentz_factor)[gi]));
      }
    }
  }
}

void test_dg_ghost_error() {
  const grmhd::ValenciaDivClean::BoundaryConditions::CartoonGhost bc{};
  Scalar<DataVector> tilde_d{};
  Scalar<DataVector> tilde_ye{};
  Scalar<DataVector> tilde_tau{};
  Scalar<DataVector> tilde_phi{};
  tnsr::i<DataVector, 3> tilde_s{};
  tnsr::i<DataVector, 3> svof{};
  const tnsr::i<DataVector, 3> normal_cov{};
  tnsr::I<DataVector, 3> tilde_b{};
  tnsr::I<DataVector, 3> tilde_d_flux{};
  tnsr::I<DataVector, 3> tilde_ye_flux{};
  tnsr::I<DataVector, 3> tilde_tau_flux{};
  tnsr::I<DataVector, 3> tilde_phi_flux{};
  tnsr::I<DataVector, 3> shift{};
  tnsr::I<DataVector, 3> vel{};
  const tnsr::I<DataVector, 3> normal_vec{};
  tnsr::Ij<DataVector, 3> tilde_s_flux{};
  tnsr::IJ<DataVector, 3> tilde_b_flux{};
  tnsr::II<DataVector, 3> inv_metric{};
  Scalar<DataVector> lapse{};
  Scalar<DataVector> rho{};
  Scalar<DataVector> ye{};
  Scalar<DataVector> temp{};
  const std::optional<tnsr::I<DataVector, 3>> face_mesh_vel{};
  CHECK_THROWS_WITH(
      bc.dg_ghost(make_not_null(&tilde_d), make_not_null(&tilde_ye),
                  make_not_null(&tilde_tau), make_not_null(&tilde_s),
                  make_not_null(&tilde_b), make_not_null(&tilde_phi),
                  make_not_null(&tilde_d_flux), make_not_null(&tilde_ye_flux),
                  make_not_null(&tilde_tau_flux), make_not_null(&tilde_s_flux),
                  make_not_null(&tilde_b_flux), make_not_null(&tilde_phi_flux),
                  make_not_null(&lapse), make_not_null(&shift),
                  make_not_null(&svof), make_not_null(&rho), make_not_null(&ye),
                  make_not_null(&temp), make_not_null(&vel),
                  make_not_null(&inv_metric), face_mesh_vel, normal_cov,
                  normal_vec),
      Catch::Matchers::ContainsSubstring("dg_ghost() should never be called"));
}

#ifdef SPECTRE_DEBUG
void test_fd_ghost_impl_wrong_direction() {
  Scalar<DataVector> out_rho{};
  Scalar<DataVector> out_ye{};
  Scalar<DataVector> out_temp{};
  Scalar<DataVector> out_press{};
  Scalar<DataVector> out_eps{};
  Scalar<DataVector> out_lf{};
  Scalar<DataVector> out_dcf{};
  Scalar<DataVector> out_sqrt_det{};
  Scalar<DataVector> out_lapse{};
  tnsr::I<DataVector, 3> out_lf_vel{};
  tnsr::I<DataVector, 3> out_vel{};
  tnsr::I<DataVector, 3> out_B{};
  tnsr::I<DataVector, 3> out_shift{};
  tnsr::ii<DataVector, 3> out_metric{};
  tnsr::II<DataVector, 3> out_inv_metric{};
  const Scalar<DataVector> empty_scalar{};
  const tnsr::I<DataVector, 3> empty_vec{};
  const tnsr::ii<DataVector, 3> empty_metric{};
  const Mesh<3> mesh{1, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  CHECK_THROWS_WITH(
      grmhd::ValenciaDivClean::BoundaryConditions::CartoonGhost::fd_ghost_impl(
          make_not_null(&out_rho), make_not_null(&out_ye),
          make_not_null(&out_temp), make_not_null(&out_press),
          make_not_null(&out_eps), make_not_null(&out_lf_vel),
          make_not_null(&out_vel), make_not_null(&out_lf),
          make_not_null(&out_B), make_not_null(&out_dcf),
          make_not_null(&out_metric), make_not_null(&out_inv_metric),
          make_not_null(&out_sqrt_det), make_not_null(&out_lapse),
          make_not_null(&out_shift), Direction<3>::lower_eta(), mesh,
          empty_scalar, empty_scalar, empty_scalar, empty_scalar, empty_scalar,
          empty_scalar, empty_scalar, empty_vec, empty_vec, empty_metric,
          empty_scalar, empty_vec, 2, false),
      Catch::Matchers::ContainsSubstring(
          "Cartoon BC can only be applied in the x-direction"));
}
#endif

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryConditions.CartoonGhost",
                  "[Unit][GrMhd]") {
  test_fd_ghost_impl(1, true);
  test_fd_ghost_impl(2, true);
  test_fd_ghost_impl(3, true);
  // Not calculating fluxes just sets a subset of the variables; we only need
  // to check once the subset is filled
  test_fd_ghost_impl(2, false);
  test_dg_ghost_error();
#ifdef SPECTRE_DEBUG
  test_fd_ghost_impl_wrong_direction();
#endif
}

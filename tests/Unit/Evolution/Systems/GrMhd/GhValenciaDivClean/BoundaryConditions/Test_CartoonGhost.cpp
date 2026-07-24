// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/CartoonGhost.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/MonotonisedCentral.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Wcns5z.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
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

template <typename TensorType>
void check_fd_parity(const TensorType& interior, const TensorType& ghost,
                     const size_t ghost_zone_size, const std::string& name,
                     const double coefficient = 1.0) {
  CAPTURE(name);
  constexpr auto parities = Spectral::make_component_parity_array<TensorType>();
  for (size_t comp = 0; comp < TensorType::size(); ++comp) {
    const double expected =
        (gsl::at(parities, comp) == Spectral::Parity::Even) ? 0.0 : coefficient;
    for (size_t iy = 0; iy < num_y; ++iy) {
      for (size_t iz = 0; iz < num_z; ++iz) {
        const size_t iidx = interior_idx(0, iy, iz);
        // Use the nearest ghost layer, which mirrors interior[ix=0].
        const size_t gidx =
            ghost_idx(ghost_zone_size - 1, iy, iz, ghost_zone_size);
        CHECK((interior[comp][iidx] - ghost[comp][gidx]) / h ==
              approx(expected));
      }
    }
  }
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

void test_gh_fd_derivative_consistency(const size_t ghost_zone_size) {
  ASSERT(ghost_zone_size == 2 or ghost_zone_size == 3,
         "ghost_zone_size must be 2 or 3, got " << ghost_zone_size);
  CAPTURE(ghost_zone_size);
  using System = grmhd::GhValenciaDivClean::System<
      RadiationTransport::NoNeutrinos::System>;

  const Mesh<3> subcell_mesh{{{num_x, num_y, num_z}},
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};

  // Reconstructors are only used for their ghost zone size
  const grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<System>
      reconstructor_2{};
  const grmhd::GhValenciaDivClean::fd::Wcns5zPrim<System> reconstructor_3{};
  const grmhd::GhValenciaDivClean::fd::Reconstructor<System>& reconstructor =
      ghost_zone_size == 2
          ? static_cast<
                const grmhd::GhValenciaDivClean::fd::Reconstructor<System>&>(
                reconstructor_2)
          : static_cast<
                const grmhd::GhValenciaDivClean::fd::Reconstructor<System>&>(
                reconstructor_3);

  // GH evolved variables: all components filled according to their parity.
  const auto interior_spacetime_metric =
      fill_by_parity<tnsr::aa<DataVector, 3>>();
  const auto interior_pi = fill_by_parity<tnsr::aa<DataVector, 3>>();
  const auto interior_phi = fill_by_parity<tnsr::iaa<DataVector, 3>>();

  // Hydro primitive variables filled by parity.
  const auto interior_rho = fill_by_parity<Scalar<DataVector>>();
  const auto interior_ye = fill_by_parity<Scalar<DataVector>>();
  const auto interior_temp = fill_by_parity<Scalar<DataVector>>();
  const auto interior_press = fill_by_parity<Scalar<DataVector>>();
  const auto interior_eps = fill_by_parity<Scalar<DataVector>>();
  const auto interior_div_phi = fill_by_parity<Scalar<DataVector>>();
  const Scalar<DataVector> interior_lorentz{
      DataVector(num_x * num_y * num_z, 1.23)};
  const auto interior_vel = fill_by_parity<tnsr::I<DataVector, 3>>();
  const auto interior_B = fill_by_parity<tnsr::I<DataVector, 3>>();

  tnsr::aa<DataVector, 3> ghost_spacetime_metric{};
  tnsr::aa<DataVector, 3> ghost_pi{};
  tnsr::iaa<DataVector, 3> ghost_phi{};
  Scalar<DataVector> ghost_rho{};
  Scalar<DataVector> ghost_ye{};
  Scalar<DataVector> ghost_temp{};
  tnsr::I<DataVector, 3> ghost_lf_vel{};
  tnsr::I<DataVector, 3> ghost_B{};
  Scalar<DataVector> ghost_div_phi{};

  grmhd::GhValenciaDivClean::BoundaryConditions::CartoonGhost<System>::fd_ghost(
      make_not_null(&ghost_spacetime_metric), make_not_null(&ghost_pi),
      make_not_null(&ghost_phi), make_not_null(&ghost_rho),
      make_not_null(&ghost_ye), make_not_null(&ghost_temp),
      make_not_null(&ghost_lf_vel), make_not_null(&ghost_B),
      make_not_null(&ghost_div_phi),

      Direction<3>::lower_xi(), interior_spacetime_metric, interior_pi,
      interior_phi, subcell_mesh,

      interior_rho, interior_ye, interior_temp, interior_press, interior_eps,
      interior_lorentz, interior_div_phi, interior_vel, interior_B,

      reconstructor);

  // Verify FD derivatives at the ghost-interior interface using the nearest
  // ghost layer.
  check_fd_parity(interior_spacetime_metric, ghost_spacetime_metric,
                  ghost_zone_size, "spacetime_metric");
  check_fd_parity(interior_pi, ghost_pi, ghost_zone_size, "pi");
  check_fd_parity(interior_phi, ghost_phi, ghost_zone_size, "phi");
  check_fd_parity(interior_rho, ghost_rho, ghost_zone_size, "rho");
  check_fd_parity(interior_ye, ghost_ye, ghost_zone_size, "ye");
  check_fd_parity(interior_temp, ghost_temp, ghost_zone_size, "temp");
  check_fd_parity(interior_div_phi, ghost_div_phi, ghost_zone_size,
                  "deriv_phi");
  tnsr::I<DataVector, 3> interior_lf_vel{};
  tenex::evaluate<ti::I>(make_not_null(&interior_lf_vel),
                         (interior_lorentz)() * (interior_vel)(ti::I));
  check_fd_parity(interior_lf_vel, ghost_lf_vel, ghost_zone_size,
                  "lorentz_times_vel", get(interior_lorentz)[0]);
  check_fd_parity(interior_B, ghost_B, ghost_zone_size, "B");

  // Verify that every ghost layer mirrors the correct interior layer.
  // Ghost layer ig should equal parity_sign * interior[ix =
  // ghost_zone_size-1-ig].
  check_all_ghost_layers(interior_spacetime_metric, ghost_spacetime_metric,
                         ghost_zone_size, "spacetime_metrix");
  check_all_ghost_layers(interior_pi, ghost_pi, ghost_zone_size, "pi");
  check_all_ghost_layers(interior_phi, ghost_phi, ghost_zone_size, "phi");
  check_all_ghost_layers(interior_rho, ghost_rho, ghost_zone_size, "rho");
  check_all_ghost_layers(interior_ye, ghost_ye, ghost_zone_size, "ye");
  check_all_ghost_layers(interior_temp, ghost_temp, ghost_zone_size, "temp");
  check_all_ghost_layers(interior_div_phi, ghost_div_phi, ghost_zone_size,
                         "div_phi");
  check_all_ghost_layers(interior_lf_vel, ghost_lf_vel, ghost_zone_size,
                         "lorentz_times_val");
  check_all_ghost_layers(interior_B, ghost_B, ghost_zone_size, "B");
}

void test_gh_dg_ghost_error() {
  using System = grmhd::GhValenciaDivClean::System<
      RadiationTransport::NoNeutrinos::System>;
  const grmhd::GhValenciaDivClean::BoundaryConditions::CartoonGhost<System>
      bc{};
  tnsr::aa<DataVector, 3> psi{};
  tnsr::aa<DataVector, 3> pi{};
  tnsr::iaa<DataVector, 3> phi{};
  Scalar<DataVector> tilde_d{};
  Scalar<DataVector> tilde_ye{};
  Scalar<DataVector> tilde_tau{};
  Scalar<DataVector> tilde_phi{};
  Scalar<DataVector> gamma1{};
  Scalar<DataVector> gamma2{};
  Scalar<DataVector> lapse{};
  Scalar<DataVector> rho{};
  Scalar<DataVector> ye{};
  Scalar<DataVector> temp{};
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
  const std::optional<tnsr::I<DataVector, 3>> face_mesh_vel{};
  CHECK_THROWS_WITH(
      bc.dg_ghost(
          make_not_null(&psi), make_not_null(&pi), make_not_null(&phi),
          make_not_null(&tilde_d), make_not_null(&tilde_ye),
          make_not_null(&tilde_tau), make_not_null(&tilde_s),
          make_not_null(&tilde_b), make_not_null(&tilde_phi),
          make_not_null(&tilde_d_flux), make_not_null(&tilde_ye_flux),
          make_not_null(&tilde_tau_flux), make_not_null(&tilde_s_flux),
          make_not_null(&tilde_b_flux), make_not_null(&tilde_phi_flux),
          make_not_null(&gamma1), make_not_null(&gamma2), make_not_null(&lapse),
          make_not_null(&shift), make_not_null(&svof), make_not_null(&rho),
          make_not_null(&ye), make_not_null(&temp), make_not_null(&vel),
          make_not_null(&inv_metric), face_mesh_vel, normal_cov, normal_vec),
      Catch::Matchers::ContainsSubstring("dg_ghost() should never be called"));
}

#ifdef SPECTRE_DEBUG
void test_gh_fd_ghost_wrong_direction() {
  using System = grmhd::GhValenciaDivClean::System<
      RadiationTransport::NoNeutrinos::System>;
  tnsr::aa<DataVector, 3> out_psi{};
  tnsr::aa<DataVector, 3> out_pi{};
  tnsr::iaa<DataVector, 3> out_phi{};
  Scalar<DataVector> out_rho{};
  Scalar<DataVector> out_ye{};
  Scalar<DataVector> out_temp{};
  Scalar<DataVector> out_dcf{};
  tnsr::I<DataVector, 3> out_lf_vel{};
  tnsr::I<DataVector, 3> out_B{};
  const tnsr::aa<DataVector, 3> empty_aa{};
  const tnsr::iaa<DataVector, 3> empty_iaa{};
  const Scalar<DataVector> empty_scalar{};
  const tnsr::I<DataVector, 3> empty_vec{};
  const Mesh<3> mesh{1, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  const grmhd::GhValenciaDivClean::fd::MonotonisedCentralPrim<System>
      reconstructor{};
  CHECK_THROWS_WITH(
      (grmhd::GhValenciaDivClean::BoundaryConditions::CartoonGhost<
          System>::fd_ghost(make_not_null(&out_psi), make_not_null(&out_pi),
                            make_not_null(&out_phi), make_not_null(&out_rho),
                            make_not_null(&out_ye), make_not_null(&out_temp),
                            make_not_null(&out_lf_vel), make_not_null(&out_B),
                            make_not_null(&out_dcf), Direction<3>::lower_eta(),
                            empty_aa, empty_aa, empty_iaa, mesh, empty_scalar,
                            empty_scalar, empty_scalar, empty_scalar,
                            empty_scalar, empty_scalar, empty_scalar, empty_vec,
                            empty_vec, reconstructor)),
      Catch::Matchers::ContainsSubstring(
          "Cartoon BC can only be applied in the x-direction"));
}
#endif

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GrMhd.GhValenciaDivClean.BoundaryConditions.CartoonGhost",
    "[Unit][GrMhd]") {
  test_gh_fd_derivative_consistency(2);
  test_gh_fd_derivative_consistency(3);
  test_gh_dg_ghost_error();
#ifdef SPECTRE_DEBUG
  test_gh_fd_ghost_wrong_direction();
#endif
}

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarWave/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/ScalarWave/SphericalShellPowerMonitor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace {

Mesh<3> shell_mesh(const size_t radial_extents, const size_t ell_max) {
  return Mesh<3>{
      {{radial_extents, ell_max + 1, 2 * ell_max + 1}},
      {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}}};
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

void set_constant_angular_radial_profile(
    const gsl::not_null<DataVector*> component, const Mesh<3>& mesh,
    const DataVector& radial_profile) {
  const auto extents = mesh.extents();
  for (size_t r = 0; r < extents[0]; ++r) {
    for (size_t theta = 0; theta < extents[1]; ++theta) {
      for (size_t phi = 0; phi < extents[2]; ++phi) {
        Index<3> index{};
        index[0] = r;
        index[1] = theta;
        index[2] = phi;
        (*component)[collapsed_index(index, extents)] = radial_profile[r];
      }
    }
  }
}

// The radial monitor for a single tensor component set to a radial profile
// (constant in angle) is |a_r| / sqrt(num_tensor_components), where a_r are
// the radial modal coefficients.
DataVector expected_radial_monitor(const Mesh<3>& mesh,
                                   const DataVector& radial_profile,
                                   const size_t num_tensor_components) {
  ModalVector radial_modes(radial_profile.size(), 0.0);
  to_modal_coefficients(make_not_null(&radial_modes), radial_profile,
                        mesh.slice_through(0));
  DataVector expected(radial_profile.size(), 0.0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = std::abs(radial_modes[i]) /
                  sqrt(static_cast<double>(num_tensor_components));
  }
  return expected;
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarWave.SphericalShellPowerMonitor",
    "[Unit][Evolution]") {
  const Mesh<3> mesh = shell_mesh(4, 3);
  Variables<ylm::TensorYlm::filter_detail::sw_vars_list<Frame::Inertial>>
      sw_vars(mesh.number_of_grid_points(), 0.0);
  const DataVector radial_profile{1.0, -0.5, 0.25, 2.0};
  ScalarWave::power_monitor::SwCartToSphereMatrix cart_to_sphere{};

  // Psi: 1 component, spin weight 0.  A constant-angular profile excites
  // only the l=0 SH mode; the radial monitor should match the modal
  // coefficients of the radial profile divided by sqrt(1).
  set_constant_angular_radial_profile(
      make_not_null(&get(get<ScalarWave::Tags::Psi>(sw_vars))), mesh,
      radial_profile);
  {
    const auto monitors = ScalarWave::power_monitor::sw_shell_power_monitors(
        make_not_null(&cart_to_sphere), sw_vars, mesh,
        identity_jacobian(mesh.number_of_grid_points()));
    CHECK_ITERABLE_APPROX(monitors.psi[0],
                          expected_radial_monitor(mesh, radial_profile, 1));
    CHECK(max(monitors.psi[1]) > 0.0);
    CHECK(max(abs(monitors.pi[0])) == 0.0);
    CHECK(max(abs(monitors.pi[1])) == 0.0);
    CHECK(max(abs(monitors.phi[0])) == 0.0);
    CHECK(max(abs(monitors.phi[1])) == 0.0);
  }

  // Pi: 1 component, spin weight 0.  Same structure as Psi.
  std::fill(sw_vars.data(), sw_vars.data() + sw_vars.size(), 0.0);
  set_constant_angular_radial_profile(
      make_not_null(&get(get<ScalarWave::Tags::Pi>(sw_vars))), mesh,
      radial_profile);
  {
    const auto monitors = ScalarWave::power_monitor::sw_shell_power_monitors(
        make_not_null(&cart_to_sphere), sw_vars, mesh,
        identity_jacobian(mesh.number_of_grid_points()));
    CHECK_ITERABLE_APPROX(monitors.pi[0],
                          expected_radial_monitor(mesh, radial_profile, 1));
    CHECK(max(monitors.pi[1]) > 0.0);
    CHECK(max(abs(monitors.psi[0])) == 0.0);
    CHECK(max(abs(monitors.psi[1])) == 0.0);
    CHECK(max(abs(monitors.phi[0])) == 0.0);
    CHECK(max(abs(monitors.phi[1])) == 0.0);
  }

  // Phi: 3 components (tnsr::i<DataVector,3>), spin weights 0 and +/-1.
  // Set Phi_x only; the cart-to-sphere transform distributes power across
  // the spin-weighted components, so only the radial monitor (which is
  // computed from the raw un-transformed data) has the simple form
  // |a_r| / sqrt(3).  Psi and Pi must remain zero.
  std::fill(sw_vars.data(), sw_vars.data() + sw_vars.size(), 0.0);
  set_constant_angular_radial_profile(
      make_not_null(&get<0>(get<ScalarWave::Tags::Phi<3>>(sw_vars))), mesh,
      radial_profile);
  {
    const auto monitors = ScalarWave::power_monitor::sw_shell_power_monitors(
        make_not_null(&cart_to_sphere), sw_vars, mesh,
        identity_jacobian(mesh.number_of_grid_points()));
    CHECK_ITERABLE_APPROX(monitors.phi[0],
                          expected_radial_monitor(mesh, radial_profile, 3));
    CHECK(max(monitors.phi[1]) > 0.0);
    CHECK(max(abs(monitors.psi[0])) == 0.0);
    CHECK(max(abs(monitors.psi[1])) == 0.0);
    CHECK(max(abs(monitors.pi[0])) == 0.0);
    CHECK(max(abs(monitors.pi[1])) == 0.0);
  }
}

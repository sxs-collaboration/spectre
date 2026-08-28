// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarWave/ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave::power_monitor {

/// Holds separate SW shell power monitors for each evolved SW variable.
/// Index 0 is the radial monitor and index 1 is the angular monitor.
struct SwShellPowerMonitors {
  std::array<DataVector, 2> psi{};
  std::array<DataVector, 2> pi{};
  std::array<DataVector, 2> phi{};
};

/// Sparse matrix for Cartesian-component to TensorYlm-basis transform.
/// Only the rank-1 (`i`) matrix is needed for the ScalarWave system, since
/// \f$\Phi_i\f$ is the only non-scalar evolved variable.
struct SwCartToSphereMatrix {
  std::optional<SimpleSparseMatrix> i{};
};

/// Fill the Cartesian-to-TensorYlm sparse matrix with Spherepack normalization
/// if not already filled.
void fill_sw_cart_to_sphere_matrix(gsl::not_null<SwCartToSphereMatrix*> matrix,
                                   size_t ell_max);

/*!
 * \brief Compute radial and angular power monitors for the evolved ScalarWave
 * variables on a spherical shell.
 *
 * Each monitor is an array containing the radial monitor at index 0 and the
 * angular monitor at index 1. Radial monitors are computed from the original
 * SW variables, matching SpEC's shell radial power monitor behavior. Angular
 * monitors are computed after transforming \f$\Phi_i\f$ to TensorYlm
 * coefficients using `jac_inertial_to_grid` for the frame transform.
 * \f$\Psi\f$ and \f$\Pi\f$ are scalars and need no frame transform.
 *
 * The Cartesian-to-TensorYlm matrix is filled on first use and reused on
 * subsequent calls.
 *
 * \param cart_to_sphere_matrix Cache for the Cartesian-to-TensorYlm sparse
 *   matrix.
 * \param sw_vars ScalarWave variables at collocation points on the shell mesh.
 * \param mesh The spherical-shell mesh with dimensions `(radial, theta, phi)`.
 * \param jac_inertial_to_grid Inverse Jacobian mapping inertial-frame spatial
 *   tensor components to the grid frame.
 */
SwShellPowerMonitors sw_shell_power_monitors(
    gsl::not_null<SwCartToSphereMatrix*> cart_to_sphere_matrix,
    const Variables<
        ylm::TensorYlm::filter_detail::sw_vars_list<Frame::Inertial>>& sw_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid);

}  // namespace ScalarWave::power_monitor

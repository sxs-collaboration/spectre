// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarWave/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/ScalarWave/SphericalShellPowerMonitor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
/// \endcond

namespace ScalarWave::power_monitor {

/// Holds separate SW filled-sphere power monitors for each evolved SW variable.
/// Index 0 is the radial monitor and index 1 is the angular monitor.
struct SwFilledSpherePowerMonitors {
  std::array<DataVector, 2> psi{};
  std::array<DataVector, 2> pi{};
  std::array<DataVector, 2> phi{};
};

/*!
 * \brief Compute radial and angular power monitors for the evolved ScalarWave
 * variables on a filled-sphere (ZernikeB3) element.
 *
 * Each monitor is an array with the radial monitor at index 0 and the angular
 * monitor at index 1. Both monitors are computed after transforming the SW
 * variables to TensorYlm coefficients in the grid frame using
 * `jac_inertial_to_grid`. \f$\Psi\f$ and \f$\Pi\f$ are scalars and need no
 * frame transform; \f$\Phi_i\f$ is transformed to the grid frame before the
 * SH analysis.
 *
 * **Radial monitor** (index 0): groups squared Jacobi spectral coefficients by
 * the radial mode index \f$\text{mode} = (n+1)/2\f$ where
 * \f$n = \ell + 2k\f$ is the total ZernikeB3 degree, \f$\ell\f$ is the
 * angular degree, and \f$k\f$ is the Jacobi radial index. The result has
 * \f$n_r\f$ entries.
 *
 * **Angular monitor** (index 1): groups squared Jacobi spectral coefficients
 * by angular degree \f$\ell\f$. Spin-weighted spherical harmonic modes with
 * \f$\ell < |s|\f$ (where \f$s\f$ is the spin weight of the TensorYlm
 * component) contribute zero to both the sum and the count. The result has
 * \f$\ell_\max + 1\f$ entries.
 *
 * The Cartesian-to-TensorYlm matrix in `cart_to_sphere_matrix` is filled on
 * first use and reused on subsequent calls.
 *
 * \param cart_to_sphere_matrix Cache for the Cartesian-to-TensorYlm sparse
 *   matrix.
 * \param sw_vars ScalarWave variables at collocation points on the B3 ball
 *   mesh.
 * \param mesh The ZernikeB3 mesh.
 * \param jac_inertial_to_grid Inverse Jacobian mapping inertial-frame spatial
 *   tensor components to the grid frame.
 */
SwFilledSpherePowerMonitors sw_filled_sphere_power_monitors(
    gsl::not_null<SwCartToSphereMatrix*> cart_to_sphere_matrix,
    const Variables<
        ylm::TensorYlm::filter_detail::sw_vars_list<Frame::Inertial>>& sw_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid);

}  // namespace ScalarWave::power_monitor

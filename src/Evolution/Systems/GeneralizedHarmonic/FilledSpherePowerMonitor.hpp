// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/SphericalShellPowerMonitor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
/// \endcond

namespace gh::power_monitor {

/// Holds separate GH filled-sphere power monitors for each evolved GH
/// variable. Index 0 is the radial monitor and index 1 is the angular monitor.
struct GhFilledSpherePowerMonitors {
  std::array<DataVector, 2> spacetime_metric{};
  std::array<DataVector, 2> pi{};
  std::array<DataVector, 2> phi{};
};

/*!
 * \brief Compute radial and angular power monitors for the evolved GH
 * variables on a filled-sphere element.
 *
 * Each monitor is an array with the radial monitor at index 0 and the angular
 * monitor at index 1. Both monitors are computed after transforming the GH
 * variables to TensorYlm coefficients in the grid frame using
 * `jac_inertial_to_grid`.
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
 * component) contribute zero to both the sum and the count, following the same
 * convention as `gh_shell_power_monitors`. The result has \f$\ell_\max + 1\f$
 * entries.
 *
 * The Cartesian-to-TensorYlm matrices in `cart_to_sphere_matrices` are filled
 * on first use and reused on subsequent calls.
 *
 * \param cart_to_sphere_matrices Cache for Cartesian-to-TensorYlm sparse
 *   matrices.
 * \param gh_vars Generalized Harmonic variables at collocation points on the
 *   B3 ball mesh.
 * \param mesh The ZernikeB3 mesh.
 * \param jac_inertial_to_grid Inverse Jacobian mapping inertial-frame spatial
 *   tensor components to the grid frame.
 */
GhFilledSpherePowerMonitors gh_filled_sphere_power_monitors(
    gsl::not_null<CartToSphereMatrices*> cart_to_sphere_matrices,
    const Variables<ylm::TensorYlm::filter_detail::gh_spacetime_vars_list>&
        gh_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid);

}  // namespace gh::power_monitor

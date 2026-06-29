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
#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
namespace ylm {
class Spherepack;
}  // namespace ylm
/// \endcond

namespace gh::power_monitor {

/// Holds separate GH shell power monitors for each evolved GH variable.
struct GhShellPowerMonitors {
  std::array<DataVector, 2> spacetime_metric{};
  std::array<DataVector, 2> pi{};
  std::array<DataVector, 2> phi{};
};

/// Sparse matrices for Cartesian-component to TensorYlm-basis transforms.
struct CartToSphereMatrices {
  std::optional<SimpleSparseMatrix> i{};
  std::optional<SimpleSparseMatrix> ii{};
  std::optional<SimpleSparseMatrix> ij{};
  std::optional<SimpleSparseMatrix> ijj{};
};

/// Fill any missing Cartesian-to-TensorYlm sparse matrices with Spherepack
/// normalization.
void fill_cart_to_sphere_matrices(gsl::not_null<CartToSphereMatrices*> matrices,
                                  size_t ell_max);

/*!
 * \brief Compute radial and angular power monitors for the evolved GH
 * variables on a spherical shell.
 *
 * Each monitor is an array containing the radial monitor at index 0 and the
 * angular monitor at index 1. Radial monitors are computed from the original
 * GH variables, matching SpEC's shell radial power monitor behavior. Angular
 * monitors are computed after transforming the GH variables to TensorYlm
 * coefficients. The Cartesian-to-TensorYlm matrices are filled on first use
 * and reused on subsequent calls.
 */
GhShellPowerMonitors gh_shell_power_monitors(
    gsl::not_null<CartToSphereMatrices*> cart_to_sphere_matrices,
    const Variables<ylm::TensorYlm::filter_detail::gh_spacetime_vars_list>&
        gh_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid);

}  // namespace gh::power_monitor

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <ostream>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Actions::detail {
/*
 * Computes the volume terms for a discontinuous Galerkin scheme.
 *
 * The function does the following (in order):
 *
 * 1. Compute the partial derivatives of the `System::gradient_variables`.
 *
 *    The partial derivatives are needed in the nonconservative product terms
 *    of the evolution equations. Any variable whose evolution equation does
 *    not contain a flux must contain a nonconservative product and the
 *    variable must be listed in the `System::gradient_variables` type alias.
 *    The partial derivatives are also needed for adding the moving mesh terms
 *    to the equations that do not have a flux term.
 *
 * 2. The volume time derivatives are calculated from
 *    `System::compute_volume_time_derivative_terms`
 *
 *    The source terms and nonconservative products are contributed directly
 *    to the `dt_vars` arguments passed to the time derivative function, while
 *    the volume fluxes are computed into the `volume_fluxes` arguments. The
 *    divergence of the volume fluxes will be computed and added to the time
 *    derivatives later in the function.
 *
 * 3. If the mesh is moving the appropriate mesh velocity terms are added to
 *    the equations.
 *
 *    For equations with fluxes this means that \f$-v^i_g u_\alpha\f$ is
 *    added to the fluxes and \f$-u_\alpha \partial_i v^i_g\f$ is added
 *    to the time derivatives. For equations without fluxes
 *    \f$v^i\partial_i u_\alpha\f$ is added to the time derivatives.
 *
 * 4. Compute flux divergence contribution and add it to the time derivatives.
 *
 *    Either the weak or strong form can be used. Currently only the strong
 *    form is coded, but adding the weak form is quite easy.
 *
 *    Note that the computation of the flux divergence and adding that to the
 *    time derivative must be done *after* the mesh velocity is subtracted
 *    from the fluxes.
 */
template <typename System, size_t Dim, typename... TimeDerivativeArguments,
          typename... VariablesTags, typename... PartialDerivTags,
          typename... FluxVariablesTags, typename... TemporaryTags>
void volume_terms(
    const gsl::not_null<Variables<tmpl::list<::Tags::dt<VariablesTags>...>>*>
        dt_vars_ptr,
    [[maybe_unused]] const gsl::not_null<
        Variables<tmpl::list<FluxVariablesTags...>>*>
        volume_fluxes,
    [[maybe_unused]] const gsl::not_null<
        Variables<tmpl::list<PartialDerivTags...>>*>
        partial_derivs,
    [[maybe_unused]] const gsl::not_null<
        Variables<tmpl::list<TemporaryTags...>>*>
        temporaries,
    const Variables<tmpl::list<VariablesTags...>>& evolved_vars,
    const ::dg::Formulation dg_formulation, const Mesh<Dim>& mesh,
    [[maybe_unused]] const tnsr::I<DataVector, Dim, Frame::Inertial>&
        inertial_coordinates,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        logical_to_inertial_inverse_jacobian,
    [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity,
    const std::optional<Scalar<DataVector>>& div_mesh_velocity,
    const TimeDerivativeArguments&... time_derivative_args) noexcept;
}  // namespace evolution::dg::Actions::detail

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <ostream>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MetricIdentityJacobian.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/LinearOperators/WeakDivergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
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
template <typename ComputeVolumeTimeDerivativeTerms, size_t Dim,
          typename... TimeDerivativeArguments, typename... VariablesTags,
          typename... PartialDerivTags, typename... FluxVariablesTags,
          typename... TemporaryTags>
void volume_terms(
    const gsl::not_null<Variables<tmpl::list<::Tags::dt<VariablesTags>...>>*>
        dt_vars_ptr,
    [[maybe_unused]] const gsl::not_null<Variables<tmpl::list<::Tags::Flux<
        FluxVariablesTags, tmpl::size_t<Dim>, Frame::Inertial>...>>*>
        volume_fluxes,
    [[maybe_unused]] const gsl::not_null<Variables<tmpl::list<::Tags::deriv<
        PartialDerivTags, tmpl::size_t<Dim>, Frame::Inertial>...>>*>
        partial_derivs,
    [[maybe_unused]] const gsl::not_null<
        Variables<tmpl::list<TemporaryTags...>>*>
        temporaries,
    const Variables<tmpl::list<VariablesTags...>>& evolved_vars,
    const ::dg::Formulation dg_formulation, const Mesh<Dim>& mesh,
    [[maybe_unused]] const tnsr::I<DataVector, Dim, Frame::Inertial>&
        inertial_coordinates,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>&
        logical_to_inertial_inverse_jacobian,
    [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity,
    const std::optional<Scalar<DataVector>>& div_mesh_velocity,
    const TimeDerivativeArguments&... time_derivative_args) noexcept {
  static constexpr bool has_partial_derivs = sizeof...(PartialDerivTags) != 0;
  static constexpr bool has_fluxes = sizeof...(FluxVariablesTags) != 0;
  static_assert(
      has_fluxes or has_partial_derivs,
      "Must have either fluxes or partial derivatives in a "
      "DG evolution scheme. This means the evolution system struct (usually in "
      "Evolution/Systems/YourSystem/System.hpp) being used does not specify "
      "any flux_variables or gradient_variables. Make sure the type aliases "
      "are defined, and that at least one of them is a non-empty list of "
      "tags.");

  using partial_derivative_tags = tmpl::list<PartialDerivTags...>;
  using flux_variables =
      tmpl::list<FluxVariablesTags...>;

  // Compute d_i u_\alpha for nonconservative products
  if constexpr (has_partial_derivs) {
    partial_derivatives<partial_derivative_tags>(
        partial_derivs, evolved_vars, mesh,
        logical_to_inertial_inverse_jacobian);
  }

  // For now just zero dt_vars. If this is a performance bottle neck we
  // can re-evaluate in the future.
  dt_vars_ptr->initialize(mesh.number_of_grid_points(), 0.0);

  // Compute volume du/dt and fluxes
  ComputeVolumeTimeDerivativeTerms::apply(
      make_not_null(&get<::Tags::dt<VariablesTags>>(*dt_vars_ptr))...,
      make_not_null(&get<::Tags::Flux<FluxVariablesTags, tmpl::size_t<Dim>,
                                      Frame::Inertial>>(*volume_fluxes))...,
      make_not_null(&get<TemporaryTags>(*temporaries))...,
      get<::Tags::deriv<PartialDerivTags, tmpl::size_t<Dim>, Frame::Inertial>>(
          *partial_derivs)...,
      time_derivative_args...);

  // Add volume terms for moving meshes
  if (mesh_velocity.has_value()) {
    tmpl::for_each<flux_variables>([&div_mesh_velocity, &dt_vars_ptr,
                                    &evolved_vars, &mesh_velocity,
                                    &volume_fluxes](auto tag_v) noexcept {
      // Modify fluxes for moving mesh
      using var_tag = typename decltype(tag_v)::type;
      using flux_var_tag =
          db::add_tag_prefix<::Tags::Flux, var_tag, tmpl::size_t<Dim>,
                             Frame::Inertial>;
      auto& flux_var = get<flux_var_tag>(*volume_fluxes);
      // Loop over all independent components of flux_var
      for (size_t flux_var_storage_index = 0;
           flux_var_storage_index < flux_var.size(); ++flux_var_storage_index) {
        // Get the flux variable's tensor index, e.g. (i,j) for a F^i of
        // the spatial velocity (or some other spatial tensor).
        const auto flux_var_tensor_index =
            flux_var.get_tensor_index(flux_var_storage_index);
        // Remove the first index from the flux tensor index, gets back
        // (j)
        const auto var_tensor_index =
            all_but_specified_element_of(flux_var_tensor_index, 0);
        // Set flux_index to (i)
        const size_t flux_index = gsl::at(flux_var_tensor_index, 0);

        // We now need to index flux(i,j) -= u(j) * v_g(i)
        flux_var[flux_var_storage_index] -=
            get<var_tag>(evolved_vars).get(var_tensor_index) *
            mesh_velocity->get(flux_index);
      }

      // Modify time derivative (i.e. source terms) for moving mesh
      auto& dt_var = get<::Tags::dt<var_tag>>(*dt_vars_ptr);
      for (size_t dt_var_storage_index = 0;
           dt_var_storage_index < dt_var.size(); ++dt_var_storage_index) {
        // This is S -> S - u d_i v^i_g
        dt_var[dt_var_storage_index] -=
            get<var_tag>(evolved_vars)[dt_var_storage_index] *
            get(*div_mesh_velocity);
      }
    });

    // We add the mesh velocity to all equations that don't have flux terms.
    // This doesn't need to be equal to the equations that have partial
    // derivatives. For example, the scalar field evolution equation in
    // first-order form does not have any partial derivatives but still needs
    // the velocity term added. This is because the velocity term arises from
    // transforming the time derivative.
    using non_flux_tags =
        tmpl::list_difference<tmpl::list<VariablesTags...>, flux_variables>;

    tmpl::for_each<non_flux_tags>([&dt_vars_ptr, &mesh_velocity,
                                   &partial_derivs](auto var_tag_v) noexcept {
      using var_tag = typename decltype(var_tag_v)::type;
      using dt_var_tag = ::Tags::dt<var_tag>;
      using deriv_var_tag =
          ::Tags::deriv<var_tag, tmpl::size_t<Dim>, Frame::Inertial>;

      const auto& deriv_var = get<deriv_var_tag>(*partial_derivs);
      auto& dt_var = get<dt_var_tag>(*dt_vars_ptr);

      // Loop over all independent components of the derivative of the
      // variable.
      for (size_t deriv_var_storage_index = 0;
           deriv_var_storage_index < deriv_var.size();
           ++deriv_var_storage_index) {
        // We grab the `deriv_tensor_index`, which would be e.g.
        // `(i, a, b)`, so `(0, 2, 3)`
        const auto deriv_var_tensor_index =
            deriv_var.get_tensor_index(deriv_var_storage_index);
        // Then we drop the derivative index (the first entry) to get
        // `(a, b)` (or `(2, 3)`)
        const auto dt_var_tensor_index =
            all_but_specified_element_of(deriv_var_tensor_index, 0);
        // Set `deriv_index` to `i` (or `0` in the example)
        const size_t deriv_index = gsl::at(deriv_var_tensor_index, 0);
        dt_var.get(dt_var_tensor_index) += mesh_velocity->get(deriv_index) *
                                           deriv_var[deriv_var_storage_index];
      }
    });
  }

  // Add the flux divergence term to du_\alpha/dt, which must be done
  // after the corrections for the moving mesh are made.
  if constexpr (has_fluxes) {
    Variables<tmpl::list<::Tags::div<::Tags::Flux<
        FluxVariablesTags, tmpl::size_t<Dim>, Frame::Inertial>>...>>
        div_fluxes{mesh.number_of_grid_points()};
    if (dg_formulation == ::dg::Formulation::StrongInertial) {
      divergence(make_not_null(&div_fluxes), *volume_fluxes, mesh,
                 logical_to_inertial_inverse_jacobian);
    } else if (dg_formulation == ::dg::Formulation::WeakInertial) {
      // We should ideally not recompute the
      // det_jac_times_inverse_jacobian for non-moving meshes.
      if constexpr (Dim == 1) {
        weak_divergence(make_not_null(&div_fluxes), *volume_fluxes, mesh, {});
      } else {
        // The Jacobian should be computed as a compute tag
        const auto jacobian =
            determinant_and_inverse(logical_to_inertial_inverse_jacobian)
                .second;
        InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
            det_jac_times_inverse_jacobian{};
        ::dg::metric_identity_det_jac_times_inv_jac(
            make_not_null(&det_jac_times_inverse_jacobian), mesh,
            inertial_coordinates, jacobian);
        weak_divergence(make_not_null(&div_fluxes), *volume_fluxes, mesh,
                        det_jac_times_inverse_jacobian);
      }
      ASSERT(det_inverse_jacobian != nullptr,
             "The determinant of the inverse Jacobian shouldn't be nullptr "
             "when using the weak form.");
      div_fluxes *= get(*det_inverse_jacobian);
    } else {
      ERROR("Unsupported DG formulation: " << dg_formulation);
    }
    tmpl::for_each<flux_variables>(
        [&dg_formulation, &dt_vars_ptr, &div_fluxes](auto var_tag_v) noexcept {
          using var_tag = typename decltype(var_tag_v)::type;
          auto& dt_var = get<::Tags::dt<var_tag>>(*dt_vars_ptr);
          const auto& div_flux = get<::Tags::div<
              ::Tags::Flux<var_tag, tmpl::size_t<Dim>, Frame::Inertial>>>(
              div_fluxes);
          if (dg_formulation == ::dg::Formulation::StrongInertial) {
            for (size_t storage_index = 0; storage_index < dt_var.size();
                 ++storage_index) {
              dt_var[storage_index] -= div_flux[storage_index];
            }
          } else {
            for (size_t storage_index = 0; storage_index < dt_var.size();
                 ++storage_index) {
              dt_var[storage_index] += div_flux[storage_index];
            }
          }
        });
  } else {
    (void)dg_formulation;
  }
}
}  // namespace evolution::dg::Actions::detail

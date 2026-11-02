// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/NewtonianMhd/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::Actions::detail {
#define VOLUME_TERMS_INSTANTIATION(DIM)                                        \
  template void volume_terms<::NewtonianMhd::TimeDerivativeTerms<DIM>>(      \
      gsl::not_null<Variables<db::wrap_tags_in<                                \
          ::Tags::dt,                                                          \
          typename ::NewtonianMhd::System<DIM>::variables_tag::tags_list>>*> \
          dt_vars_ptr,                                                         \
      gsl::not_null<Variables<db::wrap_tags_in<                                \
          ::Tags::Flux,                                                        \
          typename ::NewtonianMhd::System<DIM>::flux_variables,              \
          tmpl::size_t<DIM>, Frame::Inertial>>*>                               \
          volume_fluxes,                                                       \
      gsl::not_null<Variables<db::wrap_tags_in<                                \
          ::Tags::deriv,                                                       \
          typename ::NewtonianMhd::System<DIM>::gradient_variables,          \
          tmpl::size_t<DIM>, Frame::Inertial>>*>                               \
          partial_derivs,                                                      \
      gsl::not_null<Variables<typename ::NewtonianMhd::System<               \
          DIM>::compute_volume_time_derivative_terms::temporary_tags>*>        \
          temporaries,                                                         \
      gsl::not_null<Variables<db::wrap_tags_in<                                \
          ::Tags::div,                                                         \
          db::wrap_tags_in<                                                    \
              ::Tags::Flux,                                                    \
              typename ::NewtonianMhd::System<DIM>::flux_variables,          \
              tmpl::size_t<DIM>, Frame::Inertial>>>*>                          \
          div_fluxes,                                                          \
      const Variables<                                                         \
          typename ::NewtonianMhd::System<DIM>::variables_tag::tags_list>&   \
          evolved_vars,                                                        \
      const ::dg::Formulation dg_formulation, const Mesh<DIM>& mesh,           \
      [[maybe_unused]] const tnsr::I<DataVector, DIM, Frame::Inertial>&        \
          inertial_coordinates,                                                \
      const InverseJacobian<DataVector, DIM, Frame::ElementLogical,            \
                            Frame::Inertial>&                                  \
          logical_to_inertial_inverse_jacobian,                                \
      [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,   \
      const std::optional<tnsr::I<DataVector, DIM, Frame::Inertial>>&          \
          mesh_velocity,                                                       \
      const std::optional<Scalar<DataVector>>& div_mesh_velocity,              \
      const Scalar<DataVector>& mass_density_cons,                             \
      const tnsr::I<DataVector, DIM>& momentum_density,                        \
      const Scalar<DataVector>& energy_density,                                \
      const tnsr::I<DataVector, DIM>& velocity,                                \
      const Scalar<DataVector>& pressure,                                      \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const EquationsOfState::EquationOfState<false, 2>& eos,                  \
      const tnsr::I<DataVector, DIM>& coords, const double& time,              \
      const ::NewtonianMhd::Sources::Source<DIM>& source);
}  // namespace evolution::dg::Actions::detail

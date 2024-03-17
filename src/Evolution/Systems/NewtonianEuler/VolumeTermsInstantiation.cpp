// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <optional>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::Actions::detail {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                \
  template void                                                               \
  volume_terms<::NewtonianEuler::TimeDerivativeTerms<DIM(data)>>(             \
      gsl::not_null<Variables<db::wrap_tags_in<                               \
          ::Tags::dt, typename ::NewtonianEuler::System<DIM(                  \
                          data)>::variables_tag::tags_list>>*>                \
          dt_vars_ptr,                                                        \
      gsl::not_null<Variables<db::wrap_tags_in<                               \
          ::Tags::Flux,                                                       \
          typename ::NewtonianEuler::System<DIM(data)>::flux_variables,       \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          volume_fluxes,                                                      \
      gsl::not_null<Variables<db::wrap_tags_in<                               \
          ::Tags::deriv,                                                      \
          typename ::NewtonianEuler::System<DIM(data)>::gradient_variables,   \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          partial_derivs,                                                     \
      gsl::not_null<Variables<typename ::NewtonianEuler::System<DIM(          \
          data)>::compute_volume_time_derivative_terms::temporary_tags>*>     \
          temporaries,                                                        \
      gsl::not_null<Variables<db::wrap_tags_in<                               \
          ::Tags::div,                                                        \
          db::wrap_tags_in<                                                   \
              ::Tags::Flux,                                                   \
              typename ::NewtonianEuler::System<DIM(data)>::flux_variables,   \
              tmpl::size_t<DIM(data)>, Frame::Inertial>>>*>                   \
          div_fluxes,                                                         \
      const Variables<typename ::NewtonianEuler::System<DIM(                  \
          data)>::variables_tag::tags_list>& evolved_vars,                    \
      const ::dg::Formulation dg_formulation, const Mesh<DIM(data)>& mesh,    \
      [[maybe_unused]] const tnsr::I<DataVector, DIM(data), Frame::Inertial>& \
          inertial_coordinates,                                               \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,     \
                            Frame::Inertial>&                                 \
          logical_to_inertial_inverse_jacobian,                               \
      [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,  \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&   \
          mesh_velocity,                                                      \
      const std::optional<Scalar<DataVector>>& div_mesh_velocity,             \
      const Scalar<DataVector>& mass_density_cons,                            \
      const tnsr::I<DataVector, DIM(data)>& momentum_density,                 \
      const Scalar<DataVector>& energy_density,                               \
      const tnsr::I<DataVector, DIM(data)>& velocity,                         \
      const Scalar<DataVector>& pressure,                                     \
      const Scalar<DataVector>& specific_internal_energy,                     \
      const EquationsOfState::EquationOfState<false, 2>& eos,                 \
      const tnsr::I<DataVector, DIM(data)>& coords, const double& time,       \
      const ::NewtonianEuler::Sources::Source<DIM(data)>& source);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef SYSTEM
#undef DIM
}  // namespace evolution::dg::Actions::detail

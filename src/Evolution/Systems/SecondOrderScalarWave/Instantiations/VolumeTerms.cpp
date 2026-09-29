// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/TimeDerivative.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template void evolution::dg::Actions::detail::volume_terms<                  \
      ::SecondOrderScalarWave::TimeDerivative<DIM(data)>>(                     \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::dt,                                                          \
          typename ::SecondOrderScalarWave::System<DIM(data)>::volume_vars>>*> \
          dt_vars_ptr,                                                         \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::Flux,                                                        \
          typename ::SecondOrderScalarWave::System<DIM(data)>::flux_variables, \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                         \
          volume_fluxes,                                                       \
      const gsl::not_null<Variables<                                           \
          db::wrap_tags_in<::Tags::deriv,                                      \
                           typename ::SecondOrderScalarWave::System<DIM(       \
                               data)>::gradient_variables,                     \
                           tmpl::size_t<DIM(data)>, Frame::Inertial>>*>        \
          partial_derivs,                                                      \
      const gsl::not_null<Variables<typename ::SecondOrderScalarWave::System<  \
          DIM(data)>::compute_volume_time_derivative_terms::temporary_tags>*>  \
          temporaries,                                                         \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::div,                                                         \
          db::wrap_tags_in<::Tags::Flux,                                       \
                           typename ::SecondOrderScalarWave::System<DIM(       \
                               data)>::flux_variables,                         \
                           tmpl::size_t<DIM(data)>, Frame::Inertial>>>*>       \
          div_fluxes,                                                          \
      const Variables<typename ::SecondOrderScalarWave::System<DIM(            \
          data)>::gradient_variables>& source_vars,                            \
      const ::dg::Formulation dg_formulation, const Mesh<DIM(data)>& mesh,     \
      [[maybe_unused]] const tnsr::I<DataVector, DIM(data), Frame::Inertial>&  \
          inertial_coordinates,                                                \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            Frame::Inertial>&                                  \
          logical_to_inertial_inverse_jacobian,                                \
      [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,   \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&    \
          mesh_velocity,                                                       \
      const std::optional<Scalar<DataVector>>& div_mesh_velocity,              \
      const Scalar<DataVector>& pi);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM

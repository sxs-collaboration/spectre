// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <optional>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::Actions::detail {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                \
  template void                                                               \
  volume_terms<::GeneralizedHarmonic::TimeDerivative<DIM(data)>>(             \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::dt, typename ::GeneralizedHarmonic::System<DIM(             \
                          data)>::variables_tag::tags_list>>*>                \
          dt_vars_ptr,                                                        \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::Flux,                                                       \
          typename ::GeneralizedHarmonic::System<DIM(data)>::flux_variables,  \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          volume_fluxes,                                                      \
      const gsl::not_null<Variables<                                          \
          db::wrap_tags_in<::Tags::deriv,                                     \
                           typename ::GeneralizedHarmonic::System<DIM(        \
                               data)>::gradient_variables,                    \
                           tmpl::size_t<DIM(data)>, Frame::Inertial>>*>       \
          partial_derivs,                                                     \
      const gsl::not_null<Variables<typename ::GeneralizedHarmonic::System<   \
          DIM(data)>::compute_volume_time_derivative_terms::temporary_tags>*> \
          temporaries,                                                        \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::div,                                                        \
          db::wrap_tags_in<::Tags::Flux,                                      \
                           typename ::GeneralizedHarmonic::System<DIM(        \
                               data)>::flux_variables,                        \
                           tmpl::size_t<DIM(data)>, Frame::Inertial>>>*>      \
          div_fluxes,                                                         \
      const Variables<typename ::GeneralizedHarmonic::System<DIM(             \
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
      const tnsr::aa<DataVector, DIM(data)>& spacetime_metric,                \
      const tnsr::aa<DataVector, DIM(data)>& pi,                              \
      const tnsr::iaa<DataVector, DIM(data)>& phi,                            \
      const Scalar<DataVector>& gamma0, const Scalar<DataVector>& gamma1,     \
      const Scalar<DataVector>& gamma2,                                       \
      const tnsr::a<DataVector, DIM(data)>& gauge_function,                   \
      const tnsr::ab<DataVector, DIM(data)>& spacetime_deriv_gauge_function,  \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&   \
          mesh_velocity_from_time_deriv_args);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::Actions::detail

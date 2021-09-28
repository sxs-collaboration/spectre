// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/TimeDerivative.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::Actions::detail {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                \
  template void volume_terms<::CurvedScalarWave::TimeDerivative<DIM(data)>>(  \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::dt, typename ::CurvedScalarWave::System<DIM(                \
                          data)>::variables_tag::tags_list>>*>                \
          dt_vars_ptr,                                                        \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::Flux,                                                       \
          typename ::CurvedScalarWave::System<DIM(data)>::flux_variables,     \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          volume_fluxes,                                                      \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::deriv,                                                      \
          typename ::CurvedScalarWave::System<DIM(data)>::gradient_variables, \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          partial_derivs,                                                     \
      const gsl::not_null<Variables<typename ::CurvedScalarWave::System<DIM(  \
          data)>::compute_volume_time_derivative_terms::temporary_tags>*>     \
          temporaries,                                                        \
      const Variables<typename ::CurvedScalarWave::System<DIM(                \
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
      const Scalar<DataVector>& pi,                                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,             \
      const Scalar<DataVector>& lapse,                                        \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& deriv_lapse,     \
      const tnsr::iJ<DataVector, DIM(data), Frame::Inertial>& deriv_shift,    \
      const tnsr::II<DataVector, DIM(data), Frame::Inertial>&                 \
          upper_spatial_metric,                                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                  \
          trace_spatial_christoffel,                                          \
      const Scalar<DataVector>& trace_extrinsic_curvature,                    \
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::Actions::detail

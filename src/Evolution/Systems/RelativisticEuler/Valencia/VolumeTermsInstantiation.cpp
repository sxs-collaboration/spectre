// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/System.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/TimeDerivativeTerms.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace {
// The system is currently templated on the equation of state, but it's only
// used in code that will be removed and that we don't care about. Even the
// dependence on the thermodynamic dimension can probably be removed. In either
// case, we don't care about the thermodynamic dimension for the explicit
// instantiation either.
struct DummyEquationOfStateType {
  static constexpr size_t thermodynamic_dim = 1;
};
}  // namespace

namespace evolution::dg::Actions::detail {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define SYSTEM(data) \
  ::RelativisticEuler::Valencia::System<DIM(data), DummyEquationOfStateType>

#define INSTANTIATION(r, data)                                                 \
  template void                                                                \
  volume_terms<::RelativisticEuler::Valencia::TimeDerivativeTerms<DIM(data)>>( \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::dt, typename SYSTEM(data)::variables_tag::tags_list>>*>      \
          dt_vars_ptr,                                                         \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::Flux, typename SYSTEM(data)::flux_variables,                 \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                         \
          volume_fluxes,                                                       \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::deriv, typename SYSTEM(data)::gradient_variables,            \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                         \
          partial_derivs,                                                      \
      const gsl::not_null<Variables<typename SYSTEM(                           \
          data)::compute_volume_time_derivative_terms::temporary_tags>*>       \
          temporaries,                                                         \
      const Variables<typename SYSTEM(data)::variables_tag::tags_list>&        \
          evolved_vars,                                                        \
      const ::dg::Formulation dg_formulation, const Mesh<DIM(data)>& mesh,     \
      [[maybe_unused]] const tnsr::I<DataVector, DIM(data), Frame::Inertial>&  \
          inertial_coordinates,                                                \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,             \
                            Frame::Inertial>&                                  \
          logical_to_inertial_inverse_jacobian,                                \
      [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,   \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&    \
          mesh_velocity,                                                       \
      const std::optional<Scalar<DataVector>>& div_mesh_velocity,              \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,  \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& tilde_s,          \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,            \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                       \
      const Scalar<DataVector>& pressure,                                      \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& spatial_velocity, \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& d_lapse,          \
      const tnsr::iJ<DataVector, DIM(data), Frame::Inertial>& d_shift,         \
      const tnsr::ijj<DataVector, DIM(data), Frame::Inertial>&                 \
          d_spatial_metric,                                                    \
      const tnsr::II<DataVector, DIM(data), Frame::Inertial>&                  \
          inv_spatial_metric,                                                  \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&                  \
          extrinsic_curvature,                                                 \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&                  \
          spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef SYSTEM
#undef DIM
}  // namespace evolution::dg::Actions::detail

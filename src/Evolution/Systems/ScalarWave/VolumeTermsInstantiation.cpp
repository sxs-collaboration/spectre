// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::Actions::detail {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                \
  template void volume_terms<::ScalarWave::System<DIM(data)>>(                \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::dt, typename ::ScalarWave::System<DIM(                      \
                          data)>::variables_tag::tags_list>>*>                \
          dt_vars_ptr,                                                        \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::Flux,                                                       \
          typename ::ScalarWave::System<DIM(data)>::flux_variables,           \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          volume_fluxes,                                                      \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::deriv,                                                      \
          typename ::ScalarWave::System<DIM(data)>::gradient_variables,       \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          partial_derivs,                                                     \
      const gsl::not_null<Variables<typename ::ScalarWave::System<DIM(        \
          data)>::compute_volume_time_derivative_terms::temporary_tags>*>     \
          temporaries,                                                        \
      const Variables<typename ::ScalarWave::System<DIM(                      \
          data)>::variables_tag::tags_list>& evolved_vars,                    \
      const ::dg::Formulation dg_formulation, const Mesh<DIM(data)>& mesh,    \
      [[maybe_unused]] const tnsr::I<DataVector, DIM(data), Frame::Inertial>& \
          inertial_coordinates,                                               \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,            \
                            Frame::Inertial>&                                 \
          logical_to_inertial_inverse_jacobian,                               \
      [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,  \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&   \
          mesh_velocity,                                                      \
      const std::optional<Scalar<DataVector>>& div_mesh_velocity,             \
      const Scalar<DataVector>& pi,                                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,             \
      const Scalar<DataVector>& gamma2) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::Actions::detail

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/preprocessor/control/if.hpp>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/NewtonianEuler/System.hpp"
#include "Evolution/Systems/NewtonianEuler/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianEuler/KhInstability.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/LaneEmdenStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/RiemannProblem.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/SmoothFlow.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::Actions::detail {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INITIAL_DATA_TYPE(data)                           \
  BOOST_PP_IF(BOOST_PP_TUPLE_ELEM(2, data),               \
              BOOST_PP_TUPLE_ELEM(1, data) < DIM(data) >, \
              BOOST_PP_TUPLE_ELEM(1, data))
#define SYSTEM(data) \
  ::NewtonianEuler::System<DIM(data), INITIAL_DATA_TYPE(data)>

#define INSTANTIATION(r, data)                                                \
  template void volume_terms<::NewtonianEuler::TimeDerivativeTerms<           \
      DIM(data), INITIAL_DATA_TYPE(data)>>(                                   \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::dt, typename SYSTEM(data)::variables_tag::tags_list>>*>     \
          dt_vars_ptr,                                                        \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::Flux, typename SYSTEM(data)::flux_variables,                \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          volume_fluxes,                                                      \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::deriv, typename SYSTEM(data)::gradient_variables,           \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          partial_derivs,                                                     \
      const gsl::not_null<Variables<typename SYSTEM(                          \
          data)::compute_volume_time_derivative_terms::temporary_tags>*>      \
          temporaries,                                                        \
      const Variables<typename SYSTEM(data)::variables_tag::tags_list>&       \
          evolved_vars,                                                       \
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
      const tnsr::I<DataVector, DIM(data)>& momentum_density,                 \
      const Scalar<DataVector>& energy_density,                               \
      const tnsr::I<DataVector, DIM(data)>& velocity,                         \
      const Scalar<DataVector>& pressure);

// The last list passed to GENERATE_INSTANTIATIONS has only one element that
// must be `1` if the analytic solutions/data are templated on the dimension,
// and `0` if the analytic solutions/data are _not_ templated on dimension.
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (::NewtonianEuler::Solutions::RiemannProblem,
                         ::NewtonianEuler::Solutions::SmoothFlow),
                        (1))

GENERATE_INSTANTIATIONS(INSTANTIATION, (2, 3),
                        (::NewtonianEuler::Solutions::IsentropicVortex,
                         ::NewtonianEuler::AnalyticData::KhInstability),
                        (1))

GENERATE_INSTANTIATIONS(INSTANTIATION, (3),
                        (::NewtonianEuler::Solutions::LaneEmdenStar), (0))

#undef INSTANTIATION
#undef SYSTEM
#undef INITIAL_DATA_TYPE
#undef DIM
}  // namespace evolution::dg::Actions::detail

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
#define INITIAL_DATA_TYPE(data) BOOST_PP_TUPLE_ELEM(1, data)<DIM(data)>
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

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (::NewtonianEuler::Solutions::RiemannProblem,
                         ::NewtonianEuler::Solutions::SmoothFlow))

GENERATE_INSTANTIATIONS(INSTANTIATION, (2, 3),
                        (::NewtonianEuler::AnalyticData::KhInstability))

GENERATE_INSTANTIATIONS(INSTANTIATION, (2),
                        (::NewtonianEuler::Solutions::IsentropicVortex))

#undef INSTANTIATION
#undef SYSTEM
#undef INITIAL_DATA_TYPE
#undef DIM

using isentropic_vortex_3d_system = ::NewtonianEuler::System<
    3, ::NewtonianEuler::Solutions::IsentropicVortex<3>>;

template void volume_terms<::NewtonianEuler::TimeDerivativeTerms<
    3, ::NewtonianEuler::Solutions::IsentropicVortex<3>>>(
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::dt, isentropic_vortex_3d_system::variables_tag::tags_list>>*>
        dt_vars_ptr,
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::Flux, isentropic_vortex_3d_system::flux_variables,
        tmpl::size_t<3>, Frame::Inertial>>*>
        volume_fluxes,
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::deriv, isentropic_vortex_3d_system::gradient_variables,
        tmpl::size_t<3>, Frame::Inertial>>*>
        partial_derivs,
    const gsl::not_null<
        Variables<isentropic_vortex_3d_system::
                      compute_volume_time_derivative_terms::temporary_tags>*>
        temporaries,
    const Variables<isentropic_vortex_3d_system::variables_tag::tags_list>&
        evolved_vars,
    const ::dg::Formulation dg_formulation, const Mesh<3>& mesh,
    [[maybe_unused]] const tnsr::I<DataVector, 3, Frame::Inertial>&
        inertial_coordinates,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        logical_to_inertial_inverse_jacobian,
    [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const std::optional<Scalar<DataVector>>& div_mesh_velocity,
    const tnsr::I<DataVector, 3>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, 3>& velocity, const Scalar<DataVector>& pressure,
    const NewtonianEuler::Sources::VortexPerturbation& source,
    const NewtonianEuler::Solutions::IsentropicVortex<3>& vortex,
    const tnsr::I<DataVector, 3>& x, const double& time);

using lane_emden_star_system =
    ::NewtonianEuler::System<3, ::NewtonianEuler::Solutions::LaneEmdenStar>;

template void volume_terms<::NewtonianEuler::TimeDerivativeTerms<
    3, ::NewtonianEuler::Solutions::LaneEmdenStar>>(
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::dt,
        typename lane_emden_star_system::variables_tag::tags_list>>*>
        dt_vars_ptr,
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::Flux, typename lane_emden_star_system::flux_variables,
        tmpl::size_t<3>, Frame::Inertial>>*>
        volume_fluxes,
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::deriv, typename lane_emden_star_system::gradient_variables,
        tmpl::size_t<3>, Frame::Inertial>>*>
        partial_derivs,
    const gsl::not_null<
        Variables<typename lane_emden_star_system::
                      compute_volume_time_derivative_terms::temporary_tags>*>
        temporaries,
    const Variables<typename lane_emden_star_system::variables_tag::tags_list>&
        evolved_vars,
    const ::dg::Formulation dg_formulation, const Mesh<3>& mesh,
    [[maybe_unused]] const tnsr::I<DataVector, 3, Frame::Inertial>&
        inertial_coordinates,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        logical_to_inertial_inverse_jacobian,
    [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const std::optional<Scalar<DataVector>>& div_mesh_velocity,
    const tnsr::I<DataVector, 3>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, 3>& velocity, const Scalar<DataVector>& pressure,
    const NewtonianEuler::Sources::LaneEmdenGravitationalField& source,
    const Scalar<DataVector>& mass_density_cons_for_source,
    const tnsr::I<DataVector, 3>& momentum_density_for_source,
    const NewtonianEuler::Solutions::LaneEmdenStar& star,
    const tnsr::I<DataVector, 3>& x);
}  // namespace evolution::dg::Actions::detail

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Evolution/Systems/Burgers/TimeDerivativeTerms.hpp"

namespace evolution::dg::Actions::detail {
template void volume_terms<::Burgers::TimeDerivativeTerms>(
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::dt, typename ::Burgers::System::variables_tag::tags_list>>*>
        dt_vars_ptr,
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::Flux, typename ::Burgers::System::flux_variables,
        tmpl::size_t<1>, Frame::Inertial>>*>
        volume_fluxes,
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::deriv, typename ::Burgers::System::gradient_variables,
        tmpl::size_t<1>, Frame::Inertial>>*>
        partial_derivs,
    const gsl::not_null<
        Variables<typename ::Burgers::System::
                      compute_volume_time_derivative_terms::temporary_tags>*>
        temporaries,
    const Variables<typename ::Burgers::System::variables_tag::tags_list>&
        evolved_vars,
    const ::dg::Formulation dg_formulation, const Mesh<1>& mesh,
    [[maybe_unused]] const tnsr::I<DataVector, 1, Frame::Inertial>&
        inertial_coordinates,
    const InverseJacobian<DataVector, 1, Frame::ElementLogical,
                          Frame::Inertial>&
        logical_to_inertial_inverse_jacobian,
    [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,
    const std::optional<tnsr::I<DataVector, 1, Frame::Inertial>>& mesh_velocity,
    const std::optional<Scalar<DataVector>>& div_mesh_velocity,
    const Scalar<DataVector>& u);
}  // namespace evolution::dg::Actions::detail

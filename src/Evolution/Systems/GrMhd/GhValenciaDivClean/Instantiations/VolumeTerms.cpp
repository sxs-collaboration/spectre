// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <optional>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/TimeDerivativeTerms.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Utilities/GenerateInstantiations.hpp"

#define NEUTRINO(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template void evolution::dg::Actions::detail::volume_terms<                  \
      ::grmhd::GhValenciaDivClean::TimeDerivativeTerms>(                       \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::dt, typename ::grmhd::GhValenciaDivClean::System<NEUTRINO(   \
                          data)>::variables_tag::tags_list>>*>                 \
          dt_vars_ptr,                                                         \
      const gsl::not_null<Variables<                                           \
          db::wrap_tags_in<::Tags::Flux,                                       \
                           typename ::grmhd::GhValenciaDivClean::System<       \
                               NEUTRINO(data)>::flux_variables,                \
                           tmpl::size_t<3>, Frame::Inertial>>*>                \
          volume_fluxes,                                                       \
      const gsl::not_null<Variables<                                           \
          db::wrap_tags_in<::Tags::deriv,                                      \
                           typename ::grmhd::GhValenciaDivClean::System<       \
                               NEUTRINO(data)>::gradient_variables,            \
                           tmpl::size_t<3>, Frame::Inertial>>*>                \
          partial_derivs,                                                      \
      const gsl::not_null<                                                     \
          Variables<typename ::grmhd::GhValenciaDivClean::System<NEUTRINO(     \
              data)>::compute_volume_time_derivative_terms::temporary_tags>*>  \
          temporaries,                                                         \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::div,                                                         \
          db::wrap_tags_in<::Tags::Flux,                                       \
                           typename ::grmhd::GhValenciaDivClean::System<       \
                               NEUTRINO(data)>::flux_variables,                \
                           tmpl::size_t<3>, Frame::Inertial>>>*>               \
          div_fluxes,                                                          \
      const Variables<typename ::grmhd::GhValenciaDivClean::System<NEUTRINO(   \
          data)>::variables_tag::tags_list>& evolved_vars,                     \
      const ::dg::Formulation dg_formulation, const Mesh<3>& mesh,             \
      [[maybe_unused]] const tnsr::I<DataVector, 3, Frame::Inertial>&          \
          inertial_coordinates,                                                \
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,              \
                            Frame::Inertial>&                                  \
          logical_to_inertial_inverse_jacobian,                                \
      [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,   \
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&            \
          mesh_velocity,                                                       \
      const std::optional<Scalar<DataVector>>& div_mesh_velocity,              \
      const tnsr::aa<DataVector, 3>& spacetime_metric,                         \
      const tnsr::aa<DataVector, 3>& pi, const tnsr::iaa<DataVector, 3>& phi,  \
      const Scalar<DataVector>& gamma0, const Scalar<DataVector>& gamma1,      \
      const Scalar<DataVector>& gamma2,                                        \
      const ::gh::gauges::GaugeCondition& gauge_condition,                     \
      const Mesh<3>& mesh_for_rhs, const double& time,                         \
      const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,          \
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,              \
                            Frame::Inertial>& inverse_jacobian,                \
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&            \
          mesh_velocity_gh,                                                    \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,   \
      const Scalar<DataVector>& tilde_tau,                                     \
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,                  \
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,                  \
      const Scalar<DataVector>& tilde_phi, const Scalar<DataVector>& pressure, \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const double& constraint_damping_parameter,                              \
      const ::VariableFixing::FixToAtmosphere<3>& fix_to_atmosphere);          \
  INSTANTIATE_PARTIAL_DERIVATIVES_WITH_SYSTEM(                                 \
      grmhd::GhValenciaDivClean::System<NEUTRINO(data)>, 3, Frame::Inertial)

GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (RadiationTransport::NoNeutrinos::System))

#undef INSTANTIATION
#undef NEUTRINO

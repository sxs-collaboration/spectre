// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TimeDerivativeTerms.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::GhValenciaDivClean {
namespace detail {
// Some temporary tags appear both in the GRMHD temporary list and the
// GeneralizedHarmonic temporary list, so we wrap the GRMHD temporaries in this
// prefix tag to avoid collisions in data structures used to store the
// temporaries.
template <typename Tag>
struct ValenciaTempTag : db::SimpleTag, db::PrefixTag {
  using tag = Tag;
  using type = typename Tag::type;
};

template <typename GhDtTagList, typename ValenciaDtTagList,
          typename ValenciaFluxTagList, typename GhTempTagList,
          typename ValenciaTempTagList, typename GhGradientTagList,
          typename GhArgTagList, typename ValenciaArgTagList,
          typename ValenciaTimeDerivativeArgTagList>
struct TimeDerivativeTermsImpl;

template <typename... GhDtTags, typename... ValenciaDtTags,
          typename... ValenciaFluxTags, typename... GhTempTags,
          typename... ValenciaTempTags, typename... GhGradientTags,
          typename... GhArgTags, typename... ValenciaArgTags,
          typename... ValenciaTimeDerivativeArgTags>
struct TimeDerivativeTermsImpl<
    tmpl::list<GhDtTags...>, tmpl::list<ValenciaDtTags...>,
    tmpl::list<ValenciaFluxTags...>, tmpl::list<GhTempTags...>,
    tmpl::list<ValenciaTempTags...>, tmpl::list<GhGradientTags...>,
    tmpl::list<GhArgTags...>, tmpl::list<ValenciaArgTags...>,
    tmpl::list<ValenciaTimeDerivativeArgTags...>> {
  static void apply(
      const gsl::not_null<typename GhDtTags::type*>... gh_dts,
      const gsl::not_null<typename ValenciaDtTags::type*>... valencia_dts,
      const gsl::not_null<typename ValenciaFluxTags::type*>... valencia_fluxes,
      const gsl::not_null<typename GhTempTags::type*>... gh_temporaries,
      const gsl::not_null<
          typename ValenciaTempTags::type*>... valencia_temporaries,
      gsl::not_null<tnsr::aa<DataVector, 3_st>*> stress_energy,
      gsl::not_null<tnsr::a<DataVector, 3_st>*> four_velocity_one_form,
      gsl::not_null<tnsr::a<DataVector, 3_st>*>
          comoving_magnetic_field_one_form,
      const typename ::Tags::deriv<GhGradientTags, tmpl::size_t<3_st>,
                                   Frame::Inertial>::type&... gh_gradients,
      const typename GhArgTags::type&... gh_args,
      const typename ValenciaArgTags::type&... valencia_args) {
    GeneralizedHarmonic::TimeDerivative<3_st>::apply(
        gh_dts..., gh_temporaries..., gh_gradients..., gh_args...);

    // This is needed to be able to reuse temporary tags that the GH system
    // computed already and pass them in as argument tags to the GRMHD system.
    // Because the tag lists have order alterations from filtering the argument
    // tags, we use a tagged tuple to reorder the references to the order
    // expected by the GRMHD time derivative calculation.
    tuples::TaggedTuple<Tags::detail::TemporaryReference<ValenciaArgTags>...,
                        Tags::detail::TemporaryReference<ValenciaTempTags>...,
                        Tags::detail::TemporaryReference<GhTempTags>...>
    shuffle_refs(valencia_args..., *valencia_temporaries...,
                 *gh_temporaries...);

    grmhd::ValenciaDivClean::TimeDerivativeTerms::apply(
        valencia_dts..., valencia_fluxes..., valencia_temporaries...,
        tuples::get<
            Tags::detail::TemporaryReference<ValenciaTimeDerivativeArgTags>>(
            shuffle_refs)...);
    dispatch_to_stress_energy_calculation(stress_energy, four_velocity_one_form,
                                          comoving_magnetic_field_one_form,
                                          gh_args..., shuffle_refs);
    dispatch_to_add_stress_energy_term_to_dt_pi(gh_dts..., shuffle_refs,
                                                *stress_energy);
  }

 private:
  template <typename TupleType>
  static void dispatch_to_add_stress_energy_term_to_dt_pi(
      gsl::not_null<tnsr::aa<DataVector, 3_st>*> /*dt_spacetime_metric*/,
      gsl::not_null<tnsr::aa<DataVector, 3_st>*> dt_pi,
      gsl::not_null<tnsr::iaa<DataVector, 3_st>*> /*dt_phi*/,
      const TupleType& args, const tnsr::aa<DataVector, 3_st>& stress_energy) {
    add_stress_energy_term_to_dt_pi(
        dt_pi, stress_energy,
        tuples::get<
            Tags::detail::TemporaryReference<gr::Tags::Lapse<DataVector>>>(
            args));
  }

  template <typename TupleType>
  static void dispatch_to_stress_energy_calculation(
      gsl::not_null<tnsr::aa<DataVector, 3_st>*> local_stress_energy,
      gsl::not_null<tnsr::a<DataVector, 3_st>*> four_velocity_buffer_one_form,
      gsl::not_null<tnsr::a<DataVector, 3_st>*>
          comoving_magnetic_field_one_form,
      const tnsr::aa<DataVector, 3_st>& spacetime_metric,
      const tnsr::aa<DataVector, 3_st>& /*pi*/,
      const tnsr::iaa<DataVector, 3_st>& /*phi*/,
      const Scalar<DataVector>& /*gamma0*/,
      const Scalar<DataVector>& /*gamma1*/,
      const Scalar<DataVector>& /*gamma2*/,
      const tnsr::a<DataVector, 3_st>& /*gauge_function*/,
      const tnsr::ab<DataVector, 3_st>& /*spacetime_deriv_gauge_function*/,
      const TupleType& args) {
    trace_reversed_stress_energy(
        local_stress_energy, four_velocity_buffer_one_form,
        comoving_magnetic_field_one_form,
        tuples::get<Tags::detail::TemporaryReference<
            hydro::Tags::RestMassDensity<DataVector>>>(args),
        tuples::get<Tags::detail::TemporaryReference<
            hydro::Tags::SpecificEnthalpy<DataVector>>>(args),
        tuples::get<Tags::detail::TemporaryReference<
            ValenciaTempTag<hydro::Tags::SpatialVelocityOneForm<
                DataVector, 3_st, Frame::Inertial>>>>(args),
        tuples::get<Tags::detail::TemporaryReference<
            ValenciaTempTag<hydro::Tags::MagneticFieldOneForm<
                DataVector, 3_st, Frame::Inertial>>>>(args),
        tuples::get<Tags::detail::TemporaryReference<
            ValenciaTempTag<hydro::Tags::MagneticFieldSquared<DataVector>>>>(
            args),
        tuples::get<Tags::detail::TemporaryReference<ValenciaTempTag<
            hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>>>(args),
        tuples::get<Tags::detail::TemporaryReference<
            hydro::Tags::LorentzFactor<DataVector>>>(args),
        tuples::get<Tags::detail::TemporaryReference<
            ValenciaTempTag<grmhd::ValenciaDivClean::TimeDerivativeTerms::
                                OneOverLorentzFactorSquared>>>(args),
        tuples::get<Tags::detail::TemporaryReference<
            hydro::Tags::Pressure<DataVector>>>(args),
        spacetime_metric,
        tuples::get<Tags::detail::TemporaryReference<
            gr::Tags::Shift<3_st, Frame::Inertial, DataVector>>>(args),
        tuples::get<
            Tags::detail::TemporaryReference<gr::Tags::Lapse<DataVector>>>(
            args));
  }
};
}  // namespace detail

/*!
 * \brief Compute the RHS terms and flux values for both the Generalized
 * Harmonic formulation of Einstein's equations and the Valencia formulation of
 * the GRMHD equations with divergence cleaning.
 *
 * \details The bulk of the computations in this class dispatch to
 * `GeneralizedHarmonic::TimeDerivative` and
 * `grmhd::ValenciaDivClean::TimeDerivativeTerms` as a 'product system' -- each
 * independently operating on its own subset of the supplied variable
 * collections.
 * The additional step is taken to compute the trace-reversed stress energy
 * tensor associated with the GRMHD part of the system and add its contribution
 * to the \f$\partial_t \Pi_{a b}\f$ variable in the Generalized Harmonic
 * system, which is the only explicit coupling required to back-react the effect
 * of matter on the spacetime solution.
 *
 * \note The MHD calculation reuses any spacetime quantities in its
 * argument_tags that are computed by the GH time derivative. However, other
 * quantities that aren't computed by the GH time derivative like the extrinsic
 * curvature are currently still retrieved from the DataBox. Those calculations
 * can be explicitly inlined here to reduce memory pressure and the number of
 * compute tags.
 */
struct TimeDerivativeTerms {
  using valencia_dt_tags = db::wrap_tags_in<
      ::Tags::dt,
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list>;
  using valencia_flux_tags = tmpl::transform<
      typename grmhd::ValenciaDivClean::System::flux_variables,
      tmpl::bind<::Tags::Flux, tmpl::_1, tmpl::pin<tmpl::size_t<3_st>>,
                 tmpl::pin<Frame::Inertial>>>;

  using gh_dt_tags = db::wrap_tags_in<
      ::Tags::dt,
      typename GeneralizedHarmonic::System<3_st>::variables_tag::tags_list>;
  using gh_temp_tags =
      typename GeneralizedHarmonic::TimeDerivative<3_st>::temporary_tags;
  using gh_gradient_tags =
      typename GeneralizedHarmonic::System<3_st>::gradients_tags;
  using gh_arg_tags =
      typename GeneralizedHarmonic::TimeDerivative<3_st>::argument_tags;

  using valencia_temp_tags = db::wrap_tags_in<
      detail::ValenciaTempTag,
      typename grmhd::ValenciaDivClean::TimeDerivativeTerms::temporary_tags>;
  using valencia_arg_tags = tmpl::list_difference<
      typename grmhd::ValenciaDivClean::TimeDerivativeTerms::argument_tags,
      gh_temp_tags>;

  using temporary_tags = tmpl::flatten<tmpl::list<
      gh_temp_tags, valencia_temp_tags, Tags::TraceReversedStressEnergy,
      Tags::FourVelocityOneForm, Tags::ComovingMagneticFieldOneForm>>;
  using argument_tags = tmpl::append<gh_arg_tags, valencia_arg_tags>;

  template <typename... Args>
  static void apply(Args&&... args) {
    detail::TimeDerivativeTermsImpl<
        gh_dt_tags, valencia_dt_tags, valencia_flux_tags, gh_temp_tags,
        valencia_temp_tags, gh_gradient_tags, gh_arg_tags, valencia_arg_tags,
        typename grmhd::ValenciaDivClean::TimeDerivativeTerms::argument_tags>::
        apply(std::forward<Args>(args)...);
  }
};
}  // namespace grmhd::GhValenciaDivClean

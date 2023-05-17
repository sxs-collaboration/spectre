// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/PassVariables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
          typename ValenciaTimeDerivativeArgTagList,
          typename TraceReversedStressResultTagsList,
          typename TraceReversedStressArgumentTagsList>
struct TimeDerivativeTermsImpl;

template <typename... GhDtTags, typename... ValenciaDtTags,
          typename... ValenciaFluxTags, typename... GhTempTags,
          typename... ValenciaTempTags, typename... GhGradientTags,
          typename... GhArgTags, typename... ValenciaArgTags,
          typename... ValenciaTimeDerivativeArgTags,
          typename... TraceReversedStressResultTags,
          typename... TraceReversedStressArgumentTags>
struct TimeDerivativeTermsImpl<
    tmpl::list<GhDtTags...>, tmpl::list<ValenciaDtTags...>,
    tmpl::list<ValenciaFluxTags...>, tmpl::list<GhTempTags...>,
    tmpl::list<ValenciaTempTags...>, tmpl::list<GhGradientTags...>,
    tmpl::list<GhArgTags...>, tmpl::list<ValenciaArgTags...>,
    tmpl::list<ValenciaTimeDerivativeArgTags...>,
    tmpl::list<TraceReversedStressResultTags...>,
    tmpl::list<TraceReversedStressArgumentTags...>> {
  template <typename TemporaryTagsList, typename... ExtraTags>
  static void apply(
      const gsl::not_null<
          Variables<tmpl::list<GhDtTags..., ValenciaDtTags...>>*>
          dt_vars_ptr,
      const gsl::not_null<Variables<db::wrap_tags_in<
          ::Tags::Flux, typename ValenciaDivClean::System::flux_variables,
          tmpl::size_t<3>, Frame::Inertial>>*>
          fluxes_ptr,
      const gsl::not_null<Variables<TemporaryTagsList>*> temps_ptr,

      const tnsr::iaa<DataVector, 3>& d_spacetime_metric,
      const tnsr::iaa<DataVector, 3>& d_pi,
      const tnsr::ijaa<DataVector, 3>& d_phi,

      const tuples::TaggedTuple<ExtraTags...>& arguments) {
    gh::TimeDerivative<3_st>::apply(
        get<GhDtTags>(dt_vars_ptr)..., get<GhTempTags>(temps_ptr)...,
        d_spacetime_metric, d_pi, d_phi,
        get<Tags::detail::TemporaryReference<GhArgTags>>(arguments)...);

    for (size_t i = 0; i < 3; ++i) {
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(*temps_ptr)
          .get(i) = -get(get<gr::Tags::Lapse<DataVector>>(*temps_ptr)) *
                    get<gh::Tags::HalfPhiTwoNormals<3>>(*temps_ptr).get(i);
    }
    const auto& phi =
        get<Tags::detail::TemporaryReference<gh::Tags::Phi<DataVector, 3>>>(
            arguments);
    const auto& inv_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(*temps_ptr);
    const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(*temps_ptr);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                          Frame::Inertial>>(*temps_ptr)
            .get(i, j) = inv_spatial_metric.get(j, 0) * phi.get(i, 0, 1);
        for (size_t k = 1; k < 3; ++k) {
          get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                            Frame::Inertial>>(*temps_ptr)
              .get(i, j) += inv_spatial_metric.get(j, k) * phi.get(i, 0, k + 1);
        }
        for (size_t k = 0; k < 3; ++k) {
          for (size_t l = 0; l < 3; ++l) {
            get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                              Frame::Inertial>>(*temps_ptr)
                .get(i, j) -= shift.get(k) * inv_spatial_metric.get(j, l) *
                              phi.get(i, l + 1, k + 1);
          }
        }
      }
    }

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = j; k < 3; ++k) {
          // NOTE: it would be nice if we could just make this a reference...
          get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                            tmpl::size_t<3>, Frame::Inertial>>(*temps_ptr)
              .get(i, j, k) = phi.get(i, j + 1, k + 1);
        }
      }
    }
    const auto& pi =
        get<Tags::detail::TemporaryReference<gh::Tags::Pi<DataVector, 3>>>(
            arguments);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(*temps_ptr).get(i, j) =
            0.5 * (pi.get(i + 1, j + 1) +
                   get<gh::Tags::PhiOneNormal<3>>(*temps_ptr).get(i, j + 1) +
                   get<gh::Tags::PhiOneNormal<3>>(*temps_ptr).get(j, i + 1));
      }
    }

    using extra_tags_list = tmpl::list<ExtraTags...>;

    grmhd::ValenciaDivClean::TimeDerivativeTerms::apply(
        get<ValenciaDtTags>(dt_vars_ptr)...,
        get<ValenciaFluxTags>(fluxes_ptr)...,
        get<ValenciaTempTags>(temps_ptr)...,

        get<tmpl::conditional_t<
            tmpl::list_contains_v<extra_tags_list,
                                  Tags::detail::TemporaryReference<
                                      ValenciaTimeDerivativeArgTags>>,
            Tags::detail::TemporaryReference<ValenciaTimeDerivativeArgTags>,
            ValenciaTimeDerivativeArgTags>>(arguments, *temps_ptr)...);

    trace_reversed_stress_energy(
        get<TraceReversedStressResultTags>(temps_ptr)...,
        get<tmpl::conditional_t<
            tmpl::list_contains_v<extra_tags_list,
                                  Tags::detail::TemporaryReference<
                                      TraceReversedStressArgumentTags>>,
            Tags::detail::TemporaryReference<TraceReversedStressArgumentTags>,
            TraceReversedStressArgumentTags>>(*temps_ptr, arguments)...);

    // The addition to dt Pi is independent of the specific form of the stress
    // tensor.
    add_stress_energy_term_to_dt_pi(
        get<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>(dt_vars_ptr),
        get<grmhd::GhValenciaDivClean::Tags::TraceReversedStressEnergy>(
            *temps_ptr),
        get<gr::Tags::Lapse<DataVector>>(*temps_ptr));
  }
};
}  // namespace detail

/*!
 * \brief Compute the RHS terms and flux values for both the Generalized
 * Harmonic formulation of Einstein's equations and the Valencia formulation of
 * the GRMHD equations with divergence cleaning.
 *
 * \details The bulk of the computations in this class dispatch to
 * `gh::TimeDerivative` and
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
struct TimeDerivativeTerms : evolution::PassVariables {
  using gh_dt_tags =
      db::wrap_tags_in<::Tags::dt,
                       typename gh::System<3_st>::variables_tag::tags_list>;
  using valencia_dt_tags = db::wrap_tags_in<
      ::Tags::dt,
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list>;

  using dt_tags = tmpl::append<gh_dt_tags, valencia_dt_tags>;

  using valencia_flux_tags = tmpl::transform<
      typename grmhd::ValenciaDivClean::System::flux_variables,
      tmpl::bind<::Tags::Flux, tmpl::_1, tmpl::pin<tmpl::size_t<3_st>>,
                 tmpl::pin<Frame::Inertial>>>;

  using gh_temp_tags = typename gh::TimeDerivative<3_st>::temporary_tags;
  using gh_gradient_tags = typename gh::System<3_st>::gradients_tags;
  using gh_arg_tags = typename gh::TimeDerivative<3_st>::argument_tags;

  using valencia_temp_tags =
      typename grmhd::ValenciaDivClean::TimeDerivativeTerms::temporary_tags;
  // Additional temp tags are the derivatives of the metric since GH doesn't
  // explicitly calculate those.
  using valencia_extra_temp_tags =
      tmpl::list<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>>;
  using valencia_arg_tags = tmpl::list_difference<
      typename grmhd::ValenciaDivClean::TimeDerivativeTerms::argument_tags,
      tmpl::append<gh_temp_tags, valencia_extra_temp_tags>>;

  using trace_reversed_stress_result_tags =
      tmpl::list<Tags::TraceReversedStressEnergy, Tags::FourVelocityOneForm,
                 Tags::ComovingMagneticFieldOneForm>;
  using trace_reversed_stress_argument_tags = tmpl::list<
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpecificEnthalpy<DataVector>,
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3_st, Frame::Inertial>,
      hydro::Tags::MagneticFieldOneForm<DataVector, 3_st, Frame::Inertial>,
      hydro::Tags::MagneticFieldSquared<DataVector>,
      hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
      hydro::Tags::LorentzFactor<DataVector>,
      grmhd::ValenciaDivClean::TimeDerivativeTerms::OneOverLorentzFactorSquared,
      hydro::Tags::Pressure<DataVector>,
      gr::Tags::SpacetimeMetric<DataVector, 3>,
      gr::Tags::Shift<DataVector, 3_st>, gr::Tags::Lapse<DataVector>>;

  using temporary_tags = tmpl::remove_duplicates<
      tmpl::append<gh_temp_tags, valencia_temp_tags, valencia_extra_temp_tags,
                   trace_reversed_stress_result_tags>>;
  using argument_tags = tmpl::append<gh_arg_tags, valencia_arg_tags>;

  template <typename... Args>
  static void apply(
      const gsl::not_null<Variables<dt_tags>*> dt_vars_ptr,
      const gsl::not_null<Variables<db::wrap_tags_in<
          ::Tags::Flux, typename ValenciaDivClean::System::flux_variables,
          tmpl::size_t<3>, Frame::Inertial>>*>
          fluxes_ptr,
      const gsl::not_null<Variables<temporary_tags>*> temps_ptr,
      const tnsr::iaa<DataVector, 3>& d_spacetime_metric,
      const tnsr::iaa<DataVector, 3>& d_pi,
      const tnsr::ijaa<DataVector, 3>& d_phi, const Args&... args) {
    const tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<Tags::detail::TemporaryReference, argument_tags>>
        arguments{args...};
    detail::TimeDerivativeTermsImpl<
        gh_dt_tags, valencia_dt_tags, valencia_flux_tags, gh_temp_tags,
        valencia_temp_tags, gh_gradient_tags, gh_arg_tags, valencia_arg_tags,
        typename grmhd::ValenciaDivClean::TimeDerivativeTerms::argument_tags,
        trace_reversed_stress_result_tags,
        trace_reversed_stress_argument_tags>::apply(dt_vars_ptr, fluxes_ptr,
                                                    temps_ptr,
                                                    d_spacetime_metric, d_pi,
                                                    d_phi, arguments);
  }
};
}  // namespace grmhd::GhValenciaDivClean

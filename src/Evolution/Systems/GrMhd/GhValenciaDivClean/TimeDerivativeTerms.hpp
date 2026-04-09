// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeDerivativeDecisions.hpp"
#include "Evolution/PassVariables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Harmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TimeDerivativeTerms.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp" // For reference tag
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

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

namespace dt_type_aliases {
using gh_dt_tags =
    db::wrap_tags_in<::Tags::dt,
                     typename gh::System<3_st>::variables_tag::tags_list>;
using valencia_dt_tags = db::wrap_tags_in<
    ::Tags::dt,
    typename grmhd::ValenciaDivClean::System::variables_tag::tags_list>;

using dt_tags = tmpl::append<gh_dt_tags, valencia_dt_tags>;
}  // namespace dt_type_aliases

struct TimeDerivativeTermsImpl;
struct TimeDerivativeTermsImpl {
  // Tags related to GeneralizedHarmonic system
  using gh_arg_tags = typename gh::TimeDerivative<
      ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list,
      3_st>::argument_tags;

  using gh_temp_tags = typename gh::TimeDerivative<
      ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list,
      3_st>::temporary_tags;

  using valencia_flux_tags = db::wrap_tags_in<
      ::Tags::Flux,
      typename grmhd::ValenciaDivClean::System::variables_tag::tags_list,
      tmpl::size_t<3>, Frame::Inertial>;

  using valencia_temp_tags =
      typename grmhd::ValenciaDivClean::TimeDerivativeTerms::temporary_tags;

  using valencia_time_derivative_arg_tags =
      typename grmhd::ValenciaDivClean::TimeDerivativeTerms::argument_tags;

  using trace_reversed_stress_result_tags =
      tmpl::list<Tags::TraceReversedStressEnergy, Tags::FourVelocityOneForm,
                 Tags::ComovingMagneticFieldOneForm>;
  using trace_reversed_stress_argument_tags = tmpl::list<
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3_st, Frame::Inertial>,
      hydro::Tags::MagneticFieldOneForm<DataVector, 3_st, Frame::Inertial>,
      hydro::Tags::MagneticFieldSquared<DataVector>,
      hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
      hydro::Tags::LorentzFactor<DataVector>,
      grmhd::ValenciaDivClean::TimeDerivativeTerms::OneOverLorentzFactorSquared,
      hydro::Tags::Pressure<DataVector>,
      hydro::Tags::SpecificInternalEnergy<DataVector>,
      gr::Tags::SpacetimeMetric<DataVector, 3>,
      gr::Tags::Shift<DataVector, 3_st>, gr::Tags::Lapse<DataVector>>;

  template <typename TemporaryTagsList, typename... ExtraTags>
  static evolution::dg::TimeDerivativeDecisions<3> apply(
      const gsl::not_null<Variables<dt_type_aliases::dt_tags>*> dt_vars_ptr,
      const gsl::not_null<Variables<db::wrap_tags_in<
          ::Tags::Flux, typename ValenciaDivClean::System::flux_variables,
          tmpl::size_t<3>, Frame::Inertial>>*>
          fluxes_ptr,
      const gsl::not_null<Variables<TemporaryTagsList>*> temps_ptr,

      const tnsr::iaa<DataVector, 3>& d_spacetime_metric,
      const tnsr::iaa<DataVector, 3>& d_pi,
      const tnsr::ijaa<DataVector, 3>& d_phi,

      const tuples::TaggedTuple<ExtraTags...>& arguments) {
    generalized_harmonic_time_derivative(
        dt_vars_ptr, temps_ptr, d_spacetime_metric, d_pi, d_phi, arguments,
        dt_type_aliases::gh_dt_tags{}, gh_temp_tags{}, gh_arg_tags{});

    // Track whether or not we extracted the 3+1 quantities
    bool three_plus_one_extracted = false;

    // If we are in the atmosphere, then we can skip the evolution
    // of the GRMHD system completely. This inevitably depends on parameters
    // of the FixToAtmosphere code, so we grab those directly.
    if (const auto& fix_to_atmosphere = get<Tags::detail::TemporaryReference<
            ::Tags::VariableFixer<::VariableFixing::FixToAtmosphere<3>>>>(
            arguments);
        max(get(get<Tags::detail::TemporaryReference<
                    hydro::Tags::RestMassDensity<DataVector>>>(arguments))) <=
        std::min({fix_to_atmosphere.density_of_atmosphere(),
                  (fix_to_atmosphere.velocity_limiting().has_value()
                       ? fix_to_atmosphere.velocity_limiting()
                             ->atmosphere_density_cutoff
                       : std::numeric_limits<double>::infinity()),
                  (fix_to_atmosphere.kappa_limiting().has_value()
                       ? fix_to_atmosphere.kappa_limiting()->density_lower_bound
                       : std::numeric_limits<double>::infinity())}) *
            (1.0 + 10.0 * std::numeric_limits<double>::epsilon())) {
      // Point into the right memory, then set it to zero.

      ASSERT(
          max(get(get<tmpl::front<dt_type_aliases::valencia_dt_tags>>(
              *dt_vars_ptr))) == 0.0,
          "GH+GRMHD assumes the time derivatives are set to zero in general."
          " If this is no longer the case, please set them to zero in "
          "atmosphere by changing the code where this ASSERT was triggered.");
      // Code that we could use to set the sources to zero if needed.
      // Variables<tmpl::list<ValenciaDtTags...>> dt_div_clean(
      //     get<tmpl::front<tmpl::list<ValenciaDtTags...>>>(*dt_vars_ptr)[0]
      //         .data(),
      //     0.0);
      fluxes_ptr->initialize(fluxes_ptr->number_of_grid_points(), 0.0);
      return evolution::dg::TimeDerivativeDecisions<3>{false};
    } else {
      // MHD calls

      // extracting 3+1 quantities to assign spacetime spatial derivatives
      extract_three_plus_one_quantities(temps_ptr, arguments);
      three_plus_one_extracted = true;

      // MHD computation
      aggregate_time_derivative_terms(
          dt_vars_ptr, fluxes_ptr, temps_ptr, arguments,
          dt_type_aliases::valencia_dt_tags{}, valencia_flux_tags{},
          valencia_temp_tags{}, valencia_time_derivative_arg_tags{},
          trace_reversed_stress_result_tags{},
          trace_reversed_stress_argument_tags{});
    }

    if (!three_plus_one_extracted) {
      // If in atmosphere, extract 3+1 quantities for neutrino
      // evolution
      extract_three_plus_one_quantities(temps_ptr, arguments);
      three_plus_one_extracted = true;
      // Neutrino evolution will be called below
    }

    return evolution::dg::TimeDerivativeDecisions<3>{true};
  }

  template <typename OutputTags, typename TemporaryTagsList,
            typename... ExtraTags, typename... GhDtTags, typename... GhTempTags,
            typename... GhArgTags>
  static void generalized_harmonic_time_derivative(
      const gsl::not_null<Variables<OutputTags>*> dt_vars_ptr,
      const gsl::not_null<Variables<TemporaryTagsList>*> temps_ptr,
      const tnsr::iaa<DataVector, 3>& d_spacetime_metric,
      const tnsr::iaa<DataVector, 3>& d_pi,
      const tnsr::ijaa<DataVector, 3>& d_phi,
      const tuples::TaggedTuple<ExtraTags...>& arguments,
      tmpl::list<GhDtTags...> /*meta*/, tmpl::list<GhTempTags...> /*meta*/,
      tmpl::list<GhArgTags...> /*meta*/) {
    gh::TimeDerivative<
        ghmhd::GhValenciaDivClean::InitialData::
            analytic_solutions_and_data_list,
        3_st>::apply(get<GhDtTags>(dt_vars_ptr)...,
                     get<GhTempTags>(temps_ptr)..., d_spacetime_metric, d_pi,
                     d_phi,
                     get<Tags::detail::TemporaryReference<GhArgTags>>(
                         arguments)...);
  }

  template <typename TemporaryTagsList, typename... ExtraTags>
  static void extract_three_plus_one_quantities(
      const gsl::not_null<Variables<TemporaryTagsList>*> temps_ptr,
      const tuples::TaggedTuple<ExtraTags...>& arguments) {
    // Extract the 3+1 quantities from the spacetime metric

    // Check whether or not sqrt(det(g)) exists based on gauge condition
    if (get<Tags::detail::TemporaryReference<gh::gauges::Tags::GaugeCondition>>(
            arguments)
            .is_harmonic()) {
      get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(*temps_ptr)) =
          sqrt(get(get<gr::Tags::DetSpatialMetric<DataVector>>(*temps_ptr)));
    }

    // extract spatial derivative of lapse
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

    // extract spatial derivative of shift
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

  }  // end extract3plus1

  template <typename OutputTags, typename TemporaryTagsList,
            typename... ExtraTags, typename... ValenciaDtTags,
            typename... ValenciaFluxTags, typename... ValenciaTempTags,
            typename... ValenciaTimeDerivativeArgTags,
            typename... TraceReversedStressResultTags,
            typename... TraceReversedStressArgumentTags>
  static void aggregate_time_derivative_terms(
      const gsl::not_null<Variables<OutputTags>*> dt_vars_ptr,
      const gsl::not_null<Variables<db::wrap_tags_in<
          ::Tags::Flux, typename ValenciaDivClean::System::flux_variables,
          tmpl::size_t<3>, Frame::Inertial>>*>
          fluxes_ptr,
      const gsl::not_null<Variables<TemporaryTagsList>*> temps_ptr,
      const tuples::TaggedTuple<ExtraTags...>& arguments,
      tmpl::list<ValenciaDtTags...> /*meta*/,
      tmpl::list<ValenciaFluxTags...> /*meta*/,
      tmpl::list<ValenciaTempTags...> /*meta*/,
      tmpl::list<ValenciaTimeDerivativeArgTags...> /*meta*/,
      tmpl::list<TraceReversedStressResultTags...> /*meta*/,
      tmpl::list<TraceReversedStressArgumentTags...> /*meta*/) {
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

    add_stress_energy_term_to_dt_pi(
        get<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>(dt_vars_ptr),
        get<grmhd::GhValenciaDivClean::Tags::TraceReversedStressEnergy>(
            *temps_ptr),
        get<gr::Tags::Lapse<DataVector>>(*temps_ptr));
  }
};  // namespace detail
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

  using d_spatial_metric = ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                                         tmpl::size_t<3>, Frame::Inertial>;

  using gh_temp_tags = typename gh::TimeDerivative<
      ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list,
      3_st>::temporary_tags;
  using gh_gradient_tags = typename gh::System<3_st>::gradients_tags;
  using gh_arg_tags = typename gh::TimeDerivative<
      ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list,
      3_st>::argument_tags;

  using valencia_temp_tags =
      typename grmhd::ValenciaDivClean::TimeDerivativeTerms::temporary_tags;
  // Additional temp tags are the derivatives of the metric since GH doesn't
  // explicitly calculate those.
  using valencia_extra_temp_tags =
      tmpl::list<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>>;
  using valencia_arg_tags = tmpl::list_difference<
      typename grmhd::ValenciaDivClean::TimeDerivativeTerms::argument_tags,
      tmpl::append<gh_temp_tags, valencia_extra_temp_tags>>;

  using trace_reversed_stress_result_tags =
      tmpl::list<Tags::TraceReversedStressEnergy, Tags::FourVelocityOneForm,
                 Tags::ComovingMagneticFieldOneForm>;
  using extra_temp_tags = tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>>;

  using temporary_tags = tmpl::remove<
      tmpl::remove_duplicates<tmpl::append<
          gh_temp_tags, valencia_temp_tags, valencia_extra_temp_tags,
          trace_reversed_stress_result_tags, extra_temp_tags>>,
      gr::Tags::SpatialMetric<DataVector, 3>>;
  using argument_tags = tmpl::remove<
      tmpl::remove<tmpl::append<gh_arg_tags,

                                valencia_arg_tags,

                                tmpl::list<::Tags::VariableFixer<
                                    ::VariableFixing::FixToAtmosphere<3>>>>,
                   gr::Tags::SpatialMetric<DataVector, 3>>,
      d_spatial_metric>;

  template <typename... Args>
  static evolution::dg::TimeDerivativeDecisions<3> apply(
      const gsl::not_null<Variables<dt_tags>*> dt_vars_ptr,
      const gsl::not_null<Variables<db::wrap_tags_in<
          ::Tags::Flux, typename ValenciaDivClean::System::flux_variables,
          tmpl::size_t<3>, Frame::Inertial>>*>
          fluxes_ptr,
      const gsl::not_null<Variables<temporary_tags>*> temps_ptr,
      const tnsr::iaa<DataVector, 3>& d_spacetime_metric,
      const tnsr::iaa<DataVector, 3>& d_pi,
      const tnsr::ijaa<DataVector, 3>& d_phi, const Args&... args) {
    using args_list = tmpl::push_back<
        db::wrap_tags_in<Tags::detail::TemporaryReference, argument_tags>,
        gr::Tags::SpatialMetric<DataVector, 3>, d_spatial_metric>;
    tuples::tagged_tuple_from_typelist<args_list> arguments{
        args..., typename gr::Tags::SpatialMetric<DataVector, 3>::type{},
        typename d_spatial_metric::type{}};
    const size_t number_of_points = get<Tags::detail::TemporaryReference<
        gr::Tags::SpacetimeMetric<DataVector, 3>>>(arguments)[0]
                                        .size();
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        make_const_view(
            make_not_null(
                &std::as_const(
                     get<gr::Tags::SpatialMetric<DataVector, 3>>(arguments))
                     .get(i, j)),
            get<Tags::detail::TemporaryReference<
                gr::Tags::SpacetimeMetric<DataVector, 3>>>(arguments)
                .get(i + 1, j + 1),
            0, number_of_points);
      }
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = j; k < 3; ++k) {
          make_const_view(
              make_not_null(&std::as_const(get<d_spatial_metric>(arguments))
                                 .get(i, j, k)),
              get<Tags::detail::TemporaryReference<
                  gh::Tags::Phi<DataVector, 3>>>(arguments)
                  .get(i, j + 1, k + 1),
              0, number_of_points);
        }
      }
    }

    return detail::TimeDerivativeTermsImpl::apply(dt_vars_ptr, fluxes_ptr,
                                                  temps_ptr, d_spacetime_metric,
                                                  d_pi, d_phi, arguments);
  }
};
}  // namespace grmhd::GhValenciaDivClean

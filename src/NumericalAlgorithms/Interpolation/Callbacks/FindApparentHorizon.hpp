// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "ApparentHorizons/FastFlow.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"

/// \cond
namespace StrahlkorperTags {
template <typename Frame>
struct Strahlkorper;
}  // namespace StrahlkorperTags
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
template <class Metavariables, typename InterpolationTargetTag>
struct InterpolationTarget;
namespace Tags {
template <typename Metavariables>
struct TemporalIds;
}  // namespace Tags
}  // namespace intrp
template <typename Frame>
class Strahlkorper;
/// \endcond

namespace intrp {
namespace callbacks {

/// \brief post interpolation callback (see InterpolationTarget)
/// that does a FastFlow iteration and triggers another one until convergence.
///
/// Assumes that InterpolationTargetTag contains an additional
/// struct called `post_horizon_find_callback`, which has a function
///
/// \snippet ApparentHorizons/Test_ApparentHorizonFinder.cpp post_horizon_find_callback_example
///
/// that is called if the FastFlow iteration has converged, and an additional
/// struct called `horizon_find_failure_callback`, which has a function
///
/// \snippet ApparentHorizons/Test_ApparentHorizonFinder.cpp horizon_find_failure_callback_example
///
/// that is called if the FastFlow iteration or the interpolation has
/// failed.
///
/// Uses:
/// - Metavariables:
///   - `temporal_id`
/// - DataBox:
///   - `logging::Tags::Verbosity<InterpolationTargetTag>`
///   - `::gr::Tags::InverseSpatialMetric<3,Frame>`
///   - `::gr::Tags::ExtrinsicCurvature<3,Frame>`
///   - `::gr::Tags::SpatialChristoffelSecondKind<3,Frame>`
///   - `::ah::Tags::FastFlow`
///   - `StrahlkorperTags::Strahlkorper<Frame>`
///
/// Modifies:
/// - DataBox:
///   - `::ah::Tags::FastFlow`
///   - `StrahlkorperTags::Strahlkorper<Frame>`
///
/// This is an InterpolationTargetTag::post_interpolation_callback;
/// see InterpolationTarget for a description of InterpolationTargetTag.
///
/// ### Output
///
/// Optionally, a single line of output is printed to stdout either on each
/// iteration (if verbosity > Verbosity::Quiet) or on convergence
/// (if verbosity > Verbosity::Silent).  The output consists of the
/// following labels, and values associated with each label:
///  - its     = current iteration number.
///  - R       = min and max of residual over all prolonged grid points.
///  - |R|      = L2 norm of residual, counting only L modes solved for.
///  - |R_mesh| = L2 norm of residual over prolonged grid points.
///  - r        = min and max radius of trial horizon surface.
///
/// #### Difference between |R| and |R_mesh|:
///  The horizon is represented in a \f$Y_{lm}\f$ expansion up
///  to \f$l=l_{\mathrm{surface}}\f$;
///  the residual |R| represents the failure of that surface to satisfy
///  the apparent horizon equation.
///
///  However, at each iteration we also interpolate the horizon surface
///  to a higher resolution ("prolongation").  The prolonged surface
///  includes \f$Y_{lm}\f$ coefficents up to \f$l=l_{\mathrm{mesh}}\f$,
///  where \f$l_{\mathrm{mesh}} > l_{\mathrm{surface}}\f$.
///  The residual computed on this higher-resolution surface is |R_mesh|.
///
///  As iterations proceed, |R| should decrease until it reaches
///  numerical roundoff error, because we are varying all the \f$Y_{lm}\f$
///  coefficients up to  \f$l=l_{\mathrm{surface}}\f$
///  to minimize the residual.  However,
///  |R_mesh| should eventually stop decreasing as iterations proceed,
///  hitting a floor that represents the numerical truncation error of
///  the solution.
///
///  The convergence criterion looks at both |R| and |R_mesh|:  Once
///  |R| is small enough and |R_mesh| has stabilized, it is pointless
///  to keep iterating (even though one could iterate until |R| reaches
///  roundoff).
///
///  Problems with convergence of the apparent horizon finder can often
///  be diagnosed by looking at the behavior of |R| and |R_mesh| as a
///  function of iteration.
///
template <typename InterpolationTargetTag, typename Frame>
struct FindApparentHorizon {
  using observation_types = typename InterpolationTargetTag::
      post_horizon_find_callback::observation_types;
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static bool apply(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
      const TemporalId& temporal_id) noexcept {
    bool horizon_finder_failed = false;

    // Before doing anything else, deal with the possibility that some
    // of the points might be outside of the Domain.
    const auto& indices_of_invalid_pts =
        db::get<Tags::IndicesOfInvalidInterpPoints<TemporalId>>(*box);
    if (indices_of_invalid_pts.count(temporal_id) > 0 and
        not indices_of_invalid_pts.at(temporal_id).empty()) {
      InterpolationTargetTag::horizon_find_failure_callback::template apply<
          InterpolationTargetTag>(*box, *cache, temporal_id,
                                  FastFlow::Status::InterpolationFailure);
      horizon_finder_failed = true;
    }

    if (not horizon_finder_failed) {
      const auto& verbosity =
          db::get<logging::Tags::Verbosity<InterpolationTargetTag>>(*box);
      const auto& inv_g =
          db::get<::gr::Tags::InverseSpatialMetric<3, Frame>>(*box);
      const auto& ex_curv =
          db::get<::gr::Tags::ExtrinsicCurvature<3, Frame>>(*box);
      const auto& christoffel =
          db::get<::gr::Tags::SpatialChristoffelSecondKind<3, Frame>>(*box);

      std::pair<FastFlow::Status, FastFlow::IterInfo> status_and_info;

      // Do a FastFlow iteration.
      db::mutate<::ah::Tags::FastFlow, StrahlkorperTags::Strahlkorper<Frame>>(
          box, [&inv_g, &ex_curv, &christoffel, &status_and_info](
                   const gsl::not_null<::FastFlow*> fast_flow,
                   const gsl::not_null<::Strahlkorper<Frame>*>
                       strahlkorper) noexcept {
            status_and_info = fast_flow->template iterate_horizon_finder<Frame>(
                strahlkorper, inv_g, ex_curv, christoffel);
          });

      // Determine whether we have converged, whether we need another step,
      // or whether we have encountered an error.

      const auto& status = status_and_info.first;
      const auto& info = status_and_info.second;
      const auto has_converged = converged(status);

      if (verbosity > ::Verbosity::Quiet or
          (verbosity > ::Verbosity::Silent and has_converged)) {
        Parallel::printf(
            "%s: its=%d: %.1e<R<%.0e, |R|=%.1g, "
            "|R_grid|=%.1g, %.4g<r<%.4g\n",
            pretty_type::short_name<InterpolationTargetTag>(), info.iteration,
            info.min_residual, info.max_residual, info.residual_ylm,
            info.residual_mesh, info.r_min, info.r_max);
      }

      if (status == FastFlow::Status::SuccessfulIteration) {
        // Do another iteration of the same horizon search.
        const auto& temporal_ids =
            db::get<intrp::Tags::TemporalIds<TemporalId>>(*box);
        auto& interpolation_target = Parallel::get_parallel_component<
            intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>>(
            *cache);
        Parallel::simple_action<
            Actions::SendPointsToInterpolator<InterpolationTargetTag>>(
            interpolation_target, temporal_ids.front());
        // We return false because we don't want this iteration to clean
        // up the volume data, since we are using it for the next iteration
        // (i.e. the simple_action that we just called).
        return false;
      } else if (not has_converged) {
        InterpolationTargetTag::horizon_find_failure_callback::template apply<
            InterpolationTargetTag>(*box, *cache, temporal_id, status);
        horizon_finder_failed = true;
      }
    }

    if (not horizon_finder_failed) {
      // The interpolated variables
      // Tags::Variables<InterpolationTargetTag::vars_to_interpolate_to_target>
      // have been interpolated from the volume to the points on the
      // prolonged_strahlkorper, not to the points on the actual
      // strahlkorper.  So here we do a restriction of these
      // quantities onto the actual strahlkorper.

      // Type alias to make code more understandable.
      using vars_tags =
          typename InterpolationTargetTag::vars_to_interpolate_to_target;
      db::mutate_apply<tmpl::list<::Tags::Variables<vars_tags>>,
                       tmpl::list<StrahlkorperTags::Strahlkorper<Frame>,
                                  ::ah::Tags::FastFlow>>(
          [](const gsl::not_null<Variables<vars_tags>*> vars,
             const Strahlkorper<Frame>& strahlkorper,
             const FastFlow& fast_flow) noexcept {
            const size_t L_mesh = fast_flow.current_l_mesh(strahlkorper);
            const auto prolonged_strahlkorper =
                Strahlkorper<Frame>(L_mesh, L_mesh, strahlkorper);
            auto new_vars = ::Variables<vars_tags>(
                strahlkorper.ylm_spherepack().physical_size());

            tmpl::for_each<vars_tags>([&strahlkorper, &prolonged_strahlkorper,
                                       &vars, &new_vars](auto tag_v) noexcept {
              using tag = typename decltype(tag_v)::type;
              const auto& old_var = get<tag>(*vars);
              auto& new_var = get<tag>(new_vars);
              auto old_iter = old_var.begin();
              auto new_iter = new_var.begin();
              for (; old_iter != old_var.end() and new_iter != new_var.end();
                   ++old_iter, ++new_iter) {
                *new_iter = strahlkorper.ylm_spherepack().spec_to_phys(
                    prolonged_strahlkorper.ylm_spherepack().prolong_or_restrict(
                        prolonged_strahlkorper.ylm_spherepack().phys_to_spec(
                            *old_iter),
                        strahlkorper.ylm_spherepack()));
              }
            });
            *vars = std::move(new_vars);
          },
          box);

      InterpolationTargetTag::post_horizon_find_callback::apply(*box, *cache,
                                                                temporal_id);
    }

    // Prepare for finding horizon at a new time.  If the horizon
    // finder was successful, then do not change the strahlkorper,
    // which will be the initial guess for finding the horizon at the
    // next time.
    // Eventually we will do time-extrapolation to set the next guess.
    db::mutate<::ah::Tags::FastFlow, StrahlkorperTags::Strahlkorper<Frame>,
               ::ah::Tags::PreviousStrahlkorper<Frame>>(
        box, [&horizon_finder_failed](
                 const gsl::not_null<::FastFlow*> fast_flow,
                 const gsl::not_null<::Strahlkorper<Frame>*> strahlkorper,
                 const gsl::not_null<::Strahlkorper<Frame>*>
                     previous_strahlkorper) noexcept {
          if (horizon_finder_failed) {
            // Don't keep a partially-converged strahlkorper in the DataBox.
            // Reset to the previous value, even if that previous value
            // is the original initial guess.
            *strahlkorper = *previous_strahlkorper;
          } else {
            // Save a new previous_strahlkorper.
            *previous_strahlkorper = *strahlkorper;
          }
          fast_flow->reset_for_next_find();
        });
    // We return true because we are now done with all the volume data
    // at this temporal_id, so we want it cleaned up.
    return true;
  }
};
}  // namespace callbacks
}  // namespace intrp

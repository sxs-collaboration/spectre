// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "ApparentHorizons/FastFlow.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/Error.hpp"
#include "Informer/Verbosity.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
namespace ah {
namespace Tags {
struct FastFlow;
}  // namespace Tags
}  // namespace ah
namespace Tags {
struct Verbosity;
}  // namespace Tags
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
///```
///  static void apply(const DataBox<DbTags>&,
///                    const intrp::ConstGlobalCache<Metavariables>&,
///                    const Metavariables::temporal_id&) noexcept;
///```
/// that is called if the FastFlow iteration has converged.
///
/// Uses:
/// - Metavariables:
///   - `temporal_id`
///   - `domain_frame`
/// - DataBox:
///   - `::Tags::Verbosity`
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
template <typename InterpolationTargetTag>
struct FindApparentHorizon {
  template <typename DbTags, typename Metavariables>
  static bool apply(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const gsl::not_null<Parallel::ConstGlobalCache<Metavariables>*> cache,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    const auto& verbosity = db::get<::Tags::Verbosity>(*box);
    const auto& inv_g = db::get<::gr::Tags::InverseSpatialMetric<
        3, typename Metavariables::domain_frame>>(*box);
    const auto& ex_curv = db::get<::gr::Tags::ExtrinsicCurvature<
        3, typename Metavariables::domain_frame>>(*box);
    const auto& christoffel = db::get<::gr::Tags::SpatialChristoffelSecondKind<
        3, typename Metavariables::domain_frame>>(*box);

    std::pair<FastFlow::Status, FastFlow::IterInfo> status_and_info;

    // Do a FastFlow iteration.
    db::mutate<::ah::Tags::FastFlow, StrahlkorperTags::Strahlkorper<
                                     typename Metavariables::domain_frame>>(
        box, [&inv_g, &ex_curv, &christoffel, &status_and_info ](
                 const gsl::not_null<::FastFlow*> fast_flow,
                 const gsl::not_null<
                     ::Strahlkorper<typename Metavariables::domain_frame>*>
                     strahlkorper) noexcept {
          status_and_info = fast_flow->template iterate_horizon_finder<
              typename Metavariables::domain_frame>(strahlkorper, inv_g,
                                                    ex_curv, christoffel);
        });

    // Determine whether we have converged, whether we need another step,
    // or whether we have encountered an error.

    const auto& status = status_and_info.first;
    const auto& info = status_and_info.second;
    const auto has_converged = converged(status);

    if (verbosity > ::Verbosity::Quiet or
        (verbosity > ::Verbosity::Silent and has_converged)) {
      //  The things printed out here are:
      //  its     = current iteration number.
      //  R       = min and max of residual over all prolonged grid points.
      // |R|      = L2 norm of residual, counting only L modes solved for.
      // |R_mesh| = L2 norm of residual over prolonged grid points.
      // r        = min and max radius of trial horizon surface.
      //
      // Difference between |R| and |R_mesh|:
      //  The horizon is represented in a Y_lm expansion up to l=l_surface;
      //  the residual |R| represents the failure of that surface to satisfy
      //  the apparent horizon equation.
      //
      //  However, at each iteration we also interpolate the horizon surface
      //  to a higher resolution ("prolongation").  The prolonged surface
      //  includes Ylm coefficents up to l=l_mesh, where l_mesh > l_surface.
      //  The residual computed on this higher-resolution surface is |R_mesh|.
      //
      //  As iterations proceed, |R| should decrease until it reaches numerical
      //  roundoff error, because we are varying all the Y_lm coefficients up to
      //  l=l_surface to minimize the residual.  However, |R_mesh| should
      //  eventually stop decreasing as iterations proceed, hitting a
      //  floor that represents the numerical truncation error of the solution.
      //
      //  The convergence criterion looks at both |R| and |R_mesh|:  Once
      //  |R| is small enough and |R_mesh| has stabilized, it is pointless
      //  to keep iterating (even though one could iterate until |R| reaches
      //  roundoff).
      //
      //  Problems with convergence of the apparent horizon finder can often
      //  be diagnosed by looking at the behavior of |R| and |R_mesh| as a
      //  function of iteration.
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
          db::get<intrp::Tags::TemporalIds<Metavariables>>(*box);
      auto& interpolation_target = Parallel::get_parallel_component<
          intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>>(
          *cache);
      Parallel::simple_action<
          typename InterpolationTargetTag::compute_target_points>(
          interpolation_target, temporal_ids.front());
      // We return false because we don't want this iteration to clean
      // up the volume data, since we are using it for the next iteration
      // (i.e. the simple_action that we just called).
      return false;
    } else if (not has_converged) {
      ERROR("Apparent horizon finder "
            << pretty_type::short_name<InterpolationTargetTag>()
            << " failed, reason = " << status);
    }
    // If we get here, the horizon finder has converged.

    InterpolationTargetTag::post_horizon_find_callback::apply(*box, *cache,
                                                              temporal_id);

    // Prepare for finding horizon at a new time.
    // For now, the initial guess for the new
    // horizon is the result of the old one, so we don't need
    // to modify the strahlkorper.
    // Eventually we will hold more than one previous guess and
    // will do time-extrapolation to set the next guess.
    db::mutate<::ah::Tags::FastFlow>(
        box, [](const gsl::not_null<::FastFlow*> fast_flow) noexcept {
          fast_flow->reset_for_next_find();
        });
    // We return true because we are now done with all the volume data
    // at this temporal_id, so we want it cleaned up.
    return true;
  }
};
}  // namespace callbacks
}  // namespace intrp

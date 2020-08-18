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
#include "ErrorHandling/Error.hpp"
#include "Informer/Verbosity.hpp"
#include "Parallel/GlobalCache.hpp"
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
///                    const intrp::GlobalCache<Metavariables>&,
///                    const Metavariables::temporal_id&) noexcept;
///```
/// that is called if the FastFlow iteration has converged.
///
/// Uses:
/// - Metavariables:
///   - `temporal_id`
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
  using observation_types = typename InterpolationTargetTag::
      post_horizon_find_callback::observation_types;
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static bool apply(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
      const TemporalId& temporal_id) noexcept {
    // Before doing anything else, deal with the possibility that some
    // of the points might be outside of the Domain.
    const auto& indices_of_invalid_pts =
        db::get<Tags::IndicesOfInvalidInterpPoints<TemporalId>>(*box);
    if (indices_of_invalid_pts.count(temporal_id) > 0 and
        not indices_of_invalid_pts.at(temporal_id).empty()) {
      ERROR("FindApparentHorizon: Found points that are not in any block");
    }

    const auto& verbosity = db::get<::Tags::Verbosity>(*box);
    const auto& inv_g =
        db::get<::gr::Tags::InverseSpatialMetric<3, Frame::Inertial>>(*box);
    const auto& ex_curv =
        db::get<::gr::Tags::ExtrinsicCurvature<3, Frame::Inertial>>(*box);
    const auto& christoffel =
        db::get<::gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>>(
            *box);

    std::pair<FastFlow::Status, FastFlow::IterInfo> status_and_info;

    // Do a FastFlow iteration.
    db::mutate<::ah::Tags::FastFlow,
               StrahlkorperTags::Strahlkorper<Frame::Inertial>>(
        box, [&inv_g, &ex_curv, &christoffel, &status_and_info ](
                 const gsl::not_null<::FastFlow*> fast_flow,
                 const gsl::not_null<::Strahlkorper<Frame::Inertial>*>
                     strahlkorper) noexcept {
          status_and_info =
              fast_flow->template iterate_horizon_finder<Frame::Inertial>(
                  strahlkorper, inv_g, ex_curv, christoffel);
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
      ERROR("Apparent horizon finder "
            << pretty_type::short_name<InterpolationTargetTag>()
            << " failed, reason = " << status);
    }
    // If we get here, the horizon finder has converged.

    // The interpolated variables
    // ::Tags::Variables<InterpolationTargetTag::vars_to_interpolate_to_target>
    // have been interpolated from the volume to the points on the
    // prolonged_strahlkorper, not to the points on the actual
    // strahlkorper.  So here we do a restriction of these quantities onto
    // the actual strahlkorper.

    // Type alias to make code more understandable.
    using vars_tags =
        typename InterpolationTargetTag::vars_to_interpolate_to_target;
    db::mutate_apply<tmpl::list<::Tags::Variables<vars_tags>>,
                     tmpl::list<StrahlkorperTags::Strahlkorper<Frame::Inertial>,
                                ::ah::Tags::FastFlow>>(
        [](const gsl::not_null<Variables<vars_tags>*> vars,
           const Strahlkorper<Frame::Inertial>& strahlkorper,
           const FastFlow& fast_flow) noexcept {
          const size_t L_mesh = fast_flow.current_l_mesh(strahlkorper);
          const auto prolonged_strahlkorper =
              Strahlkorper<Frame::Inertial>(L_mesh, L_mesh, strahlkorper);
          auto new_vars = db::item_type<::Tags::Variables<vars_tags>>(
              strahlkorper.ylm_spherepack().physical_size());

          tmpl::for_each<vars_tags>([
            &strahlkorper, &prolonged_strahlkorper, &vars, &new_vars
          ](auto tag_v) noexcept {
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

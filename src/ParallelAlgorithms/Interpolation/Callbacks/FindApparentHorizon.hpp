// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <deque>
#include <utility>

#include "ApparentHorizons/FastFlow.hpp"
#include "ApparentHorizons/StrahlkorperCoordsInDifferentFrame.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

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

/// \brief post interpolation callback (see
/// intrp::protocols::PostInterpolationCallback) that does a FastFlow iteration
/// and triggers another one until convergence.
///
/// Assumes that InterpolationTargetTag contains an additional
/// type alias called `post_horizon_find_callbacks`, which is a list of
/// structs, each of which has a function
///
/// \snippet ApparentHorizons/Test_ApparentHorizonFinder.cpp
/// post_horizon_find_callback_example
///
/// that is called if the FastFlow iteration has converged.
/// InterpolationTargetTag also is assumed to contain an additional
/// struct called `horizon_find_failure_callback`, which has a function
///
/// \snippet ApparentHorizons/Test_ApparentHorizonFinder.cpp
/// horizon_find_failure_callback_example
///
/// that is called if the FastFlow iteration or the interpolation has
/// failed.
///
/// Uses:
/// - Metavariables:
///   - `temporal_id`
/// - DataBox:
///   - `logging::Tags::Verbosity<InterpolationTargetTag>`
///   - `::gr::Tags::InverseSpatialMetric<DataVector, 3, Frame>`
///   - `::gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame>`
///   - `::gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, Frame>`
///   - `::ah::Tags::FastFlow`
///   - `StrahlkorperTags::Strahlkorper<Frame>`
///
/// Modifies:
/// - DataBox:
///   - `::ah::Tags::FastFlow`
///   - `StrahlkorperTags::Strahlkorper<Frame>`
///
/// This is an InterpolationTargetTag::post_interpolation_callback;
/// see intrp::protocols::InterpolationTargetTag for details on
/// InterpolationTargetTag.
///
/// ### Output
///
/// Optionally, a single line of output is printed to stdout either on each
/// iteration (if verbosity > Verbosity::Quiet) or on convergence
/// (if verbosity > Verbosity::Silent).  The output consists of the
/// following labels, and values associated with each label:
///  - t        = time given by value of `temporal_id` argument to `apply`
///  - its      = current iteration number.
///  - R        = min and max of residual over all prolonged grid points.
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
///  includes \f$Y_{lm}\f$ coefficients up to \f$l=l_{\mathrm{mesh}}\f$,
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
struct FindApparentHorizon
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          typename InterpolationTargetTag::post_horizon_find_callbacks>;
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static bool apply(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
      const TemporalId& temporal_id) {
    bool horizon_finder_failed = false;

    if (get<::ah::Tags::FastFlow>(*box).current_iteration() == 0) {
      // If we get here, we are in a new apparent horizon search, as
      // opposed to a subsequent iteration of the same horizon search.
      //
      // So put new initial guess into StrahlkorperTags::Strahlkorper<Frame>.
      // We need to do this now, and not at the end of the previous horizon
      // search, because only now do we know the temporal_id of this horizon
      // search.
      db::mutate<StrahlkorperTags::Strahlkorper<Frame>,
                 ::ah::Tags::PreviousStrahlkorpers<Frame>>(
          [&temporal_id](
              const gsl::not_null<::Strahlkorper<Frame>*> strahlkorper,
              const gsl::not_null<
                  std::deque<std::pair<double, ::Strahlkorper<Frame>>>*>
                  previous_strahlkorpers) {
            // If we have zero previous_strahlkorpers, then the
            // initial guess is already in strahlkorper, so do
            // nothing.
            //
            // If we have one previous_strahlkorper, then we have had
            // a successful horizon find, and the initial guess for the
            // next horizon find is already in strahlkorper, so
            // again we do nothing.
            //
            // If we have 2 previous_strahlkorpers and the time of the second
            // one is a NaN, this means that the corresponding
            // previous_strahlkorper is the original initial guess, so
            // again we do nothing.
            //
            // If we have 2 or more valid previous_strahlkorpers, then
            // we set the initial guess by linear extrapolation in time
            // using the last 2 previous_strahlkorpers.
            if (previous_strahlkorpers->size() > 1 and
                not std::isnan((*previous_strahlkorpers)[1].first)) {
              const double new_time =
                  InterpolationTarget_detail::get_temporal_id_value(
                      temporal_id);
              const double dt_0 = (*previous_strahlkorpers)[0].first - new_time;
              const double dt_1 = (*previous_strahlkorpers)[1].first - new_time;
              const double fac_0 = dt_1 / (dt_1 - dt_0);
              const double fac_1 = 1.0 - fac_0;
              // Here we assume that
              // * Expansion center of all the Strahlkorpers are equal.
              // * Maximum L of all the Strahlkorpers are equal.
              // It is easy to relax the max L assumption once we start
              // adaptively changing the L of the strahlkorpers.
              strahlkorper->coefficients() =
                  fac_0 * (*previous_strahlkorpers)[0].second.coefficients() +
                  fac_1 * (*previous_strahlkorpers)[1].second.coefficients();
            }
          },
          box);
    }

    // Deal with the possibility that some of the points might be
    // outside of the Domain.
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
          db::get<::gr::Tags::InverseSpatialMetric<DataVector, 3, Frame>>(*box);
      const auto& ex_curv =
          db::get<::gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame>>(*box);
      const auto& christoffel = db::get<
          ::gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, Frame>>(*box);

      std::pair<FastFlow::Status, FastFlow::IterInfo> status_and_info;

      // Do a FastFlow iteration.
      db::mutate<::ah::Tags::FastFlow, StrahlkorperTags::Strahlkorper<Frame>>(
          [&inv_g, &ex_curv, &christoffel, &status_and_info](
              const gsl::not_null<::FastFlow*> fast_flow,
              const gsl::not_null<::Strahlkorper<Frame>*> strahlkorper) {
            status_and_info = fast_flow->template iterate_horizon_finder<Frame>(
                strahlkorper, inv_g, ex_curv, christoffel);
          },
          box);

      // Determine whether we have converged, whether we need another step,
      // or whether we have encountered an error.

      const auto& status = status_and_info.first;
      const auto& info = status_and_info.second;
      const auto has_converged = converged(status);

      if (verbosity > ::Verbosity::Quiet or
          (verbosity > ::Verbosity::Silent and has_converged)) {
        Parallel::printf(
            "%s: t=%.6g: its=%d: %.1e<R<%.0e, |R|=%.1g, "
            "|R_grid|=%.1g, %.4g<r<%.4g\n",
            pretty_type::name<InterpolationTargetTag>(),
            InterpolationTarget_detail::get_temporal_id_value(temporal_id),
            info.iteration, info.min_residual, info.max_residual,
            info.residual_ylm, info.residual_mesh, info.r_min, info.r_max);
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

    // If it failed, don't update any variables, just reset the Strahlkorper to
    // it's previous value
    if (horizon_finder_failed) {
      db::mutate<StrahlkorperTags::Strahlkorper<Frame>>(
          [](const gsl::not_null<::Strahlkorper<Frame>*> strahlkorper,
             const std::deque<std::pair<double, ::Strahlkorper<Frame>>>&
                 previous_strahlkorpers) {
            // Don't keep a partially-converged strahlkorper in the
            // DataBox.  Reset to either the original initial guess or
            // to the last-found Strahlkorper (whichever one happens
            // to be in previous_strahlkorpers).
            *strahlkorper = previous_strahlkorpers.front().second;
          },
          box, db::get<::ah::Tags::PreviousStrahlkorpers<Frame>>(*box));
    } else {
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
             const FastFlow& fast_flow) {
            const size_t L_mesh = fast_flow.current_l_mesh(strahlkorper);
            const auto prolonged_strahlkorper =
                Strahlkorper<Frame>(L_mesh, L_mesh, strahlkorper);
            auto new_vars = ::Variables<vars_tags>(
                strahlkorper.ylm_spherepack().physical_size());

            tmpl::for_each<vars_tags>([&strahlkorper, &prolonged_strahlkorper,
                                       &vars, &new_vars](auto tag_v) {
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

      // Compute Strahlkorper Cartesian coordinates in Inertial frame
      // if the current frame is not inertial.
      if constexpr (not std::is_same_v<Frame, ::Frame::Inertial>) {
        db::mutate_apply<
            tmpl::list<StrahlkorperTags::CartesianCoords<::Frame::Inertial>>,
            tmpl::list<StrahlkorperTags::Strahlkorper<Frame>,
                       domain::Tags::Domain<Metavariables::volume_dim>>>(
            [&cache, &temporal_id](
                const gsl::not_null<tnsr::I<DataVector, 3, ::Frame::Inertial>*>
                    inertial_strahlkorper_coords,
                const Strahlkorper<Frame>& strahlkorper,
                const Domain<Metavariables::volume_dim>& domain) {
              // Note that functions_of_time must already be up to
              // date at temporal_id because they were used in the AH
              // search above.
              const auto& functions_of_time =
                  get<domain::Tags::FunctionsOfTime>(*cache);
              strahlkorper_coords_in_different_frame(
                  inertial_strahlkorper_coords, strahlkorper, domain,
                  functions_of_time,
                  InterpolationTarget_detail::get_temporal_id_value(
                      temporal_id));
            },
            box);
      }

      // Update the previous strahlkorpers. We do this before the callbacks
      // in case any of the callbacks need the previous strahlkorpers with the
      // current strahlkorper already in it.
      db::mutate<StrahlkorperTags::Strahlkorper<Frame>,
                 ::ah::Tags::PreviousStrahlkorpers<Frame>>(
          [&temporal_id](
              const gsl::not_null<::Strahlkorper<Frame>*> strahlkorper,
              const gsl::not_null<
                  std::deque<std::pair<double, ::Strahlkorper<Frame>>>*>
                  previous_strahlkorpers) {
            // This is the number of previous strahlkorpers that we
            // keep around.
            const size_t num_previous_strahlkorpers = 3;

            // Save a new previous_strahlkorper.
            previous_strahlkorpers->emplace_front(
                InterpolationTarget_detail::get_temporal_id_value(temporal_id),
                *strahlkorper);

            // Remove old previous_strahlkorpers that are no longer relevant.
            while (previous_strahlkorpers->size() >
                   num_previous_strahlkorpers) {
              previous_strahlkorpers->pop_back();
            }
          },
          box);

      // Finally call callbacks
      tmpl::for_each<
          typename InterpolationTargetTag::post_horizon_find_callbacks>(
          [&box, &cache, &temporal_id](auto callback_v) {
            using callback = tmpl::type_from<decltype(callback_v)>;
            callback::apply(*box, *cache, temporal_id);
          });
    }

    // Prepare for finding horizon at a new time. Regardless of if we failed or
    // not, we reset fast flow.
    db::mutate<::ah::Tags::FastFlow>(
        [](const gsl::not_null<::FastFlow*> fast_flow) {
          fast_flow->reset_for_next_find();
        },
        box);

    // We return true because we are now done with all the volume data
    // at this temporal_id, so we want it cleaned up.
    return true;
  }
};
}  // namespace callbacks
}  // namespace intrp

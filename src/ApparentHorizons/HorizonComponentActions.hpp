// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "ApparentHorizons/FastFlow.hpp"
#include "ApparentHorizons/HorizonManagerComponentActions.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "Options/Options.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Time.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \ingroup SurfacesGroup
/// Holds objects used by horizon finders.
namespace ah {
template <class Metavariables>
struct DataInterpolator;

/// Tags for items held in the DataBox of an ah::Finder.
namespace Tags {

/// Keeps track of which points on the Strahlkorper have been filled
/// with interpolated data.
struct IndicesOfFilledInterpPoints : db::SimpleTag {
  static std::string name() noexcept { return "IndicesOfFilledInterpPoints"; }
  using type = std::unordered_set<size_t>;
};

/// Time steps on which to find horizons.
struct Timesteps : db::SimpleTag {
  using type = std::deque<Time>;
  static std::string name() noexcept { return "Timesteps"; }
};

struct FastFlow : db::SimpleTag {
  static std::string name() noexcept { return "FastFlow"; }
  using type = ::FastFlow;
};
template <typename Frame>
struct Strahlkorper : db::SimpleTag {
  static std::string name() noexcept { return "Strahlkorper"; }
  using type = ::Strahlkorper<Frame>;
};
template <size_t Dim, typename Frame>
struct Domain : db::SimpleTag {
  static std::string name() noexcept { return "Domain"; }
  using type = ::Domain<Dim, Frame>;
};
}  // namespace Tags

namespace Finder_detail {

// Returns the next set of points, in Frame coordinates, that data
// should be interpolated onto.
template <typename Frame>
const tnsr::I<DataVector, 3, Frame> next_interp_points(
    const Strahlkorper<Frame>& strahlkorper,
    const FastFlow& fast_flow) noexcept {
  // Construct strahlkorper that has a larger mesh than the number of
  // coefficients that are being minimized.  We interpolate to all points
  // on this prolonged_strahlkorper, not on the current strahlkorper.
  const size_t L_mesh = fast_flow.current_l_mesh(strahlkorper);
  const auto prolonged_strahlkorper =
      Strahlkorper<Frame>(L_mesh, L_mesh, strahlkorper);

  // Make a DataBox so we can get coords from prolonged_strahlkorper
  auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame>>,
      db::AddComputeTags<StrahlkorperTags::compute_items_tags<Frame>>>(
      std::move(prolonged_strahlkorper));

  return db::get<StrahlkorperTags::CartesianCoords<Frame>>(box);
}

// Sends to all the DataInterpolators a list of all the points that
// need to be interpolated onto.
// Also clears information about data that has been interpolated.
template <typename AhTag, typename DbTags, typename Metavariables>
void send_points_to_horizon_manager(
    db::DataBox<DbTags>& box, Parallel::ConstGlobalCache<Metavariables>& cache,
    const Time& timestep) noexcept {
  using frame = typename AhTag::frame;
  const auto& strahlkorper = db::get<Tags::Strahlkorper<frame>>(box);
  const auto& fast_flow = db::get<Tags::FastFlow>(box);
  const auto& domain = db::get<Tags::Domain<3, frame>>(box);
  auto& receiver_proxy =
      Parallel::get_parallel_component<ah::DataInterpolator<Metavariables>>(
          cache);
  auto coords = block_logical_coordinates(
      domain, next_interp_points<frame>(strahlkorper, fast_flow));

  db::mutate<Tags::IndicesOfFilledInterpPoints, vars_tags<frame>>(
      make_not_null(&box),
      [&coords](
          const gsl::not_null<db::item_type<Tags::IndicesOfFilledInterpPoints>*>
              indices_of_filled,
          const gsl::not_null<db::item_type<vars_tags<frame>>*>
              vars_dest) noexcept {
        indices_of_filled->clear();
        if (vars_dest->number_of_grid_points() != coords.size()) {
          *vars_dest = typename vars_tags<frame>::type(coords.size());
        }
      });
  Parallel::simple_action<
      Actions::DataInterpolator::ReceiveInterpolationPoints<AhTag>>(
      receiver_proxy, timestep, std::move(coords));
}

// Does a single iteration of the FastFlow algorithm.
// If another iteration is needed, sends the new points to the
// DataInterpolators (which starts the next iteration).
// If converged, check if another horizon should be
// found, and if so, send the new points to the DataInterpolators, starting
// the next iteration.
template <typename AhTag, typename DbTags, typename Metavariables>
void do_fastflow_iteration(
    db::DataBox<DbTags>& box,
    Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
  const auto& verbose = db::get<::Tags::Verbosity>(box);
  const auto& inv_g =
      db::get<::gr::Tags::InverseSpatialMetric<3, typename AhTag::frame>>(box);
  const auto& ex_curv =
      db::get<::gr::Tags::ExtrinsicCurvature<3, typename AhTag::frame>>(box);
  const auto& christ = db::get<
      ::gr::Tags::SpatialChristoffelSecondKind<3, typename AhTag::frame>>(box);

  if (verbose == ::Verbosity::Debug) {
    Parallel::printf(
        "### Node:%d  Proc:%d ###\n"
        "%s: FastFlowIteration\n\n",
        Parallel::my_node(), Parallel::my_proc(), AhTag::label());
  }

  std::pair<FastFlow::Status, FastFlow::IterInfo> status_and_info;

  db::mutate<Tags::FastFlow, Tags::Strahlkorper<typename AhTag::frame>>(
      make_not_null(&box),
      [&inv_g, &ex_curv, &christ, &status_and_info ](
          const gsl::not_null<::FastFlow*> fast_flow,
          const gsl::not_null<::Strahlkorper<typename AhTag::frame>*>
              strahlkorper) noexcept {
        status_and_info =
            fast_flow->template iterate_horizon_finder<typename AhTag::frame>(
                strahlkorper, inv_g, ex_curv, christ);
      });

  const auto& status = status_and_info.first;
  const auto& info = status_and_info.second;
  if (verbose > ::Verbosity::Quiet or
      (verbose > ::Verbosity::Silent and converged(status))) {
    Parallel::printf(
        "%s: its=%d: %.1e<R<%.0e, |R|=%.1g, "
        "|R_grid|=%.1g, %.4g<r<%.4g\n",
        AhTag::label(), info.iteration, info.min_residual, info.max_residual,
        info.residual_ylm, info.residual_mesh, info.r_min, info.r_max);
  }

  if (converged(status)) {
    const auto step = db::get<Tags::Timesteps>(box).front();
    if (verbose > ::Verbosity::Quiet) {
      Parallel::printf("FastFlowIteration: %s has converged, timestep %s\n",
                       AhTag::label(), step);
    }

    // Tell all the DataInterpolators about the convergence.
    // Eventually when there are Actions for observers, control
    // systems, etc. that depend on the horizon, they should be called
    // here.
    auto& horizon_manager_proxy =
        Parallel::get_parallel_component<ah::DataInterpolator<Metavariables>>(
            cache);
    Parallel::simple_action<
        Actions::DataInterpolator::HorizonHasConverged<AhTag>>(
        horizon_manager_proxy, step);

    // Call the hook.
    const auto& strahlkorper =
        db::get<Tags::Strahlkorper<typename AhTag::frame>>(box);
    AhTag::convergence_hook::apply(strahlkorper, step, cache);

    // Prepare for finding horizon at a new time.
    // For now, the initial guess for the new
    // horizon is the result of the old one.
    // Eventually ah::Finder will hold the previous horizons and
    // will do time-extrapolation to set the next guess.
    db::mutate<Tags::FastFlow, Tags::Timesteps>(
        make_not_null(&box), [](const gsl::not_null<::FastFlow*> fast_flow,
                                const gsl::not_null<
                                    db::item_type<Tags::Timesteps>*>
                                    steps) noexcept {
          fast_flow->reset_for_next_find();
          steps->pop_front();
        });

    const auto& timesteps = db::get<Tags::Timesteps>(box);
    if (not timesteps.empty()) {
      // There are still horizon searches to be done, so do the next one.
      // Note that we haven't yet implemented time extrapolation for a
      // new initial guess.  Currently the previous Strahlkorper
      // (which is in the DataBox) is used for the initial guess of the
      // next search.  To add the time extrapolation, we need to store
      // one or more extra Strahlkorper(s) for the previous iteration(s),
      // and do the extrapolation here.
      send_points_to_horizon_manager<AhTag, DbTags, Metavariables>(
          box, cache, timesteps.front());
    }
  } else if (status == FastFlow::Status::SuccessfulIteration) {
    // Do another iteration of the same horizon search.
    if (verbose == ::Verbosity::Debug) {
      Parallel::printf(
          "### Node:%d  Proc:%d ###\n"
          "%s: FastFlowIteration: Horizon next iter %d\n\n",
          Parallel::my_node(), Parallel::my_proc(), AhTag::label(),
          info.iteration);
    }
    const auto& timesteps = db::get<Tags::Timesteps>(box);
    send_points_to_horizon_manager<AhTag, DbTags, Metavariables>(
        box, cache, timesteps.front());
  } else {
    Parallel::printf("Finder %s failed, reason = %s\n", AhTag::label(), status);
    Parallel::abort("Finder failed");
  }
}
}  // namespace Finder_detail

namespace Actions {
namespace Finder {
template <typename AhTag>
struct Initialize {
  using return_tag_list =
      tmpl::list<Tags::IndicesOfFilledInterpPoints, ::Tags::Verbosity,
                 Tags::FastFlow, Tags::Strahlkorper<typename AhTag::frame>,
                 Tags::Timesteps, Tags::Domain<3, typename AhTag::frame>,
                 vars_tags<typename AhTag::frame>>;

  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    ::Verbosity verbosity, FastFlow&& fast_flow,
                    Strahlkorper<typename AhTag::frame>&& strahlkorper,
                    Domain<3, Frame::Inertial>&& domain) noexcept {
    return std::make_tuple(db::create<db::get_items<return_tag_list>>(
        Tags::IndicesOfFilledInterpPoints::type{}, verbosity,
        // clang-tidy: std::move of trivially copyable type
        std::move(fast_flow),  // NOLINT
        std::move(strahlkorper), Tags::Timesteps::type{}, std::move(domain),
        typename vars_tags<typename AhTag::frame>::type{}));
  }
};

/// Adds time steps on which this horizon should be found, and starts
/// a horizon search.  The time steps are removed once the horizon has
/// been found.
template <typename AhTag>
struct AddTimeSteps {
  using frame = typename AhTag::frame;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<DbTags, typename Tags::FastFlow>> =
                nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const std::vector<Time>& timesteps) noexcept {
    // If the horizon finder is currently running, it is holding a
    // non-empty list of timesteps.  Otherwise, we need to trigger a new
    // horizon search (done below).
    const bool start_horizon_finder = db::get<Tags::Timesteps>(box).empty();

    db::mutate<Tags::Timesteps>(
        make_not_null(&box), [&timesteps](const gsl::not_null<
                                          db::item_type<Tags::Timesteps>*>
                                              steps) noexcept {
          steps->insert(steps->end(), timesteps.begin(), timesteps.end());
        });

    // Trigger horizon finder if necessary.
    const auto& steps = db::get<Tags::Timesteps>(box);
    if (start_horizon_finder and not steps.empty()) {
      Finder_detail::send_points_to_horizon_manager<AhTag, DbTags,
                                                    Metavariables>(
          box, cache, steps.front());
    }
  }
};

/// Receives interpolated variables from the DataInterpolators.
/// When enough data has been received, it does a horizon-finding
/// iteration.
template <typename AhTag>
struct ReceiveInterpolatedVars {
  using frame = typename AhTag::frame;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<DbTags, typename Tags::FastFlow>> =
                nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const std::vector<typename vars_tags<frame>::type>& vars_src,
      const std::vector<std::vector<size_t>>& global_offsets) noexcept {
    db::mutate<Tags::IndicesOfFilledInterpPoints, vars_tags<frame>>(
        make_not_null(&box),
        [
          &vars_src, &global_offsets
        ](const gsl::not_null<db::item_type<Tags::IndicesOfFilledInterpPoints>*>
              indices_of_filled,
          const gsl::not_null<db::item_type<vars_tags<frame>>*>
              vars_dest) noexcept {
          const size_t npts_dest = vars_dest->number_of_grid_points();
          const size_t nvars = vars_dest->number_of_independent_components;
          for (size_t j = 0; j < global_offsets.size(); ++j) {
            const size_t npts_src = global_offsets[j].size();
            for (size_t i = 0; i < npts_src; ++i) {
              // If a point is on the boundary of two (or more)
              // elements, it is possible that we have received data
              // for this point from more than one DataInterpolator.
              // This will rarely occur, but it does occur, e.g. when
              // an initial-guess point is exactly on some symmetry
              // boundary (such as the x-y plane) and this symmetry
              // boundary is exactly the boundary between two
              // elements.  If this happens, we accept the first
              // duplicated point here, and we ignore subsequent
              // duplicated points.  The points are easy to keep track
              // of because global_offsets uniquely identifies the
              // points.
              if (indices_of_filled->insert(global_offsets[j][i]).second) {
                for (size_t v = 0; v < nvars; ++v) {
                  // clang-tidy: no pointer arithmetic
                  vars_dest->data()[global_offsets[j][i] +   // NOLINT
                                    v * npts_dest] =         // NOLINT
                      vars_src[j].data()[i + v * npts_src];  // NOLINT
                }
              }
            }
          }
        });

    const auto& verbose = db::get<::Tags::Verbosity>(box);
    if (verbose == ::Verbosity::Debug) {
      Parallel::printf(
          "### Node:%d  Proc:%d ###\n"
          "{%s}: ReceiveInterpolatedVars:filled %d of %d points\n\n",
          Parallel::my_node(), Parallel::my_proc(), AhTag::label(),
          db::get<Tags::IndicesOfFilledInterpPoints>(box).size(),
          db::get<vars_tags<frame>>(box).number_of_grid_points());
    }

    if (db::get<Tags::IndicesOfFilledInterpPoints>(box).size() ==
        db::get<vars_tags<frame>>(box).number_of_grid_points()) {
      Finder_detail::do_fastflow_iteration<AhTag, DbTags, Metavariables>(box,
                                                                         cache);
    }
  }
};
}  // namespace Finder
}  // namespace Actions
}  // namespace ah

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/Actions/GetItemFromDistributedObject.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Component.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeVarsToInterpolateToTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CleanupRoutine.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::Events {
/*!
 * \brief Starts a horizon find by sending volume data (`ah::source_vars`) to
 * the horizon finder component (`ah::Component`) using the
 * `ah::FindApparentHorizon` simple action.
 *
 * \details Only sends data if this Element is in the
 * `ah::Tags::BlocksForHorizonFind` tag.
 *
 */
template <typename HorizonMetavars>
class FindApparentHorizon
    : public SPECTRE_CHARM_DERIVED(
          SINGLE_ARG(FindApparentHorizon<HorizonMetavars>), Event) {
 public:
  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FindApparentHorizon);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help = "Start a horizon find.";

  static std::string name() { return pretty_type::name<HorizonMetavars>(); }

  FindApparentHorizon() = default;

  /// This constructor is not available through options
  explicit FindApparentHorizon(std::optional<std::string> dependency)
      : dependency_(std::move(dependency)) {}

  using compute_tags_for_observation_box =
      typename HorizonMetavars::compute_tags_on_element;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::append<
      tmpl::list<typename HorizonMetavars::time_tag,
                 ::Events::Tags::ObserverMesh<3>, domain::Tags::Element<3>>,
      ah::source_vars<3>>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(const LinkedMessageId<double>& time, const Mesh<3>& mesh,
                  const Element<3>& element,
                  const tnsr::aa<DataVector, 3>& spacetime_metric,
                  const tnsr::aa<DataVector, 3>& pi,
                  const tnsr::iaa<DataVector, 3>& phi,
                  const tnsr::ijaa<DataVector, 3>& deriv_phi,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<3>& element_id,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    const auto& blocks_to_interpolate =
        Parallel::get<ah::Tags::BlocksForHorizonFind>(cache);
    ASSERT(blocks_to_interpolate.contains(name_),
           "Blocks to interpolate doesn't contain target " << name_);
    const auto& blocks_to_interpolate_for_this_target =
        blocks_to_interpolate.at(name_);
    const auto& domain = Parallel::get<domain::Tags::Domain<3>>(cache);
    const auto& blocks = domain.blocks();
    const auto& block = blocks[element_id.block_id()];
    const auto& block_name = block.name();

    // Only send data if this target needs this blocks data
    if (not blocks_to_interpolate_for_this_target.contains(block_name)) {
      return;
    }

    // Send volume data ONLY if this element intersected with the previous
    // horizon or if it's a neighbor of an intersecting element.
    // - WARNING: the algorithm WILL deadlock if the horizon has moved outside
    //   of the elements that send data here. So we can't be too greedy with the
    //   elements that send data. We may have to send data from corner neighbors
    //   as well if deadlocks occur.
    // - WARNING: we don't currently wait for the `PreviousSurface` to be
    //   updated by the last horizon find. This could be done by storing the
    //   time of the last horizon find in the element DataBox and comparing it
    //   to the time that's stored in `PreviousSurface`. However, this added
    //   dependency would possibly introduce waiting. So far, I (NV) haven't
    //   found that waiting for the update is necessary, meaning that whichever
    //   intersecting element IDs are stored are close enough to the latest
    //   state so that they cover the new apparent horizon (no deadlocks have
    //   occurred in testing).
    //   Alternatives: if we find that we need the up-to-date intersecting
    //   elements we could just send the data without waiting if the
    //   intersecting element IDs are not up-to-date, which just risks sending
    //   more data than necessary (and therefore doing more work, potentially
    //   undoing this performance optimization if it happens a lot).
    const auto& locked_previous_surface =
        Parallel::get<ah::Tags::PreviousSurface<HorizonMetavars>>(cache);
    {
      // Scope to lock and unlock the read lock
      locked_previous_surface.lock.read_lock();
      const CleanupRoutine unlock_read_lock = [&locked_previous_surface]() {
        locked_previous_surface.lock.read_unlock();
      };
      // Only skip elements if we have a previous surface
      if (locked_previous_surface.surface.has_value()) {
        const auto& previous_surface = locked_previous_surface.surface.value();
        // If we already found a horizon AFTER the current time, skip sending
        // anything (the data won't be needed anymore). Unlikely to occur.
        if (previous_surface.time.id >= time.id) {
          return;
        }
        // Check if this element or any of its neighbors overlaps an element
        // that intersected the horizon in the last find. Only send volume data
        // if so.
        const bool send_volume_data = alg::any_of(
            previous_surface.intersecting_element_ids,
            [&element_id,
             &element](const ElementId<3>& intersecting_element_id) {
              return overlapping(element_id, intersecting_element_id) or
                     alg::any_of(
                         element.neighbors(),
                         [&intersecting_element_id](
                             const std::pair<Direction<3>, Neighbors<3>>&
                                 direction_and_neighbors) {
                           return alg::any_of(
                               direction_and_neighbors.second.ids(),
                               [&intersecting_element_id](
                                   const ElementId<3>& neighbor_id) {
                                 return overlapping(neighbor_id,
                                                    intersecting_element_id);
                               });
                         });
            });
        if (not send_volume_data) {
          return;
        }
      }  // if previous_surface.has_value()
    }    // Unlock read lock here

    // Make Variables<ah::vars_to_interpolate_to_target> and
    // fill it by calling compute_vars_to_interpolate_to_target().
    // Then pass this variables to the FindApparentHorizon simple action.
    using horizon_frame = typename HorizonMetavars::frame;
    Variables<ah::vars_to_interpolate_to_target<3, horizon_frame>>
        vars_to_interpolate_to_target{mesh.number_of_grid_points()};
    if (block.is_time_dependent()) {
      if constexpr (Parallel::is_in_global_cache<
                        Metavariables, domain::Tags::FunctionsOfTime>) {
        const auto& functions_of_time =
            Parallel::get<domain::Tags::FunctionsOfTime>(cache);
        ah::compute_vars_to_interpolate_to_target(
            make_not_null(&vars_to_interpolate_to_target), spacetime_metric, pi,
            phi, deriv_phi, time, domain, mesh, element_id, functions_of_time);
      } else {
        ERROR(
            "Block is time-dependent but FunctionsOfTime are not available "
            "in the global cache.");
      }
    } else {
      ah::compute_vars_to_interpolate_to_target(
          make_not_null(&vars_to_interpolate_to_target), spacetime_metric, pi,
          phi, deriv_phi, time, domain, mesh, element_id, {});
    }

    auto& horizon_finder_proxy = Parallel::get_parallel_component<
        ah::Component<Metavariables, HorizonMetavars>>(cache);

    Parallel::simple_action<ah::FindApparentHorizon<HorizonMetavars>>(
        horizon_finder_proxy, time, element_id, mesh,
        std::move(vars_to_interpolate_to_target), dependency_);
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*component*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  void pup(PUP::er& p) override {
    Event::pup(p);
    p | dependency_;
  }

 private:
  std::optional<std::string> dependency_;
  // Evaluate the static name() function only once to avoid repeated allocations
  std::string name_ = name();
};

#if defined(SPECTRE_USE_CHARM)
/// \cond
// NOLINTBEGIN
template <typename HorizonMetavars>
PUP::able::PUP_ID FindApparentHorizon<HorizonMetavars>::my_PUP_ID = 0;
// NOLINTEND
/// \endcond
#endif  // SPECTRE_USE_CHARM
}  // namespace ah::Events

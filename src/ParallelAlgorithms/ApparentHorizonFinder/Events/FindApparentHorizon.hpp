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
class FindApparentHorizon : public Event {
 public:
  /// \cond
  explicit FindApparentHorizon(CkMigrateMessage* /*unused*/) {}
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
  using argument_tags =
      tmpl::append<tmpl::list<typename HorizonMetavars::time_tag,
                              ::Events::Tags::ObserverMesh<3>>,
                   ah::source_vars<3>>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(const LinkedMessageId<double>& time, const Mesh<3>& mesh,
                  const tnsr::aa<DataVector, 3>& spacetime_metric,
                  const tnsr::aa<DataVector, 3>& pi,
                  const tnsr::iaa<DataVector, 3>& phi,
                  const tnsr::ijaa<DataVector, 3>& deriv_phi,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<3>& array_index,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    const auto& blocks_to_interpolate =
        Parallel::get<ah::Tags::BlocksForHorizonFind>(cache);
    ASSERT(blocks_to_interpolate.contains(name()),
           "Blocks to interpolate doesn't contain target " << name());
    const auto& blocks_to_interpolate_for_this_target =
        blocks_to_interpolate.at(name());
    const auto& domain = Parallel::get<domain::Tags::Domain<3>>(cache);
    const auto& blocks = domain.blocks();
    const auto& block = blocks[array_index.block_id()];
    const auto& block_name = block.name();

    // Only send data if this target needs this blocks data
    if (not blocks_to_interpolate_for_this_target.contains(block_name)) {
      return;
    }

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
            phi, deriv_phi, time, domain, mesh, array_index, functions_of_time);
      } else {
        ERROR(
            "Block is time-dependent but FunctionsOfTime are not available "
            "in the global cache.");
      }
    } else {
      ah::compute_vars_to_interpolate_to_target(
          make_not_null(&vars_to_interpolate_to_target), spacetime_metric, pi,
          phi, deriv_phi, time, domain, mesh, array_index, {});
    }

    auto& horizon_finder_proxy = Parallel::get_parallel_component<
        ah::Component<Metavariables, HorizonMetavars>>(cache);

    Parallel::simple_action<ah::FindApparentHorizon<HorizonMetavars>>(
        horizon_finder_proxy, time, array_index, mesh,
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
};

/// \cond
// NOLINTBEGIN
template <typename HorizonMetavars>
PUP::able::PUP_ID FindApparentHorizon<HorizonMetavars>::my_PUP_ID = 0;
// NOLINTEND
/// \endcond
}  // namespace ah::Events

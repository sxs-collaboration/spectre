// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <linux/limits.h>
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
#include "ParallelAlgorithms/Actions/GetItemFromDistributedObject.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Component.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ReceiveVolumeData.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::Events {
/*!
 * \brief Starts a horizon find by sending volume data to the horizon finder
 * component.
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
        Parallel::get<ah::Tags::BlocksForInterpolation2>(cache);
    ASSERT(blocks_to_interpolate.contains(name()),
           "Blocks to interpolate doesn't contain target " << name());
    const auto& blocks_to_interpolate_for_this_target =
        blocks_to_interpolate.at(name());
    const auto& blocks = Parallel::get<domain::Tags::Domain<3>>(cache).blocks();
    const auto& block_name = blocks[array_index.block_id()].name();

    // Only send data if this target needs this blocks data
    if (not blocks_to_interpolate_for_this_target.contains(block_name)) {
      return;
    }

    // Put everything into a single variables
    Variables<ah::source_vars<3>> source_vars{mesh.number_of_grid_points()};
    get<gr::Tags::SpacetimeMetric<DataVector, 3>>(source_vars) =
        spacetime_metric;
    get<gh::Tags::Pi<DataVector, 3>>(source_vars) = pi;
    get<gh::Tags::Phi<DataVector, 3>>(source_vars) = phi;
    get<::Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                      Frame::Inertial>>(source_vars) = deriv_phi;

    auto& horizon_finder_proxy = Parallel::get_parallel_component<
        ah::Component<Metavariables, HorizonMetavars>>(cache);

    Parallel::simple_action<ah::ReceiveVolumeData<HorizonMetavars>>(
        horizon_finder_proxy, time, array_index, mesh, std::move(source_vars),
        dependency_);
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
template <typename HorizonMetavars>
PUP::able::PUP_ID FindApparentHorizon<HorizonMetavars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace ah::Events

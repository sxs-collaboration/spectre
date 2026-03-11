// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionSingleton.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::bbh::Events {
/*!
 * \brief Computes per-element Linf norms of the generalized-harmonic
 * constraints, latching BBH completion criteria when thresholds are exceeded.
 *
 * \details This event is intended to run once common-horizon finding is
 * active (typically behind a separation-based trigger). It monitors the gauge
 * and three-index constraints, using a reduction to determine if their Linf
 * norms exceed completion thresholds.
 */
class CheckConstraintThresholds : public Event {
  using ReductionData = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<double, funcl::Max<>>,
      Parallel::ReductionDatum<double, funcl::Max<>>>;

 public:
  /// \cond
  explicit CheckConstraintThresholds(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CheckConstraintThresholds);  // NOLINT
  /// \endcond

  using compute_tags_for_observation_box =
      tmpl::list<gh::Tags::GaugeConstraintCompute<3, Frame::Inertial>,
                 gh::Tags::ThreeIndexConstraintCompute<3, Frame::Inertial>>;
  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Checks local Linf norms of constraints against BBH completion "
      "thresholds and forwards reduced maxima to the BBH completion singleton "
      "reduction callback.";
  static std::string name() { return "BbhCheckConstraintThresholds"; }

  CheckConstraintThresholds() = default;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<
      ::Tags::Time, gh::Tags::GaugeConstraint<DataVector, 3, Frame::Inertial>,
      gh::Tags::ThreeIndexConstraint<DataVector, 3, Frame::Inertial>>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(
      const double time,
      const tnsr::a<DataVector, 3, Frame::Inertial>& gauge_constraint,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& three_index_constraint,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const Component* const /*component*/,
      const ObservationValue& /*observation_value*/) const {
    const double local_gauge_linf = local_linf_norm(gauge_constraint);
    const double local_three_index_linf =
        local_linf_norm(three_index_constraint);
    if constexpr (Parallel::is_dg_element_collection_v<Component>) {
      ERROR(
          "BbhCheckConstraintThresholds currently requires array components "
          "(not DgElementCollection).");
    } else {
      const auto& self_proxy =
          Parallel::get_parallel_component<Component>(cache)[array_index];
      auto& reduction_target_proxy = Parallel::get_parallel_component<
          gh::bbh::CompletionSingleton<Metavariables>>(cache);
      Parallel::contribute_to_reduction<
          gh::bbh::Actions::ProcessConstraintMaxima>(
          ReductionData{time, local_gauge_linf, local_three_index_linf},
          self_proxy, reduction_target_proxy);
    }
  }

  using is_ready_argument_tags = tmpl::list<>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

 private:
  template <typename TensorType>
  static double local_linf_norm(const TensorType& tensor) {
    double result = 0.0;
    for (size_t storage_index = 0; storage_index < tensor.size();
         ++storage_index) {
      const auto& component = tensor[storage_index];
      result = std::max(result, max(abs(component)));
    }
    return result;
  }
};
}  // namespace gh::bbh::Events

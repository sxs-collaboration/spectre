// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <map>
#include <pup.h>
#include <tuple>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Options/String.hpp"
#include "Parallel/DistributedObject.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace Events {

namespace detail {

struct map_add {
  std::map<std::string, size_t> operator()(
      std::map<std::string, size_t> map_1,
      const std::map<std::string, size_t>& map_2) {
    for (const auto& [key, value] : map_2) {
      map_1.at(key) += value;
    }
    return map_1;
  }
};

using ReductionType = Parallel::ReductionData<
    // Time
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    // Map of total mem usage in MB per item in DataBoxes
    Parallel::ReductionDatum<std::map<std::string, size_t>, map_add>>;

template <typename ContributingComponent>
struct ReduceDataBoxSize {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const double time,
                    const std::map<std::string, size_t>& item_sizes) {
    auto& observer_writer_proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);
    const std::string subfile_name =
        "/DataBoxSizeInMb/" + pretty_type::name<ContributingComponent>();
    std::vector<std::string> legend;
    legend.reserve(item_sizes.size() + 1);
    legend.emplace_back("Time");
    std::vector<double> columns;
    columns.reserve(item_sizes.size() + 1);
    columns.emplace_back(time);
    const double scaling = 1.0 / 1048576.0;  // so size is in MB
    for (const auto& [name, size] : item_sizes) {
      legend.emplace_back(name);
      columns.emplace_back(scaling * static_cast<double>(size));
    }
    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        // Node 0 is always the writer
        observer_writer_proxy[0], subfile_name, legend,
        std::make_tuple(columns));
  }
};

struct ContributeDataBoxSize {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const double time) {
    const auto& my_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index];
    auto& target_proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);
    const auto item_sizes = box.size_of_items();
    if constexpr (Parallel::is_singleton_v<ParallelComponent>) {
      Parallel::simple_action<ReduceDataBoxSize<ParallelComponent>>(
          target_proxy[0], time, item_sizes);
    } else {
      Parallel::contribute_to_reduction<ReduceDataBoxSize<ParallelComponent>>(
          ReductionType{time, item_sizes}, my_proxy, target_proxy[0]);
    }
  }
};
}  // namespace detail

/// \brief Event that will collect the size in MBs used by each DataBox item on
/// each parallel component.
///
/// \details The data will be written to disk in the reductions file under the
/// `/DataBoxSizeInMb/` group. The name of each file is the `pretty_type::name`
/// of each parallel component.  There will be a column for each item in the
/// DataBox that is not a subitem or reference item.
class ObserveDataBox : public Event {
 public:
  /// \cond
  explicit ObserveDataBox(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveDataBox);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Observe size (in MB) of each item (except reference items) in each "
      "DataBox"};

  ObserveDataBox() = default;

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::DataBox>;

  template <typename DataBoxType, typename ArrayIndex,
            typename ParallelComponent, typename Metavariables>
  void operator()(const DataBoxType& box,
                  Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& array_index,
                  const ParallelComponent* /*meta*/,
                  const ObservationValue& observation_value) const;

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;
};

template <typename DataBoxType, typename ArrayIndex, typename ParallelComponent,
          typename Metavariables>
void ObserveDataBox::operator()(
    const DataBoxType& /*box*/, Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& array_index, const ParallelComponent* const /*meta*/,
    const ObservationValue& observation_value) const {
  if (is_zeroth_element(array_index)) {
    using component_list =
        tmpl::push_back<typename Metavariables::component_list>;
    tmpl::for_each<component_list>([&observation_value,
                                    &cache](auto component_v) {
      using component = tmpl::type_from<decltype(component_v)>;
      auto& target_proxy = Parallel::get_parallel_component<component>(cache);
      Parallel::simple_action<detail::ContributeDataBoxSize>(
          target_proxy, observation_value.value);
    });
  }
}
}  // namespace Events

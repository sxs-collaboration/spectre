// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ScriPlusInterpolationManager.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/Observer/WriteSimpleData.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace detail {
// Provide a nicer name for the output h5 files for some of the uglier
// combinations we need
template <typename Tag>
struct ScriOutput {
  static std::string name() noexcept { return db::tag_name<Tag>(); }
};
template <typename Tag>
struct ScriOutput<Tags::ScriPlus<Tag>> {
  static std::string name() noexcept {
    return pretty_type::short_name<Tag>();
  }
};
template <>
struct ScriOutput<::Tags::Multiplies<
  Tags::Du<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>>,
  Tags::ScriPlusFactor<Tags::Psi4>>> {
  static std::string name() noexcept { return "Psi4"; }
};
}  // namespace detail

namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Checks the interpolation managers and if they are ready, performs the
 * interpolation and sends the data to file via
 * `observers::ThreadedActions::WriteSimpleData`.
 *
 * \details This uses the `ScriPlusInterpolationManager` to perform the
 * interpolations of all requested scri quantities (determined by
 * `scri_values_to_observe` in the metavariables), and write them to disk using
 * `observers::threadedActions::WriteSimpleData`.
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: `InterpolagionManager<ComplexDataVector, Tag>` for each `Tag` in
 * `Metavariables::scri_values_to_observe`
 */
template <typename ObserverWriterComponent>
struct ScriObserveInterpolated {
  using const_global_cache_tags = tmpl::list<Tags::ObservationLMax>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const size_t observation_l_max = db::get<Tags::ObservationLMax>(box);
    const size_t l_max = db::get<Tags::LMax>(box);
    std::vector<double> data_to_write(2 * square(observation_l_max + 1) + 1);
    std::vector<std::string> file_legend;
    file_legend.reserve(2 * square(observation_l_max + 1) + 1);
    file_legend.emplace_back("time");
    for (int i = 0; i <= static_cast<int>(observation_l_max); ++i) {
      for(int j = -i; j <= i; ++j) {
        file_legend.push_back(MakeString{} << "Real Y_" << i << "," << j);
        file_legend.push_back(MakeString{} << "Imag Y_" << i << "," << j);
      }
    }
    auto observer_proxy =
        Parallel::get_parallel_component<ObserverWriterComponent>(
            cache)[static_cast<size_t>(Parallel::my_node())];
    while (
        db::get<Tags::InterpolationManager<
            ComplexDataVector,
            tmpl::front<typename Metavariables::scri_values_to_observe>>>(box)
            .first_time_is_ready_to_interpolate()) {
      tmpl::for_each<typename Metavariables::scri_values_to_observe>([
        &box, &data_to_write, &observer_proxy, &file_legend, &observation_l_max,
        &l_max
      ](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        std::pair<double, ComplexDataVector> interpolation;
        db::mutate<Tags::InterpolationManager<ComplexDataVector, tag>>(
            make_not_null(&box),
            [&interpolation](
                const gsl::not_null<
                    ScriPlusInterpolationManager<ComplexDataVector, tag>*>
                    interpolation_manager) {
              interpolation =
                  interpolation_manager->interpolate_and_pop_first_time();
            });
        // swsh transform
        const auto to_transform =
            SpinWeighted<ComplexDataVector, tag::type::type::spin>{
                interpolation.second};
        const ComplexModalVector goldberg_modes =
            Spectral::Swsh::libsharp_to_goldberg_modes(
                Spectral::Swsh::swsh_transform(l_max, 1, to_transform), l_max)
                .data();

        data_to_write[0] = interpolation.first;
        for(size_t i = 0; i < square(observation_l_max + 1); ++i) {
          data_to_write[2 * i + 1] = real(goldberg_modes[i]);
          data_to_write[2 * i + 2] = imag(goldberg_modes[i]);
        }
        Parallel::threaded_action<observers::ThreadedActions::WriteSimpleData>(
            observer_proxy, file_legend, data_to_write,
            "/" + ::Cce::detail::ScriOutput<tag>::name());
      });
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Cce

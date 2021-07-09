// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Time/Tags.hpp"

namespace Cce {
namespace Actions {

namespace detail {
template <typename Tag>
struct get_interpolator_argument_tag {
  using type = Tag;
};

template <typename Tag>
struct get_interpolator_argument_tag<Tags::Du<Tag>> {
  using type = Tag;
};

template <typename Tag>
struct InsertIntoInterpolationManagerImpl {
  using return_tags =
      tmpl::list<Tags::InterpolationManager<ComplexDataVector, Tag>>;
  using argument_tags =
      tmpl::list<typename get_interpolator_argument_tag<Tag>::type,
                 Tags::InertialRetardedTime>;
  static void apply(
      const gsl::not_null<ScriPlusInterpolationManager<ComplexDataVector, Tag>*>
          interpolation_manager,
      const typename Tag::type& scri_data,
      const Scalar<DataVector>& inertial_retarded_time) noexcept {
    interpolation_manager->insert_data(get(inertial_retarded_time),
                                       get(scri_data).data());
  }
};

template <typename LhsTag, typename RhsTag>
struct InsertIntoInterpolationManagerImpl<::Tags::Multiplies<LhsTag, RhsTag>> {
  using return_tags = tmpl::list<Tags::InterpolationManager<
      ComplexDataVector, ::Tags::Multiplies<LhsTag, RhsTag>>>;
  using argument_tags =
      tmpl::list<typename get_interpolator_argument_tag<LhsTag>::type, RhsTag,
                 Tags::InertialRetardedTime>;
  static void apply(const gsl::not_null<ScriPlusInterpolationManager<
                        ComplexDataVector, ::Tags::Multiplies<LhsTag, RhsTag>>*>
                        interpolation_manager,
                    const typename LhsTag::type& lhs_data,
                    const typename RhsTag::type& rhs_data,
                    const Scalar<DataVector>& inertial_retarded_time) noexcept {
    interpolation_manager->insert_data(get(inertial_retarded_time),
                                       get(lhs_data).data(),
                                       get(rhs_data).data());
  }
};

template <typename Tag, typename ParallelComponent, typename Metavariables>
void output_impl(const size_t observation_l_max, const size_t l_max,
                 const TimeStepId& time_id, const typename Tag::type& tag_data,
                 Parallel::GlobalCache<Metavariables>& cache) noexcept {
  std::vector<double> data_to_write(2 * square(observation_l_max + 1) + 1);
  std::vector<std::string> file_legend;
  file_legend.reserve(2 * square(observation_l_max + 1) + 1);
  file_legend.emplace_back("time");
  for (int l = 0; l <= static_cast<int>(observation_l_max); ++l) {
    for (int m = -l; m <= l; ++m) {
      file_legend.push_back(MakeString{} << "Real Y_" << l << "," << m);
      file_legend.push_back(MakeString{} << "Imag Y_" << l << "," << m);
    }
  }
  auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
  auto observer_proxy = Parallel::get_parallel_component<
      observers::ObserverWriter<Metavariables>>(cache)[static_cast<size_t>(
      Parallel::my_node(*Parallel::local(my_proxy)))];
  // swsh transform
  const ComplexModalVector goldberg_modes =
      Spectral::Swsh::libsharp_to_goldberg_modes(
          Spectral::Swsh::swsh_transform(l_max, 1, get(tag_data)), l_max)
          .data();

  data_to_write[0] = time_id.substep_time().value();
  for (size_t i = 0; i < square(observation_l_max + 1); ++i) {
    data_to_write[2 * i + 1] = real(goldberg_modes[i]);
    data_to_write[2 * i + 2] = imag(goldberg_modes[i]);
  }
  Parallel::threaded_action<observers::ThreadedActions::WriteSimpleData>(
      observer_proxy, file_legend, data_to_write,
      "/" + db::tag_name<Tag>() + "_Noninertial");
}
}  // namespace detail

/*!
 * \ingroup ActionsGroup
 * \brief Places the data from the current hypersurface necessary to compute
 * `Tag` in the `ScriPlusInterpolationManager` associated with the `Tag`.
 *
 * \details Adds both the appropriate scri+ value(s) and a number of target
 * inertial times to interpolate of quantity equal to the
 * `InitializationTags::ScriOutputDensity` determined from options, equally
 * spaced between the current time and the next time in the algorithm.
 *
 * Uses:
 * - `::Tags::TimeStepId`
 * - `::Tags::Next<::Tags::TimeStepId>`
 * - `Cce::InitializationTags::ScriOutputDensity`
 * - if `Tag` is `::Tags::Multiplies<Lhs, Rhs>`:
 *   - `Lhs` and `Rhs`
 * - if `Tag` has `Cce::Tags::Du<Argument>`:
 *   - `Argument`
 * - otherwise uses `Tag`
 *
 * \ref DataBoxGroup changes:
 * - Modifies:
 *   - `Tags::InterpolationManager<ComplexDataVector, Tag>`
 * - Adds: nothing
 * - Removes: nothing
 */
template <typename Tag, typename BoundaryComponent>
struct InsertInterpolationScriData {
  using const_global_cache_tags = tmpl::flatten<
      tmpl::list<InitializationTags::ScriOutputDensity,
                 std::conditional_t<
                     tt::is_a_v<AnalyticWorldtubeBoundary, BoundaryComponent>,
                     tmpl::list<Tags::OutputNoninertialNews>, tmpl::list<>>>>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (tt::is_a_v<AnalyticWorldtubeBoundary, BoundaryComponent>) {
      if (db::get<Tags::OutputNoninertialNews>(box) and
          db::get<::Tags::TimeStepId>(box).substep() == 0 and
          std::is_same_v<Tag, Tags::News>) {
        detail::output_impl<Tags::News, ParallelComponent>(
            db::get<Tags::ObservationLMax>(box), db::get<Tags::LMax>(box),
            db::get<::Tags::TimeStepId>(box), db::get<Tags::News>(box), cache);
        db::get<Tags::AnalyticBoundaryDataManager>(box)
            .template write_news<ParallelComponent>(
                cache, db::get<::Tags::TimeStepId>(box).substep_time().value());
      }
    }
    if (db::get<::Tags::TimeStepId>(box).substep() == 0) {
      // insert the data points into the interpolator.
      db::mutate_apply<detail::InsertIntoInterpolationManagerImpl<Tag>>(
          make_not_null(&box));

      const auto& time_span_deque =
          db::get<Tags::InterpolationManager<ComplexDataVector, Tag>>(box)
              .get_u_bondi_ranges();

      const double this_time = time_span_deque.back().first;
      double time_delta_estimate = db::get<::Tags::TimeStep>(box).value();
      if (time_span_deque.size() > 1) {
        time_delta_estimate =
            this_time - time_span_deque[time_span_deque.size() - 2].first;
      }

      // insert the target times into the interpolator.
      db::mutate<Tags::InterpolationManager<ComplexDataVector, Tag>>(
          make_not_null(&box),
          [&this_time, &time_delta_estimate](
              const gsl::not_null<
                  ScriPlusInterpolationManager<ComplexDataVector, Tag>*>
                  interpolation_manager,
              const size_t number_of_interpolated_times) noexcept {
            for (size_t i = 0; i < number_of_interpolated_times; ++i) {
              interpolation_manager->insert_target_time(
                  this_time +
                  time_delta_estimate * static_cast<double>(i) /
                      static_cast<double>(number_of_interpolated_times));
            }
          },
          db::get<InitializationTags::ScriOutputDensity>(box));
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Cce

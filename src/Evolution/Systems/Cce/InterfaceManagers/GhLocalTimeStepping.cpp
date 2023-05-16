// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"

#include <cstddef>
#include <deque>
#include <map>
#include <memory>
#include <tuple>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Cce::InterfaceManagers {

namespace detail {
template <typename VarsToInterpolate>
auto create_span_for_time_value(
    double time, int interpolator_length,
    const std::map<LinkedMessageId<double>, VarsToInterpolate,
                   LinkedMessageIdLessComparator<double>>& gh_data) {
  const auto now = gh_data.upper_bound(time);
  if (std::distance(gh_data.begin(), now) < interpolator_length) {
    return gh_data.begin();
  } else if (std::distance(now, gh_data.end()) < interpolator_length) {
    return std::prev(gh_data.end(), 2 * interpolator_length);
  }
  return std::prev(now, interpolator_length);
}

template <typename VarsToInterpolate>
bool interpolate_to_time(
    const gsl::not_null<VarsToInterpolate*> vars_to_interpolate,
    const std::map<LinkedMessageId<double>, VarsToInterpolate,
                   LinkedMessageIdLessComparator<double>>& gh_data,
    const std::unique_ptr<intrp::SpanInterpolator>& interpolator,
    const double target_time) {
  auto iterator_start = detail::create_span_for_time_value(
      target_time, interpolator->required_number_of_points_before_and_after(),
      gh_data);
  for (auto gh_data_it = gh_data.begin();
       gh_data_it !=
       std::next(
           iterator_start,
           2 * static_cast<int>(
                   interpolator->required_number_of_points_before_and_after()) -
               1);
       ++gh_data_it) {
    if (gh_data_it->first.id != std::next(gh_data_it)->first.previous) {
      return false;
    }
  }
  DataVector time_points{
      2 * interpolator->required_number_of_points_before_and_after()};
  DataVector tensor_component_values{
      2 * interpolator->required_number_of_points_before_and_after()};
  for (auto [i, map_it] = std::make_tuple(0_st, iterator_start);
       i < time_points.size(); ++i, ++map_it) {
    time_points[i] = map_it->first.id;
  }
  tmpl::for_each<typename VarsToInterpolate::tags_list>(
      [&vars_to_interpolate, &interpolator, &time_points,
       &tensor_component_values, &target_time,
       &iterator_start](auto tensor_tag_v) {
        using tensor_tag = typename decltype(tensor_tag_v)::type;
        auto& tensor = get<tensor_tag>(*vars_to_interpolate);
        for (size_t i = 0; i < tensor.size(); ++i) {
          for (size_t offset = 0;
               offset < vars_to_interpolate->number_of_grid_points();
               ++offset) {
            // assemble data into easily interpolated structures
            for (auto [time_index, gh_data_it] =
                     std::make_tuple(0_st, iterator_start);
                 time_index <
                 2 * interpolator->required_number_of_points_before_and_after();
                 ++time_index, ++gh_data_it) {
              tensor_component_values[time_index] =
                  get<tensor_tag>(gh_data_it->second)[i][offset];
            }
            tensor[i][offset] = interpolator->interpolate(
                gsl::span<const double>{time_points.data(), time_points.size()},
                gsl::span<const double>{tensor_component_values.data(),
                                        tensor_component_values.size()},
                target_time);
          }
        }
      });
  return true;
}
}  // namespace detail

std::unique_ptr<GhInterfaceManager> GhLocalTimeStepping::get_clone() const {
  return std::make_unique<GhLocalTimeStepping>(*this);
}

void GhLocalTimeStepping::insert_gh_data(
    const LinkedMessageId<double>& time_and_previous,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& pi) {
  gh_variables input_gh_variables{get<0, 0>(spacetime_metric).size()};
  get<gr::Tags::SpacetimeMetric<DataVector, 3>>(input_gh_variables) =
      spacetime_metric;
  get<gh::Tags::Pi<DataVector, 3>>(input_gh_variables) = pi;
  get<gh::Tags::Phi<DataVector, 3>>(input_gh_variables) = phi;
  gh_data_.insert({time_and_previous, std::move(input_gh_variables)});
  clean_up_gh_data();
}

void GhLocalTimeStepping::request_gh_data(const TimeStepId& time_id) {
  requests_.insert(time_id);
  if (requests_.size() == 1) {
    clean_up_gh_data();
  }
}

auto GhLocalTimeStepping::retrieve_and_remove_first_ready_gh_data()
    -> std::optional<std::tuple<TimeStepId, gh_variables>> {
  if (requests_.empty()) {
    return std::nullopt;
  }
  const double first_request = requests_.begin()->substep_time();
  if (gh_data_.size() >=
          interpolator_->required_number_of_points_before_and_after() * 2 and
      gh_data_.rbegin()->first.id > first_request) {
    gh_variables requested_values{
        gh_data_.rbegin()->second.number_of_grid_points()};
    bool success =
        gh_data_.begin()->first.previous == latest_removed_ and
        detail::interpolate_to_time(make_not_null(&requested_values), gh_data_,
                                    interpolator_, first_request);
    if (not success) {
      return std::nullopt;
    }
    std::tuple requested_data{*requests_.begin(), std::move(requested_values)};
    requests_.erase(requests_.begin());
    clean_up_gh_data();
    return requested_data;
  }
  return std::nullopt;
}

void GhLocalTimeStepping::clean_up_gh_data() {
  if (requests_.empty() or gh_data_.empty()) {
    return;
  }
  // count the number of elements to remove from the start
  const size_t max_to_remove =
      gh_data_.size() -
      interpolator_->required_number_of_points_before_and_after() * 2;
  size_t number_of_points_before_first_request =
      static_cast<size_t>(std::distance(
          gh_data_.begin(),
          gh_data_.upper_bound(requests_.begin()->substep_time())));
  const size_t number_of_points_to_remove = std::min(
      max_to_remove, std::max(number_of_points_before_first_request, 1_st) - 1);
  for (size_t i = 0; i < number_of_points_to_remove; ++i) {
    if (gh_data_.begin()->first.previous != latest_removed_) {
      // times are currently not ordered, so it is not safe to remove any more.
      break;
    }
    latest_removed_ = gh_data_.begin()->first.id;
    gh_data_.erase(gh_data_.begin());
  }
}

void GhLocalTimeStepping::pup(PUP::er& p) {
  pup_override(p, gh_data_);
  p | requests_;
  p | interpolator_;
  p | latest_removed_;
}

PUP::able::PUP_ID GhLocalTimeStepping::my_PUP_ID = 0;
}  // namespace Cce::InterfaceManagers

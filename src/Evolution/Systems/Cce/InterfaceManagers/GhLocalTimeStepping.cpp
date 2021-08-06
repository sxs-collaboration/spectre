// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"

#include <cstddef>
#include <deque>
#include <memory>
#include <tuple>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"

namespace Cce::InterfaceManagers {

namespace detail {
template <typename VarsToInterpolate>
auto create_span_for_time_value(
    double time, size_t interpolator_length,
    const std::map<double, VarsToInterpolate>& gh_data) noexcept {
  auto lower_iterator = gh_data.upper_bound(time);
  auto upper_iterator = lower_iterator;
  size_t lower_span_size = 0;
  while (lower_iterator != gh_data.begin() and
         lower_span_size < interpolator_length) {
    --lower_iterator;
    ++lower_span_size;
  }
  size_t upper_span_size = 0;
  while (upper_iterator != gh_data.end() and
         upper_span_size < interpolator_length) {
    ++upper_iterator;
    ++upper_span_size;
  }
  if (lower_span_size < interpolator_length) {
    while (lower_span_size < interpolator_length) {
      ++upper_iterator;
      ++lower_span_size;
    }
  } else if (upper_span_size < interpolator_length) {
    while (upper_span_size < interpolator_length) {
      --lower_iterator;
      ++upper_span_size;
    }
  }
  return lower_iterator;
}

template <typename VarsToInterpolate>
void interpolate_to_time(
    const gsl::not_null<VarsToInterpolate*> vars_to_interpolate,
    const std::map<double, VarsToInterpolate>& gh_data,
    const std::unique_ptr<intrp::SpanInterpolator>& interpolator,
    const double target_time) noexcept {
  auto iterator_start = detail::create_span_for_time_value(
      target_time, interpolator->required_number_of_points_before_and_after(),
      gh_data);
  DataVector time_points{
      2 * interpolator->required_number_of_points_before_and_after()};
  DataVector tensor_component_values{
      2 * interpolator->required_number_of_points_before_and_after()};
  for (auto [i, map_it] = std::make_tuple(0_st, iterator_start);
       i < time_points.size(); ++i, ++map_it) {
    time_points[i] = map_it->first;
  }
  tmpl::for_each<typename VarsToInterpolate::tags_list>(
      [&vars_to_interpolate, &interpolator, &time_points,
       &tensor_component_values, &target_time,
       &iterator_start](auto tensor_tag_v) noexcept {
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
}
}  // namespace detail

std::unique_ptr<GhInterfaceManager> GhLocalTimeStepping::get_clone()
    const noexcept {
  return std::make_unique<GhLocalTimeStepping>(*this);
}

void GhLocalTimeStepping::insert_gh_data(
    double time, const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::aa<DataVector, 3>& pi) noexcept {
  gh_variables input_gh_variables{get<0, 0>(spacetime_metric).size()};
  get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
      input_gh_variables) = spacetime_metric;
  get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(input_gh_variables) =
      pi;
  get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
      input_gh_variables) = phi;
  gh_data_.insert({time, std::move(input_gh_variables)});
  clean_up_gh_data();
}

void GhLocalTimeStepping::request_gh_data(const TimeStepId& time_id) noexcept {
  requests_.push_back(time_id);
  if (requests_.size() == 1) {
    clean_up_gh_data();
  }
}

auto GhLocalTimeStepping::retrieve_and_remove_first_ready_gh_data() noexcept
    -> std::optional<std::tuple<TimeStepId, gh_variables>> {
  if (requests_.empty()) {
    return std::nullopt;
  }
  const double first_request = requests_.front().substep_time().value();
  if (gh_data_.size() >=
          interpolator_->required_number_of_points_before_and_after() * 2 and
      gh_data_.rbegin()->first > first_request) {
    gh_variables requested_values{
        gh_data_.rbegin()->second.number_of_grid_points()};
    detail::interpolate_to_time(make_not_null(&requested_values), gh_data_,
                                interpolator_, first_request);
    std::tuple requested_data{requests_.front(), std::move(requested_values)};
    requests_.pop_front();
    clean_up_gh_data();
    return requested_data;
  }
  return std::nullopt;
}

void GhLocalTimeStepping::clean_up_gh_data() noexcept {
  if (requests_.empty() or gh_data_.empty()) {
    return;
  }
  // count the number of elements to remove from the start
  const size_t max_to_remove =
      gh_data_.size() -
      interpolator_->required_number_of_points_before_and_after() * 2;
  size_t number_of_points_before_first_request = 0;
  for (auto gh_data_it = gh_data_.begin();
       gh_data_it != gh_data_.end() and
       gh_data_it->first < requests_.front().substep_time().value();
       ++gh_data_it, ++number_of_points_before_first_request) {
  }
  const size_t number_of_points_to_remove = std::min(
      max_to_remove, std::max(number_of_points_before_first_request, 1_st) - 1);
  for (size_t i = 0; i < number_of_points_to_remove; ++i) {
    gh_data_.erase(gh_data_.begin());
  }
}

void GhLocalTimeStepping::pup(PUP::er& p) noexcept {
  p | gh_data_;
  p | requests_;
  p | interpolator_;
}

PUP::able::PUP_ID GhLocalTimeStepping::my_PUP_ID = 0;
}  // namespace Cce::InterfaceManagers

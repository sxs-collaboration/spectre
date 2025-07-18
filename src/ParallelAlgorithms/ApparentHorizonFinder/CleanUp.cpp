// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/CleanUp.hpp"

#include <optional>
#include <set>
#include <unordered_map>

#include "DataStructures/LinkedMessageId.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
template <typename Fr>
void clean_up_horizon_finder(
    const gsl::not_null<std::optional<LinkedMessageId<double>>*>
        current_time_optional,
    const gsl::not_null<std::unordered_map<LinkedMessageId<double>,
                                           ah::Storage::SingleTimeStorage<Fr>>*>
        all_storage,
    const gsl::not_null<std::set<LinkedMessageId<double>>*> completed_times,
    const gsl::not_null<FastFlow*> fast_flow) {
  ASSERT(current_time_optional->has_value(),
         "Current time must be set in order to clean up the horizon finder.");

  const auto& current_time = current_time_optional->value();

  all_storage->erase(current_time);

  // We want to keep track of all completed times to deal with the
  // possibility of late arrivals of volume data. We could keep all
  // completed times forever, but we probably don't want it to get too
  // large, so we limit its size.  We assume that asynchronous calls to
  // this action do not span more than 1000 temporal_ids.
  completed_times->insert(current_time);
  while (completed_times->size() > 1000) {
    completed_times->erase(completed_times->begin());
  }

  // Reset time before we choose a new time
  current_time_optional->reset();

  fast_flow->reset_for_next_find();
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void clean_up_horizon_finder(                                       \
      const gsl::not_null<std::optional<LinkedMessageId<double>>*>             \
          current_time_optional,                                               \
      const gsl::not_null<                                                     \
          std::unordered_map<LinkedMessageId<double>,                          \
                             ah::Storage::SingleTimeStorage<FRAME(data)>>*>    \
          all_storage,                                                         \
      const gsl::not_null<std::set<LinkedMessageId<double>>*> completed_times, \
      const gsl::not_null<FastFlow*> fast_flow);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Inertial, Frame::Distorted, Frame::Grid))

#undef INSTANTIATE
#undef FRAME
}  // namespace ah

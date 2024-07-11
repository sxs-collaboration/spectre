// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/Random.hpp"

#include <boost/functional/hash.hpp>
#include <cmath>
#include <cstddef>
#include <pup.h>
#include <random>
#include <utility>

#include "Domain/Structure/Element.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace StepChoosers {

template <typename StepChooserUse, size_t VolumeDim>
Random<StepChooserUse, VolumeDim>::Random() = default;

template <typename StepChooserUse, size_t VolumeDim>
Random<StepChooserUse, VolumeDim>::Random(CkMigrateMessage* /*unused*/) {}

template <typename StepChooserUse, size_t VolumeDim>
Random<StepChooserUse, VolumeDim>::Random(const double minimum,
                                          const double maximum,
                                          const size_t seed,
                                          const Options::Context& context)
    : minimum_(minimum), maximum_(maximum), seed_(seed) {
  if (minimum_ >= maximum_) {
    PARSE_ERROR(context, "Must have Minimum < Maximum");
  }
}

template <typename StepChooserUse, size_t VolumeDim>
std::pair<TimeStepRequest, bool> Random<StepChooserUse, VolumeDim>::operator()(
    const Element<VolumeDim>& element, const TimeStepId& time_step_id,
    const double last_step) const {
  size_t local_seed = seed_;
  boost::hash_combine(local_seed, element.id());
  boost::hash_combine(local_seed, time_step_id);
  std::mt19937_64 rng(local_seed);
  std::uniform_real_distribution<> dist(log(minimum_), log(maximum_));
  for (;;) {
    const double step = exp(dist(rng));
    // Don't produce out-of-range values because of roundoff.
    if (step >= minimum_ and step <= maximum_) {
      return {{.size_goal = std::copysign(step, last_step)}, true};
    }
  }
}

template <typename StepChooserUse, size_t VolumeDim>
bool Random<StepChooserUse, VolumeDim>::uses_local_data() const {
  return true;
}

template <typename StepChooserUse, size_t VolumeDim>
bool Random<StepChooserUse, VolumeDim>::can_be_delayed() const {
  return true;
}

template <typename StepChooserUse, size_t VolumeDim>
void Random<StepChooserUse, VolumeDim>::pup(PUP::er& p) {
  p | minimum_;
  p | maximum_;
  p | seed_;
}

template <typename StepChooserUse, size_t VolumeDim>
PUP::able::PUP_ID Random<StepChooserUse, VolumeDim>::my_PUP_ID = 0;  // NOLINT

#define USE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class Random<StepChooserUse::USE(data), DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (Slab, LtsStep), (1, 2, 3))

#undef INSTANTIATE
#undef DIM
#undef USE
}  // namespace StepChoosers

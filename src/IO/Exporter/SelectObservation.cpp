// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/SelectObservation.hpp"

#include <cstddef>

#include "IO/H5/VolumeData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace spectre::Exporter {

size_t SelectObservation::operator()(const ObservationId observation_id) const {
  return observation_id.value;
}

size_t SelectObservation::operator()(
    const ObservationStep observation_step) const {
  int step = observation_step.value;
  const auto obs_ids = volfile.list_observation_ids();
  if (step < 0) {
    step += static_cast<int>(obs_ids.size());
  }
  if (step < 0 or static_cast<size_t>(step) >= obs_ids.size()) {
    ERROR_NO_TRACE("Invalid observation step: "
                   << observation_step.value << ". There are " << obs_ids.size()
                   << " observations in the file.");
  }
  return obs_ids[static_cast<size_t>(step)];
}

size_t SelectObservation::operator()(const double observation_value) const {
  return (*this)(ObservationValue{observation_value});
}

size_t SelectObservation::operator()(
    const ObservationValue observation_value) const {
  return volfile.find_observation_id(observation_value.value,
                                     observation_value.epsilon);
}

}  // namespace spectre::Exporter

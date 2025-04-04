// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <variant>

#include "IO/Exporter/Exporter.hpp"
#include "IO/H5/VolumeData.hpp"

namespace spectre::Exporter {

/// Determines the selected observation ID in the volume data file, given
/// multiple possible ways to specify the observation.
struct SelectObservation {
  size_t operator()(ObservationId observation_id) const;
  size_t operator()(ObservationStep observation_step) const;
  size_t operator()(double observation_value) const;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const h5::VolumeData& volfile;
  double obs_value_eps = 1e-12;
};

}  // namespace spectre::Exporter

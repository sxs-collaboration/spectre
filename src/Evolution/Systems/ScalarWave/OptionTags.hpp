// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/String.hpp"

namespace ScalarWave::OptionTags {
struct MassSquared {
  using type = double;
  static constexpr Options::String help {"The squared
    mass value for the scalar wave"};
  };
}  // namespace ScalarWave::OptionTags

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DhGaugeParameters.hpp"

#include <array>
#include <pup.h>
#include <pup_stl.h>

GeneralizedHarmonic::gauges::DhGaugeParameters<true>::DhGaugeParameters(
    const double start, const double window, const double width,
    const std::array<double, 3>& amps, const std::array<int, 3>& exps)
    : rollon_start(start),
      rollon_window(window),
      spatial_decay_width(width),
      amplitudes(amps),
      exponents(exps) {}

GeneralizedHarmonic::gauges::DhGaugeParameters<false>::DhGaugeParameters(
    const double width, const std::array<double, 3>& amps,
    const std::array<int, 3>& exps)
    : spatial_decay_width(width), amplitudes(amps), exponents(exps) {}

// clang-tidy: google-runtime-references
void GeneralizedHarmonic::gauges::DhGaugeParameters<true>::pup(
    PUP::er& p) noexcept {  // NOLINT
  p | rollon_start;
  p | rollon_window;
  p | spatial_decay_width;
  p | amplitudes;
  p | exponents;
}

// clang-tidy: google-runtime-references
void GeneralizedHarmonic::gauges::DhGaugeParameters<false>::pup(
    PUP::er& p) noexcept {  // NOLINT
  p | spatial_decay_width;
  p | amplitudes;
  p | exponents;
}

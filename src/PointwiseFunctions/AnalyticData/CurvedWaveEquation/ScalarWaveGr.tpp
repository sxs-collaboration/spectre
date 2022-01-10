// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/ScalarWaveGr.hpp"

#include <pup.h>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace CurvedScalarWave ::AnalyticData {

template <typename ScalarFieldData, typename BackgroundGrData>
void ScalarWaveGr<ScalarFieldData, BackgroundGrData>::pup(PUP::er& p) {
  p | flat_space_scalar_wave_data_;
  p | background_gr_data_;
}

template <typename ScalarFieldData, typename BackgroundGrData>
tuples::TaggedTuple<Tags::Pi>
ScalarWaveGr<ScalarFieldData, BackgroundGrData>::variables(
    const tnsr::I<DataVector, volume_dim>& x,
    tmpl::list<Tags::Pi> /*meta*/) const {
  constexpr double default_initial_time = 0.;
  const auto flat_space_scalar_wave_vars =
      flat_space_scalar_wave_data_.variables(
          x, default_initial_time,
          tmpl::list<ScalarWave::Tags::Psi, ScalarWave::Tags::Pi,
                     ScalarWave::Tags::Phi<volume_dim>>{});
  const auto spacetime_variables = background_gr_data_.variables(
      x, default_initial_time, spacetime_tags<DataVector>{});
  auto result = make_with_value<tuples::TaggedTuple<Tags::Pi>>(x, 0.);

  const auto shift_dot_dpsi = dot_product(
      get<gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>>(
          spacetime_variables),
      get<ScalarWave::Tags::Phi<volume_dim>>(flat_space_scalar_wave_vars));

  get(get<Tags::Pi>(result)) =
      (get(shift_dot_dpsi) +
       get(get<ScalarWave::Tags::Pi>(flat_space_scalar_wave_vars))) /
      get(get<gr::Tags::Lapse<DataVector>>(spacetime_variables));

  return result;
}

template <typename LocalScalarFieldData, typename LocalBackgroundData>
bool operator==(
    const ScalarWaveGr<LocalScalarFieldData, LocalBackgroundData>& lhs,
    const ScalarWaveGr<LocalScalarFieldData, LocalBackgroundData>& rhs) {
  return lhs.background_gr_data_ == rhs.background_gr_data_;
}

template <typename ScalarFieldData, typename BackgroundData>
bool operator!=(const ScalarWaveGr<ScalarFieldData, BackgroundData>& lhs,
                const ScalarWaveGr<ScalarFieldData, BackgroundData>& rhs) {
  return not(lhs == rhs);
}
}  // namespace CurvedScalarWave::AnalyticData
/// \endcond

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/ScalarWaveGr.hpp"

#include <pup.h>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/PureSphericalHarmonic.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::AnalyticData {

template <typename ScalarFieldData, typename BackgroundGrData>
void ScalarWaveGr<ScalarFieldData, BackgroundGrData>::pup(PUP::er& p) {
  p | scalar_wave_data_;
  p | background_gr_data_;
}

template <typename ScalarFieldData, typename BackgroundGrData>
tuples::TaggedTuple<Tags::Pi>
ScalarWaveGr<ScalarFieldData, BackgroundGrData>::variables(
    const tnsr::I<DataVector, volume_dim>& x,
    tmpl::list<Tags::Pi> /*meta*/) const {
  constexpr double default_initial_time = 0.;
  const auto scalar_wave_vars = scalar_wave_data_.variables(
      x, default_initial_time, evolved_field_vars_tags{});
  if constexpr (is_curved) {
    return std::move(get<InitialDataPi>(scalar_wave_vars));
  }
  const auto spacetime_variables = background_gr_data_.variables(
      x, default_initial_time, spacetime_tags<DataVector>{});
  auto result = make_with_value<tuples::TaggedTuple<Tags::Pi>>(x, 0.);

  const auto shift_dot_dpsi =
      dot_product(get<gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>>(
                      spacetime_variables),
                  get<InitialDataPhi>(scalar_wave_vars));

  get(get<Tags::Pi>(result)) =
      (get(shift_dot_dpsi) + get(get<InitialDataPi>(scalar_wave_vars))) /
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

// background
#define BG(data) BOOST_PP_TUPLE_ELEM(0, data)
// analytic soution/data
#define SOL(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template class ScalarWaveGr<SOL(data), BG(data)>;                       \
  template bool operator==(const ScalarWaveGr<SOL(data), BG(data)>& lhs,  \
                           const ScalarWaveGr<SOL(data), BG(data)>& rhs); \
  template bool operator!=(const ScalarWaveGr<SOL(data), BG(data)>& lhs,  \
                           const ScalarWaveGr<SOL(data), BG(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE, (gr::Solutions::Minkowski<1>),
                        (ScalarWave::Solutions::PlaneWave<1>))
GENERATE_INSTANTIATIONS(INSTANTIATE, (gr::Solutions::Minkowski<2>),
                        (ScalarWave::Solutions::PlaneWave<2>))
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (gr::Solutions::Minkowski<3>,
                         gr::Solutions::KerrSchild),
                        (ScalarWave::Solutions::PlaneWave<3>,
                         ScalarWave::Solutions::RegularSphericalWave,
                         CurvedScalarWave::AnalyticData::PureSphericalHarmonic))

#undef BG
#undef SOL
#undef INSTANTIATE
}  // namespace CurvedScalarWave::AnalyticData

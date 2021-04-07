// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/PlaneWave3DKerrSchild.hpp"

namespace CurvedScalarWave ::AnalyticData {
template class ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,
                            gr::Solutions::KerrSchild>;
template bool operator==(
    const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,
                       gr::Solutions::KerrSchild>& lhs,
    const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,
                       gr::Solutions::KerrSchild>& rhs) noexcept;
template bool operator!=(
    const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,
                       gr::Solutions::KerrSchild>& lhs,
    const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,
                       gr::Solutions::KerrSchild>& rhs) noexcept;
}  // namespace CurvedScalarWave::AnalyticData

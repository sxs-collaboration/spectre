// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/ScalarWaveGr.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"

namespace CurvedScalarWave ::AnalyticData {
template class ScalarWaveGr<ScalarWave::Solutions::RegularSphericalWave,
                            gr::Solutions::KerrSchild>;
template bool operator==(
    const ScalarWaveGr<ScalarWave::Solutions::RegularSphericalWave,
                       gr::Solutions::KerrSchild>& lhs,
    const ScalarWaveGr<ScalarWave::Solutions::RegularSphericalWave,
                       gr::Solutions::KerrSchild>& rhs) noexcept;
template bool operator!=(
    const ScalarWaveGr<ScalarWave::Solutions::RegularSphericalWave,
                       gr::Solutions::KerrSchild>& lhs,
    const ScalarWaveGr<ScalarWave::Solutions::RegularSphericalWave,
                       gr::Solutions::KerrSchild>& rhs) noexcept;
}  // namespace CurvedScalarWave::AnalyticData

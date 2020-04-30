// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace CurvedScalarWave {
namespace AnalyticData {
template <typename ScalarFieldSolution, typename BackgroundGrData>
class ScalarWaveGr;
}  // namespace AnalyticData
}  // namespace CurvedScalarWave

namespace gr::Solutions {
class KerrSchild;
}  // namespace gr::Solutions

namespace ScalarWave {
namespace Solutions {
template <size_t Dim>
class PlaneWave;

class RegularSphericalWave;
}  // namespace Solutions
}  // namespace ScalarWave

template <size_t Dim, typename InitialData>
struct EvolutionMetavars;

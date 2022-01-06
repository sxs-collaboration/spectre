// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace CurvedScalarWave::AnalyticData {
template <typename ScalarFieldSolution, typename BackgroundGrData>
class ScalarWaveGr;
}  // namespace CurvedScalarWave::AnalyticData

namespace gr::Solutions {
template <size_t Dim>
class Minkowski;
}  // namespace gr::Solutions

namespace ScalarWave::Solutions {
template <size_t Dim>
class PlaneWave;
}  // namespace ScalarWave::Solutions

template <size_t Dim, typename InitialData>
struct EvolutionMetavars;
/// \endcond

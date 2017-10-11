// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"

#include "Utilities/StdHelpers.hpp"

namespace ScalarWave {
namespace Solutions {

template <size_t Dim>
PlaneWave<Dim>::PlaneWave(std::array<double, Dim> wave_vector,
                          std::array<double, Dim> center,
                          std::unique_ptr<MathFunction<1>>&& profile) noexcept
    : wave_vector_(std::move(wave_vector)),
      center_(std::move(center)),
      profile_(std::move(profile)),
      omega_(magnitude(wave_vector_)) {}
}  // namespace Solutions
}  // namespace ScalarWave

template class ScalarWave::Solutions::PlaneWave<1>;
template class ScalarWave::Solutions::PlaneWave<2>;
template class ScalarWave::Solutions::PlaneWave<3>;

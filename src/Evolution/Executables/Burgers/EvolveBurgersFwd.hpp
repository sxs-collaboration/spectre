// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace Burgers {
namespace AnalyticData {
class Sinusoid;
}  // namespace AnalyticData

namespace Solutions {
class Bump;
class Linear;
class Step;
}  // namespace Solutions
}  // namespace Burgers

template <typename InitialData>
struct EvolutionMetavars;

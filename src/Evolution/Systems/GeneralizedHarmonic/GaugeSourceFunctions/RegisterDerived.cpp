// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/RegisterDerived.hpp"

#include <cstddef>

#include "Evolution/Systems/GeneralizedHarmonic/AllSolutions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Factory.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::gauges {
namespace {
template <size_t Dim>
void impl() {
  // The analytic gauge condition also can hold all the different solutions, so
  // register those too.
  using solutions = gh::solutions_including_matter<Dim>;
  register_classes_with_charm(solutions{});
}
}  // namespace

void register_derived_with_charm() {
  impl<1>();
  impl<2>();
  impl<3>();
  register_classes_with_charm(
      tmpl::list<AnalyticChristoffel, DampedHarmonic, Harmonic>{});
}
}  // namespace gh::gauges

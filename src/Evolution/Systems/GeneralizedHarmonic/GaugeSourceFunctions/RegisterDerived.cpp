// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Factory.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace GeneralizedHarmonic::gauges {
namespace {
template <size_t Dim>
void impl() {
  Parallel::register_classes_with_charm(tmpl::list<>{});
}
}  // namespace

void register_derived_with_charm() {
  impl<1>();
  impl<2>();
  impl<3>();
  Parallel::register_classes_with_charm(tmpl::list<DampedHarmonic, Harmonic>{});
}
}  // namespace GeneralizedHarmonic::gauges

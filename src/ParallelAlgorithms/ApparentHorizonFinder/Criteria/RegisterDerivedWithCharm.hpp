// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Residual.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Shape.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace ah::Criteria {
void register_derived_with_charm() {
  register_classes_with_charm<Residual>();
  register_classes_with_charm<Shape>();
}
}  // namespace ah::Criteria

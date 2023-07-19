// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/MathFunctions/RegisterDerivedWithCharm.hpp"

#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace MathFunctions {
void register_derived_with_charm() {
  register_classes_with_charm<Gaussian<1, Frame::Inertial>,
                              PowX<1, Frame::Inertial>,
                              Sinusoid<1, Frame::Inertial>>();
}
}  // namespace MathFunctions

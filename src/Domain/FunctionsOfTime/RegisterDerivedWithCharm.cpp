// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>

#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace FunctionsOfTime {
void register_derived_with_charm() noexcept {
  Parallel::register_classes_with_charm<FunctionsOfTime::PiecewisePolynomial<2>,
                                        FunctionsOfTime::PiecewisePolynomial<3>,
                                        FunctionsOfTime::PiecewisePolynomial<4>,
                                        FunctionsOfTime::SettleToConstant>();
}
}  // namespace FunctionsOfTime
}  // namespace domain

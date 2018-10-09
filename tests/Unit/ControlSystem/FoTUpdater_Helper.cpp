// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/ControlSystem/FoTUpdater_Helper.hpp"

#include <algorithm>

#include "Utilities/Gsl.hpp"

/// \cond
namespace FunctionsOfTime {
template <size_t DerivOrder>
class PiecewisePolynomial;
}  // namespace FunctionsOfTime
/// \endcond

namespace TestHelpers {
namespace ControlErrors {

template <size_t DerivOrder>
Translation<DerivOrder>::Translation(DataVector target_coords) noexcept
    : target_coords_{std::move(target_coords)} {}

template <size_t DerivOrder>
void Translation<DerivOrder>::operator()(
    gsl::not_null<FunctionOfTimeUpdater<DerivOrder>*> updater,
    const FunctionsOfTime::PiecewisePolynomial<DerivOrder>& f_of_t, double time,
    const DataVector& coords) noexcept {
  const DataVector q = coords - target_coords_ - f_of_t.func(time)[0];
  updater->measure(time, q);
}
}  // namespace ControlErrors
}  // namespace TestHelpers

/// \cond
template class TestHelpers::ControlErrors::Translation<2>;
/// \endcond

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/ControlSystem/FoTUpdater_Helper.hpp"

#include <algorithm>

#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers {
namespace ControlErrors {

template <size_t DerivOrder>
Translation<DerivOrder>::Translation(DataVector target_coords) noexcept
    : target_coords_{std::move(target_coords)} {}

template <size_t DerivOrder>
void Translation<DerivOrder>::operator()(
    const gsl::not_null<FunctionOfTimeUpdater<DerivOrder>*> updater,
    const domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>& f_of_t,
    const double time, const DataVector& coords) noexcept {
  const DataVector q = coords - target_coords_ - f_of_t.func(time)[0];
  updater->measure(time, q);
}
}  // namespace ControlErrors
}  // namespace TestHelpers

template class TestHelpers::ControlErrors::Translation<2>;

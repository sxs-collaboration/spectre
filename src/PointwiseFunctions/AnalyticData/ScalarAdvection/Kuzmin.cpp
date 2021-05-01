// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ScalarAdvection/Kuzmin.hpp"

#include <limits>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Math.hpp"

namespace ScalarAdvection::AnalyticData {

template <typename T>
Scalar<T> Kuzmin::u(const tnsr::I<T, 2>& x) const noexcept {
  const double r0{0.15};
  double r_cylinder = std::numeric_limits<double>::signaling_NaN();
  double r_cone = std::numeric_limits<double>::signaling_NaN();
  double r_hump = std::numeric_limits<double>::signaling_NaN();

  const auto r_xy = [&r0](const double x_var, const double x0_var,
                          const double y_var, const double y0_var) noexcept {
    return sqrt(pow(x_var - x0_var, 2.0) + pow(y_var - y0_var, 2.0)) / r0;
  };

  Scalar<T> u_variable{get<0>(x)};
  for (size_t i = 0; i < get_size(get<0>(x)); ++i) {
    const auto& xi = get<0>(x)[i];
    const auto& yi = get<1>(x)[i];
    auto& ui = get(u_variable)[i];

    // slotted cylinder centered at (0.5, 0.75)
    r_cylinder = r_xy(xi, 0.5, yi, 0.75);
    // cone centered at (0.5, 0.25)
    r_cone = r_xy(xi, 0.5, yi, 0.25);
    // hump centered at (0.25, 0.5)
    r_hump = r_xy(xi, 0.25, yi, 0.5);

    if (r_cylinder <= 1.0) {
      if ((abs(xi - 0.5) >= 0.025) or (yi >= 0.85)) {
        ui = 1.0;
      } else {
        ui = 0.0;
      }
    } else if (r_cone <= 1.0) {
      ui = 1.0 - r_cone;
    } else if (r_hump <= 1.0) {
      ui = 0.25 * (1.0 + cos(M_PI * r_hump));
    } else {
      ui = 0.0;
    }
  }
  return u_variable;
}

tuples::TaggedTuple<Tags::U> Kuzmin::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<Tags::U> /*meta*/) const noexcept {
  return {u(x)};
}

void Kuzmin::pup(PUP::er& /*p*/) noexcept {}

bool operator==(const Kuzmin& /*lhs*/, const Kuzmin& /*rhs*/) noexcept {
  return true;
}

bool operator!=(const Kuzmin& lhs, const Kuzmin& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace ScalarAdvection::AnalyticData

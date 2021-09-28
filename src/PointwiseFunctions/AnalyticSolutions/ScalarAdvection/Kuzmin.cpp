// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Kuzmin.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarAdvection::Solutions {

template <typename DataType>
tuples::TaggedTuple<ScalarAdvection::Tags::U> Kuzmin::variables(
    const tnsr::I<DataType, 2>& x, double t,
    tmpl::list<ScalarAdvection::Tags::U> /*meta*/) const {
  auto coords_init = make_with_value<tnsr::I<DataType, 2>>(x, 0.0);
  auto& x0 = get<0>(coords_init);
  auto& y0 = get<1>(coords_init);

  // Map each grid points [x(t),y(t)] back to its initial position at t=0:
  // applying 2D rotation to translated coordinates, and translate them back.
  // Note that in this analytic solution, all the regions with nonzero values of
  // U lies within the circle centered at (0.5, 0.5) with radius 0.5. Therefore
  // we do not need to take care of a specific boundary condition and it is
  // sufficient to simply rotate all the coordinates back to t=0.
  x0 = (get<0>(x) - 0.5) * cos(t) + (get<1>(x) - 0.5) * sin(t) + 0.5;
  y0 = -(get<0>(x) - 0.5) * sin(t) + (get<1>(x) - 0.5) * cos(t) + 0.5;

  // parameters and functions for the initial profile (from the Kuzmin paper)
  const double r0{0.15};
  const auto r_xy = [&r0](const double x_var, const double x0_var,
                          const double y_var, const double y0_var) {
    return sqrt(pow(x_var - x0_var, 2.0) + pow(y_var - y0_var, 2.0)) / r0;
  };

  double r_cylinder = std::numeric_limits<double>::signaling_NaN();
  double r_cone = std::numeric_limits<double>::signaling_NaN();
  double r_hump = std::numeric_limits<double>::signaling_NaN();

  // evaluate u(x,y,t) = u(x0,y0,0)
  auto u_variable = make_with_value<Scalar<DataType>>(coords_init, 0.0);
  for (size_t i = 0; i < get_size(get<0>(x)); ++i) {
    const auto& xi = x0[i];
    const auto& yi = y0[i];
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

void Kuzmin::pup(PUP::er& /*p*/) {}

bool operator==(const Kuzmin& /*lhs*/, const Kuzmin& /*rhs*/) { return true; }

bool operator!=(const Kuzmin& lhs, const Kuzmin& rhs) {
  return not(lhs == rhs);
}

}  // namespace ScalarAdvection::Solutions

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                             \
  template tuples::TaggedTuple<ScalarAdvection::Tags::U> \
  ScalarAdvection::Solutions::Kuzmin::variables(         \
      const tnsr::I<DTYPE(data), 2>& x, double t,        \
      tmpl::list<ScalarAdvection::Tags::U> /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector))

#undef DTYPE
#undef INSTANTIATE

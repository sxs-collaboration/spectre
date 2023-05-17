// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsCoefficients.hpp"

#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Rational.hpp"

namespace TimeSteppers::adams_coefficients {
OrderVector<double> constant_adams_bashforth_coefficients(const size_t order) {
  switch (order) {
    case 0:
      return {};
    case 1:
      return {1.};
    case 2:
      return {-0.5, 1.5};
    case 3:
      return {5.0 / 12.0, -4.0 / 3.0, 23.0 / 12.0};
    case 4:
      return {-3.0 / 8.0, 37.0 / 24.0, -59.0 / 24.0, 55.0 / 24.0};
    case 5:
      return {251.0 / 720.0, -637.0 / 360.0, 109.0 / 30.0, -1387.0 / 360.0,
              1901.0 / 720.0};
    case 6:
      return {-95.0 / 288.0,  959.0 / 480.0,   -3649.0 / 720.0,
              4991.0 / 720.0, -2641.0 / 480.0, 4277.0 / 1440.0};
    case 7:
      return {19087.0 / 60480.0, -5603.0 / 2520.0,   135713.0 / 20160.0,
              -10754.0 / 945.0,  235183.0 / 20160.0, -18637.0 / 2520.0,
              198721.0 / 60480.0};
    case 8:
      return {-5257.0 / 17280.0,     32863.0 / 13440.0,   -115747.0 / 13440.0,
              2102243.0 / 120960.0,  -296053.0 / 13440.0, 242653.0 / 13440.0,
              -1152169.0 / 120960.0, 16083.0 / 4480.0};
    default:
      ERROR("Bad order: " << order);
  }
}

OrderVector<double> constant_adams_moulton_coefficients(const size_t order) {
  switch (order) {
    case 0:
      return {};
    case 1:
      return {1.0};
    case 2:
      return {0.5, 0.5};
    case 3:
      return {-1.0 / 12.0, 2.0 / 3.0, 5.0 / 12.0};
    case 4:
      return {1.0 / 24.0, -5.0 / 24.0, 19.0 / 24.0, 3.0 / 8.0};
    case 5:
      return {-19.0 / 720.0, 53.0 / 360.0, -11.0 / 30.0, 323.0 / 360.0,
              251.0 / 720.0};
    case 6:
      return {3.0 / 160.0,    -173.0 / 1440.0, 241.0 / 720.0,
              -133.0 / 240.0, 1427.0 / 1440.0, 95.0 / 288.0};
    case 7:
      return {-863.0 / 60480.0, 263.0 / 2520.0,     -6737.0 / 20160.0,
              586.0 / 945.0,    -15487.0 / 20160.0, 2713.0 / 2520.0,
              19087.0 / 60480.0};
    case 8:
      return {275.0 / 24192.0,     -11351.0 / 120960.0, 1537.0 / 4480.0,
              -88547.0 / 120960.0, 123133.0 / 120960.0, -4511.0 / 4480.0,
              139849.0 / 120960.0, 5257.0 / 17280.0};
    default:
      ERROR("Bad order: " << order);
  }
}

template <typename T>
OrderVector<T> variable_coefficients(OrderVector<T> control_times,
                                     const T& step_start, const T& step_end) {
  // The coefficients are, for each j,
  // \int_{step_start}^{step_end} dt ell_j(t; control_times),

  // Shift the step to be near zero to minimize the roundoff error
  // from the final polynomial evaluations.
  alg::for_each(control_times, [&](T& t) { t -= step_start; });

  const size_t order = control_times.size();
  OrderVector<T> result;
  for (size_t j = 0; j < order; ++j) {
    // Calculate coefficients of the Lagrange interpolating polynomials,
    // in the standard a_0 + a_1 t + a_2 t^2 + ... form.
    OrderVector<T> poly(order, 0);

    poly[0] = 1;

    for (size_t m = 0; m < order; ++m) {
      if (m == j) {
        continue;
      }
      const T denom = 1 / (control_times[j] - control_times[m]);
      for (size_t i = m < j ? m + 1 : m; i > 0; --i) {
        poly[i] = (poly[i - 1] - poly[i] * control_times[m]) * denom;
      }
      poly[0] *= -control_times[m] * denom;
    }

    // Integrate p(t), term by term.  We choose the constant of
    // integration so the indefinite integral is zero at t=0.  We do
    // not adjust the indexing, so the t^n term in the integral is in
    // the (n-1)th entry of the vector (as opposed to the nth entry
    // before integrating).
    for (size_t m = 0; m < order; ++m) {
      poly[m] /= m + 1;
    }
    result.push_back((step_end - step_start) *
                     evaluate_polynomial(poly, step_end - step_start));
  }
  return result;
}

#define TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template OrderVector<TYPE(data)> variable_coefficients(                   \
      OrderVector<TYPE(data)> control_times, const TYPE(data) & step_start, \
      const TYPE(data) & step_end);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, Rational))

#undef INSTANTIATE
#undef TYPE
}  // namespace TimeSteppers::adams_coefficients

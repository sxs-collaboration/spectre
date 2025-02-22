// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/LargestStepperError.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <typename T>
double largest_stepper_error(const T& values, const T& errors,
                             const StepperErrorTolerances& tolerances) {
  using std::abs;
  // Outer call to max() may be from blaze or an identity function
  // from ContainerHelpers.hpp for doubles.
  using ::max;
  // Inner max() is either blaze or std::max.
  using std::max;
  return max(abs(errors) /
             (tolerances.absolute +
              tolerances.relative * max(abs(values), abs(values + errors))));
}

#define TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                \
  template double largest_stepper_error(                    \
      const TYPE(data) & values, const TYPE(data) & errors, \
      const StepperErrorTolerances& tolerances);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, std::complex<double>, DataVector,
                                      ComplexDataVector))

#undef INSTANTIATE
#undef TYPE

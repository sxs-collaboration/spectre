// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>

#include "DataStructures/BoostMultiArray.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/VectorImpl.hpp"

// Boost MultiArray is used internally in odeint, so odeint must be included
// later
#include <boost/numeric/odeint.hpp>

namespace detail {
struct vector_impl_algebra : boost::numeric::odeint::vector_space_algebra {
  template <typename VectorType>
  static double norm_inf(const VectorType& vector) {
    return max(abs(vector));
  }
};

struct vector_impl_array_algebra : boost::numeric::odeint::array_algebra {
  template <typename VectorType, size_t Size>
  static double norm_inf(const std::array<VectorType, Size>& vector_array) {
    return std::accumulate(
        vector_array.begin(), vector_array.end(), 0.0,
        [](const double maximum_so_far, const VectorType& element) {
          return std::max(maximum_so_far, max(abs(element)));
        });
  }
};
}  // namespace detail

namespace boost {
namespace numeric {
namespace odeint {

namespace detail {
template <typename V>
struct set_unit_value_impl<ComplexDataVector, V, void> {
  static void set_value(ComplexDataVector& t, const V& v) {  // NOLINT
    // this ensures that the blaze expression template resolve the result to a
    // complex vector
    t = std::complex<double>(1.0, 0.0) * v;
  }
};
template <typename V>
struct set_unit_value_impl<ComplexModalVector, V, void> {
  static void set_value(ComplexModalVector& t, const V& v) {  // NOLINT
    // this ensures that the blaze expression template resolve the result to a
    // complex vector
    t = std::complex<double>(1.0, 0.0) * v;
  }
};
}  // namespace detail

// In some integration contexts, boost requires the ability to resize the
// integration arguments in preparation for writing to output buffers passed by
// reference. These specializations make the resize work correctly with spectre
// vector types.
template <class VectorType1, class VectorType2>
struct resize_impl_sfinae<VectorType1, VectorType2,
                          typename boost::enable_if_c<
                              is_derived_of_vector_impl_v<VectorType1> and
                              is_derived_of_vector_impl_v<VectorType2>>::type> {
  static void resize(VectorType1& x1, const VectorType2& x2) {
    x1.destructive_resize(x2.size());
  }
};

template <class VectorType>
struct is_resizeable_sfinae<
    VectorType,
    typename boost::enable_if_c<is_derived_of_vector_impl_v<VectorType>>::type>
    : boost::true_type {};

template <>
struct algebra_dispatcher<DataVector> {
  using algebra_type = ::detail::vector_impl_algebra;
};

template <>
struct algebra_dispatcher<ComplexDataVector> {
  using algebra_type = ::detail::vector_impl_algebra;
};

template <>
struct algebra_dispatcher<ModalVector> {
  using algebra_type = ::detail::vector_impl_algebra;
};

template <>
struct algebra_dispatcher<ComplexModalVector> {
  using algebra_type = ::detail::vector_impl_algebra;
};

template <size_t Size>
struct algebra_dispatcher<std::array<DataVector, Size>> {
  using algebra_type = ::detail::vector_impl_array_algebra;
};

template <size_t Size>
struct algebra_dispatcher<std::array<ComplexDataVector, Size>> {
  using algebra_type = ::detail::vector_impl_array_algebra;
};

template <size_t Size>
struct algebra_dispatcher<std::array<ModalVector, Size>> {
  using algebra_type = ::detail::vector_impl_array_algebra;
};

template <size_t Size>
struct algebra_dispatcher<std::array<ComplexModalVector, Size>> {
  using algebra_type = ::detail::vector_impl_array_algebra;
};
}  // namespace odeint
}  // namespace numeric
}  // namespace boost

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief For ODE integration, we suggest using the boost libraries whenever
 * possible.
 *
 * \details Here we describe briefly the suggested setup to use a boost ODE
 * integrator in SpECTRE. The boost utilities have a number of more elaborate
 * features, including features that may result in simpler function calls than
 * the specific recipes below. For full documentation, see the Boost odeint
 * documentation:
 * https://www.boost.org/doc/libs/1_72_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/
 *
 * The integration methods can be used largely as in the boost documentation. We
 * summarize the salient points necessary to get an example running. The main
 * steps in using a boost integrator are:
 * - define the system
 * - construct the stepper
 * - initialize (might be performed implicitly, depending on the stepper)
 * - perform steps
 * - (compute dense output, for dense-output steppers)
 *
 * For most cases the Dormand-Prince fifth order controlled, dense-output
 * stepper is recommended. That stepper is used in the section "SpECTRE vectors
 * or `std::array` thereof in boost integrators" below.
 *
 * #### Fundamental types or `std::array` thereof in fixed-step integrators
 *
 * Let us consider a simple oscillator system, which we'll declare as a lambda:
 *
 * \snippet Test_OdeIntegration.cpp explicit_fundamental_array_system
 *
 * Note that the first argument is a `const` lvalue reference to the state type,
 * `std::array<double, 2>`, the second argument is an lvalue reference for the
 * time derivatives that is written to by the system function, and the final
 * argument is the current time.
 *
 * The construction and initialization of the stepper is simple:
 *
 * \snippet Test_OdeIntegration.cpp explicit_fundamental_stepper_construction
 *
 * Finally, we can perform the steps and examine the output,
 *
 * \snippet Test_OdeIntegration.cpp explicit_fundamental_stepper_use
 *
 * #### Fundamental types or `std::array` thereof in Dense-output integrators
 *
 * The dense-output and controlled-step-size ODE integrators in boost comply
 * with a somewhat different interface, as significantly more state information
 * must be managed. The result is a somewhat simpler, but more opaque, interface
 * to the user code.
 *
 * Once again, we start by constructing the system we'd like to integrate,
 *
 * \snippet Test_OdeIntegration.cpp dense_output_fundamental_system
 *
 * The constructor for dense steppers takes optional arguments for tolerance,
 * and dense steppers need to be initialized.
 *
 * \snippet Test_OdeIntegration.cpp dense_output_fundamental_construction
 *
 * It is important to take note of the tolerance arguments, as the defaults are
 * `1.0e-6`, which is often looser than we want for calculations in SpECTRE.
 *
 * We then perform the step supplying the system function, and can retrieve the
 * dense output state with `calc_state`, which returns by reference by the
 * second argument.
 *
 * \snippet Test_OdeIntegration.cpp dense_output_fundamental_stepper_use
 *
 * #### SpECTRE vectors or `std::array` thereof in boost integrators
 *
 * The additional template specializations present in the `OdeIntegration.hpp`
 * file ensure that the boost ODE methods work transparently for SpECTRE vectors
 * as well. We'll run through a brief example to emphasize the functionality.
 *
 * \snippet Test_OdeIntegration.cpp dense_output_vector_stepper
 */
namespace OdeIntegration {}

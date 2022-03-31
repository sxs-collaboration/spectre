// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/DataStructures/MathWrapper.hpp"

#include <type_traits>

#include "Utilities/GenerateInstantiations.hpp"

namespace TestHelpers::MathWrapper::detail {
template <typename T>
void do_assignment(const ::MathWrapper<T>& dest,
                   const ::MathWrapper<const T>& source) {
  *dest = *source;
}

template <typename T>
void do_multiply(const ::MathWrapper<T>& dest,
                 const typename ::MathWrapper<T>::scalar_type& scalar,
                 const ::MathWrapper<const T>& source) {
  *dest = scalar * *source;
}

#define MATH_WRAPPER_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template void do_assignment(                                            \
      const ::MathWrapper<MATH_WRAPPER_TYPE(data)>& dest,                 \
      const ::MathWrapper<const MATH_WRAPPER_TYPE(data)>& source);        \
  template void do_multiply(                                              \
      const ::MathWrapper<MATH_WRAPPER_TYPE(data)>& dest,                 \
      const typename ::MathWrapper<MATH_WRAPPER_TYPE(data)>::scalar_type& \
          scalar,                                                         \
      const ::MathWrapper<const MATH_WRAPPER_TYPE(data)>& source);

// [MATH_WRAPPER_TYPES_instantiate]
GENERATE_INSTANTIATIONS(INSTANTIATE, (MATH_WRAPPER_TYPES))
// [MATH_WRAPPER_TYPES_instantiate]

#undef INSTANTIATE
#undef MATH_WRAPPER_TYPE
}  // namespace TestHelpers::MathWrapper::detail

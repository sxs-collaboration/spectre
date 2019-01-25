// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines make_with_value

#pragma once

#include <array>
#include <type_traits>

#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \ingroup DataStructuresGroup
/// Implementations of make_with_value.
namespace MakeWithValueImpls {
template <typename R, typename T>
struct MakeWithValueImpl {
  template <typename ValueType>
  static SPECTRE_ALWAYS_INLINE R apply(const T& input,
                                       ValueType value) noexcept;
};
}  // namespace MakeWithValueImpls

/// \ingroup DataStructuresGroup
/// \brief Given an object of type `T`, create an object of type `R` whose
/// elements are initialized to `value`.
///
/// \details This function is useful in function templates in order to
/// initialize the return type of a function template with `value` for functions
/// that can be called either at a single grid-point or to fill a data structure
/// at the same set of grid-points as the `input`

/// \tparam ValueType The type of `value`. For many containers, this will be
/// `double`, but when making a `ComplexDataVector` with value,
/// `std::complex<double>` is appropriate.
///
/// \see MakeWithValueImpls
template <typename R, typename T, typename ValueType>
SPECTRE_ALWAYS_INLINE std::remove_const_t<R> make_with_value(
    const T& input, const ValueType& value) noexcept {
  return MakeWithValueImpls::MakeWithValueImpl<std::remove_const_t<R>,
                                               T>::apply(input, value);
}

namespace MakeWithValueImpls {
/// \brief Returns a double initialized to `value` (`input` is ignored)
template <typename T>
struct MakeWithValueImpl<double, T> {
  static SPECTRE_ALWAYS_INLINE double apply(const T& /* input */,
                                            const double value) noexcept {
    return value;
  }
};

template <typename T>
struct MakeWithValueImpl<std::complex<double>, T> {
  static SPECTRE_ALWAYS_INLINE std::complex<double> apply(
      const T& /* input */, const std::complex<double> value) noexcept {
    return value;
  }
};

/// \brief Makes a `std::array`; each element of the `std::array`
/// must be `make_with_value`-creatable from a `InputType`.
template <size_t Size, typename T, typename InputType>
struct MakeWithValueImpl<std::array<T, Size>, InputType> {
  template <typename ValueType>
  static SPECTRE_ALWAYS_INLINE std::array<T, Size> apply(
      const InputType& input, const ValueType value) noexcept {
    return make_array<Size>(make_with_value<T>(input, value));
  }
};

/// \brief Makes a `TaggedTuple`; each element of the `TaggedTuple`
/// must be `make_with_value`-creatable from a `T`.
template <typename... Tags, typename T>
struct MakeWithValueImpl<tuples::TaggedTuple<Tags...>, T> {
  template <typename ValueType>
  static SPECTRE_ALWAYS_INLINE tuples::TaggedTuple<Tags...> apply(
      const T& input, const ValueType value) noexcept {
    return tuples::TaggedTuple<Tags...>(
        make_with_value<typename Tags::type>(input, value)...);
  }
};

}  // namespace MakeWithValueImpls

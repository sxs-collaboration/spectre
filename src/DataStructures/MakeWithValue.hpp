// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines make_with_value

#pragma once

#include "Utilities/ForceInline.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \ingroup DataStructuresGroup
/// Implementations of make_with_value.
namespace MakeWithValueImpls {
template <typename R, typename T>
struct MakeWithValueImpl {
  static SPECTRE_ALWAYS_INLINE R apply(const T& input, double value);
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
///
/// \see MakeWithValueImpls
template <typename R, typename T>
SPECTRE_ALWAYS_INLINE R make_with_value(const T& input, double value) {
  return MakeWithValueImpls::MakeWithValueImpl<R, T>::apply(input, value);
}

namespace MakeWithValueImpls {
/// \brief Returns a double initialized to `value` (`input` is ignored)
template <>
SPECTRE_ALWAYS_INLINE double MakeWithValueImpl<double, double>::apply(
    const double& /* input */, const double value) {
  return value;
}

/// \brief Makes a `TaggedTuple`; each element of the `TaggedTuple`
/// must be `make_with_value`-creatable from a `T`.
template <typename... Tags, typename T>
struct MakeWithValueImpl<tuples::TaggedTuple<Tags...>, T> {
  static SPECTRE_ALWAYS_INLINE tuples::TaggedTuple<Tags...> apply(
      const T& input, const double value) {
    return tuples::TaggedTuple<Tags...>(
        make_with_value<typename Tags::type>(input, value)...);
  }
};

}  // namespace MakeWithValueImpls

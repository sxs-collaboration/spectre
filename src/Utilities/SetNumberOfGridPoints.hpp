// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// Implementations of \link set_number_of_grid_points \endlink
///
/// Specializations may be trivial, meaning the object is similar to a
/// `double` and doesn't actually represent grid data.  The only
/// requirement for trivial specializations is defining `is_trivial`
/// as `true`.
///
/// Non-trivial specializations must define `is_trivial` to `false`,
/// and must define a static `apply` function with the signature shown
/// in the below example.  Non-trivial specializations must also be
/// accompanied by a specialization of the
/// MakeWithValueImpls::NumberOfPoints struct.  The `apply` function
/// will only be called if the current size is incorrect, so
/// implementations should not check the current size.
///
/// \snippet Test_SetNumberOfGridPoints.cpp SetNumberOfGridPointsImpl
namespace SetNumberOfGridPointsImpls {
/// Default implementation is not defined.
template <typename T, typename = std::nullptr_t>
struct SetNumberOfGridPointsImpl;
}  // namespace SetNumberOfGridPointsImpls

/// \ingroup DataStructuresGroup
/// Change the number of grid points in an object.
///
/// Change the number of points stored in \p result to a given value
/// or to match another object.  If \p pattern is a `size_t` it will
/// be used as the number of points, otherwise it will be interpreted
/// as data on a collection of grid points.
///
/// \see SetNumberOfGridPointsImpls, make_with_value
template <typename T, typename U>
void set_number_of_grid_points(const gsl::not_null<T*> result,
                               const U& pattern) {
  (void)result;
  (void)pattern;
  if constexpr (not SetNumberOfGridPointsImpls::SetNumberOfGridPointsImpl<
                    T>::is_trivial) {
    const size_t size = MakeWithValueImpls::number_of_points(pattern);
    if (UNLIKELY(MakeWithValueImpls::number_of_points(*result) != size)) {
      SetNumberOfGridPointsImpls::SetNumberOfGridPointsImpl<T>::apply(result,
                                                                      size);
    }
  }
}

template <>
struct SetNumberOfGridPointsImpls::SetNumberOfGridPointsImpl<double> {
  static constexpr bool is_trivial = true;
};

template <>
struct SetNumberOfGridPointsImpls::SetNumberOfGridPointsImpl<
    std::complex<double>> {
  static constexpr bool is_trivial = true;
};

template <typename T, size_t N>
struct SetNumberOfGridPointsImpls::SetNumberOfGridPointsImpl<std::array<T, N>> {
  static constexpr bool is_trivial =
      N == 0 or SetNumberOfGridPointsImpl<T>::is_trivial;
  static void apply(const gsl::not_null<std::array<T, N>*> result,
                    const size_t size) {
    for (auto& entry : *result) {
      set_number_of_grid_points(make_not_null(&entry), size);
    }
  }
};

template <typename T>
struct SetNumberOfGridPointsImpls::SetNumberOfGridPointsImpl<std::vector<T>> {
  static constexpr bool is_trivial = SetNumberOfGridPointsImpl<T>::is_trivial;
  static void apply(const gsl::not_null<std::vector<T>*> result,
                    const size_t size) {
    for (auto& entry : *result) {
      set_number_of_grid_points(make_not_null(&entry), size);
    }
  }
};

template <typename... Tags>
struct SetNumberOfGridPointsImpls::SetNumberOfGridPointsImpl<
    tuples::TaggedTuple<Tags...>> {
  static constexpr bool is_trivial =
      (... and SetNumberOfGridPointsImpl<typename Tags::type>::is_trivial);
  static void apply(const gsl::not_null<tuples::TaggedTuple<Tags...>*> result,
                    const size_t size) {
    expand_pack((set_number_of_grid_points(
                     make_not_null(&tuples::get<Tags>(*result)), size),
                 0)...);
  }
};

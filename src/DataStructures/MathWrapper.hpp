// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup DataStructuresGroup
/// A comma-separated list of valid template arguments to MathWrapper.
/// Useful for explicit instantiations.
///
/// \snippet Helpers/DataStructures/MathWrapper.cpp MATH_WRAPPER_TYPES_instantiate
#define MATH_WRAPPER_TYPES \
  double, std::complex<double>, DataVector, ComplexDataVector

/// \ingroup DataStructuresGroup
/// Type-erased data for performing math on.
///
/// This class can only be instantiated with possibly const-qualified
/// types from \ref MATH_WRAPPER_TYPES, which can be assumed to
/// support the mathematical operations of a linear-algebra vector.
/// Instances of this class with those template arguments can be
/// created using overloads of `make_math_wrapper` (passing a `const
/// T&` for const versions and a `gsl::not_null<T*>` for mutable
/// versions).  Other data structures (such as `Variables`) can add
/// additional overloads implemented on top of these basic ones.
///
/// \snippet Test_MathWrapper.cpp MathWrapper
template <typename T>
class MathWrapper {
 private:
  using MutableT = std::remove_const_t<T>;

  static_assert(
      tmpl::list_contains_v<tmpl::list<MATH_WRAPPER_TYPES>, MutableT>);

  template <typename U = T,
            bool IsVector =
                not(std::is_same_v<std::decay_t<T>, double> or
                    std::is_same_v<std::decay_t<T>, std::complex<double>>),
            bool IsConst = std::is_const_v<T>>
  struct Impl {
    using scalar_type = std::remove_const_t<U>;
    T& data;
    Impl(const gsl::not_null<T*> data_in) : data(*data_in) {}
  };

  template <typename U>
  struct Impl<U, true, false> {
    using scalar_type = typename U::value_type;
    mutable T data;
    Impl(const gsl::not_null<T*> data_in) : data(std::move(*data_in)) {}
  };

  template <typename U>
  struct Impl<U, true, true> {
    using scalar_type = typename U::value_type;
    const T data;
    // Need to invoke the move-from-mutable constructor on DataVector, etc.
    Impl(const gsl::not_null<MutableT*> data_in) : data(std::move(*data_in)) {}
  };

  explicit MathWrapper(const gsl::not_null<MutableT*> data) : data_(data) {}

  friend MathWrapper<T> make_math_wrapper(
      tmpl::conditional_t<std::is_const_v<T>, T&, gsl::not_null<T*>>);

 public:
  /// The class's template parameter.
  using value_type = T;
  /// Scalar type for linear-algebra operations.  Either double or
  /// std::complex<double>.
  using scalar_type = typename Impl<>::scalar_type;

  T& operator*() const { return data_.data; }

  MathWrapper(MathWrapper&&) = default;

  MathWrapper() = delete;
  MathWrapper(const MathWrapper&) = delete;
  MathWrapper& operator=(const MathWrapper&) = delete;
  MathWrapper& operator=(MathWrapper&&) = delete;

  /// Convert MathWrapper wrapping a mutable value to one wrapping a
  /// const one.
  ///
  /// These methods will fail to compile if called on a MathWrapper
  /// wrapping a const value.  The `to_const` method is useful because
  /// C++ fails to resolve the implicit conversion in many cases.
  /// @{
  operator MathWrapper<const T>() const;

  MathWrapper<const T> to_const() const {
    return static_cast<MathWrapper<const T>>(*this);
  }
  /// @}

 private:
  Impl<> data_;
};

/// \ingroup DataStructuresGroup
/// A fundamental overload of the MathWrapper construction functions.
///
/// Additional overloads can be implemented in terms of the
/// fundamental overloads.
/// @{
inline MathWrapper<double> make_math_wrapper(
    const gsl::not_null<double*> data) {
  return MathWrapper<double>(data);
}

inline MathWrapper<const double> make_math_wrapper(const double& data) {
  return MathWrapper<const double>(const_cast<double*>(&data));
}

inline MathWrapper<std::complex<double>> make_math_wrapper(
    const gsl::not_null<std::complex<double>*> data) {
  return MathWrapper<std::complex<double>>(data);
}

inline MathWrapper<const std::complex<double>> make_math_wrapper(
    const std::complex<double>& data) {
  return MathWrapper<const std::complex<double>>(
      const_cast<std::complex<double>*>(&data));
}

inline MathWrapper<DataVector> make_math_wrapper(
    const gsl::not_null<DataVector*> data) {
  DataVector referencing(data->data(), data->size());
  return MathWrapper<DataVector>(&referencing);
}

inline MathWrapper<const DataVector> make_math_wrapper(const DataVector& data) {
  DataVector referencing(const_cast<double*>(data.data()), data.size());
  return MathWrapper<const DataVector>(&referencing);
}

inline MathWrapper<ComplexDataVector> make_math_wrapper(
    const gsl::not_null<ComplexDataVector*> data) {
  ComplexDataVector referencing(data->data(), data->size());
  return MathWrapper<ComplexDataVector>(&referencing);
}

inline MathWrapper<const ComplexDataVector> make_math_wrapper(
    const ComplexDataVector& data) {
  ComplexDataVector referencing(const_cast<std::complex<double>*>(data.data()),
                                data.size());
  return MathWrapper<const ComplexDataVector>(&referencing);
}
/// @}

template <typename T>
MathWrapper<T>::operator MathWrapper<const T>() const {
  return make_math_wrapper(data_.data);
}

template <typename T, size_t N>
auto make_math_wrapper(const gsl::not_null<std::array<T, N>*> array) {
  DataVector referencing(array->data(), array->size());
  return make_math_wrapper(&referencing);
}

template <typename T, size_t N>
auto make_math_wrapper(const std::array<T, N>& array) {
  const DataVector referencing(const_cast<double*>(array.data()), array.size());
  return make_math_wrapper(referencing);
}

/// \ingroup DataStructuresGroup
/// The `value_type` for a MathWrapper wrapping `T`.
template <typename T>
using math_wrapper_type = typename decltype(make_math_wrapper(
    std::declval<tmpl::conditional_t<std::is_const_v<T>, const T&,
                                     gsl::not_null<T*>>>()))::value_type;

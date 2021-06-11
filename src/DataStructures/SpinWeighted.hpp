// Distributed under the MIT License.
// See LICENSE.txt for details

#pragma once

#include "DataStructures/ComplexDataVector.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

/// @{
/*!
 * \ingroup DataStructuresGroup
 * \brief Make a spin-weighted type `T` with spin-weight `Spin`. Mathematical
 * operators are restricted to addition, subtraction, multiplication and
 * division, with spin-weights checked for validity.
 *
 * \details For a spin-weighted object, we limit operations to those valid for a
 * pair of spin-weighted quantities - i.e. addition only makes sense when the
 * two summands possess the same spin weight, and multiplication (or division)
 * result in a summed (or subtracted) spin weight.
 */
template <typename T, int Spin, bool is_vector = is_derived_of_vector_impl_v<T>>
struct SpinWeighted;

template <typename T, int Spin>
struct SpinWeighted<T, Spin, false> {
  using value_type = T;
  constexpr static int spin = Spin;

  SpinWeighted() = default;
  SpinWeighted(const SpinWeighted&) noexcept = default;
  SpinWeighted(SpinWeighted&&) noexcept = default;
  SpinWeighted& operator=(const SpinWeighted&) noexcept = default;
  SpinWeighted& operator=(SpinWeighted&&) noexcept = default;
  ~SpinWeighted() noexcept = default;

  // clang-tidy asks that these be marked explicit, but we actually do not want
  // them explicit for use in the math operations below.
  template <typename Rhs>
  SpinWeighted(const SpinWeighted<Rhs, Spin>& rhs) noexcept  // NOLINT
      : data_{rhs.data()} {}

  template <typename Rhs>
  SpinWeighted(SpinWeighted<Rhs, Spin>&& rhs) noexcept  // NOLINT
      : data_{std::move(rhs.data())} {}

  SpinWeighted(const T& rhs) noexcept : data_{rhs} {}        // NOLINT
  SpinWeighted(T&& rhs) noexcept : data_{std::move(rhs)} {}  // NOLINT
  explicit SpinWeighted(const size_t size) noexcept : data_{size} {}
  template <typename U>
  SpinWeighted(const size_t size, const U& val) noexcept : data_{size, val} {}

  template <typename Rhs>
  SpinWeighted& operator=(const SpinWeighted<Rhs, Spin>& rhs) noexcept {
    data_ = rhs.data();
    return *this;
  }

  template <typename Rhs>
  SpinWeighted& operator=(SpinWeighted<Rhs, Spin>&& rhs) noexcept {
    data_ = std::move(rhs.data());
    return *this;
  }

  SpinWeighted& operator=(const T& rhs) noexcept {
    data_ = rhs;
    return *this;
  }
  SpinWeighted& operator=(T&& rhs) noexcept {
    data_ = std::move(rhs);
    return *this;
  }

  template <typename Rhs>
  auto& operator+=(const SpinWeighted<Rhs, Spin>& rhs) noexcept {
    data_ += rhs.data();
    return *this;
  }

  auto& operator+=(const T& rhs) noexcept {
    data_ += rhs;
    return *this;
  }

  template <typename Rhs>
  auto& operator-=(const SpinWeighted<Rhs, Spin>& rhs) noexcept {
    data_ -= rhs.data();
    return *this;
  }

  auto& operator-=(const T& rhs) noexcept {
    data_ -= rhs;
    return *this;
  }

  T& data() noexcept { return data_; }
  const T& data() const noexcept { return data_; }

 size_t size() const noexcept { return data_.size(); }

  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  T data_;
};

template <typename T, int Spin>
struct SpinWeighted<T, Spin, true> {
  using value_type = T;
  constexpr static int spin = Spin;

  void set_data_ref(const gsl::not_null<T*> rhs) noexcept {
    data_.set_data_ref(rhs);
  }

  // needed for invoking the check in `Variables.hpp` that ensures that
  // default-constructed `Variables` are never used.
  void set_data_ref(const std::nullptr_t null, const size_t size) noexcept {
    data_.set_data_ref(null, size);
  }

  void set_data_ref(const gsl::not_null<SpinWeighted<T, spin>*> rhs) noexcept {
    data_.set_data_ref(make_not_null(&(rhs->data_)));
  }

  template <typename ValueType>
  void set_data_ref(ValueType* const start, const size_t set_size) noexcept {
    data_.set_data_ref(start, set_size);
  }

  void destructive_resize(const size_t new_size) noexcept {
    data_.destructive_resize(new_size);
  }

  SpinWeighted() = default;
  SpinWeighted(const SpinWeighted&) noexcept = default;
  SpinWeighted(SpinWeighted&&) noexcept = default;
  SpinWeighted& operator=(const SpinWeighted&) noexcept = default;
  SpinWeighted& operator=(SpinWeighted&&) noexcept = default;
  ~SpinWeighted() noexcept = default;

  // clang-tidy asks that these be marked explicit, but we actually do not want
  // them explicit for use in the math operations below.
  template <typename Rhs>
  SpinWeighted(const SpinWeighted<Rhs, Spin>& rhs) noexcept  // NOLINT
      : data_{rhs.data()} {}

  template <typename Rhs>
  SpinWeighted(SpinWeighted<Rhs, Spin>&& rhs) noexcept  // NOLINT
      : data_{std::move(rhs.data())} {}

  SpinWeighted(const T& rhs) noexcept : data_{rhs} {}        // NOLINT
  SpinWeighted(T&& rhs) noexcept : data_{std::move(rhs)} {}  // NOLINT
  explicit SpinWeighted(const size_t size) noexcept : data_{size} {}
  template <typename U>
  SpinWeighted(const size_t size, const U& val) noexcept : data_{size, val} {}

  template <typename Rhs>
  SpinWeighted& operator=(const SpinWeighted<Rhs, Spin>& rhs) noexcept {
    data_ = rhs.data();
    return *this;
  }

  template <typename Rhs>
  SpinWeighted& operator=(SpinWeighted<Rhs, Spin>&& rhs) noexcept {
    data_ = std::move(rhs.data());
    return *this;
  }

  SpinWeighted& operator=(const T& rhs) noexcept {
    data_ = rhs;
    return *this;
  }
  SpinWeighted& operator=(T&& rhs) noexcept {
    data_ = std::move(rhs);
    return *this;
  }

  template <typename Rhs>
  auto& operator+=(const SpinWeighted<Rhs, Spin>& rhs) noexcept {
    data_ += rhs.data();
    return *this;
  }

  auto& operator+=(const T& rhs) noexcept {
    data_ += rhs;
    return *this;
  }

  template <typename Rhs>
  auto& operator-=(const SpinWeighted<Rhs, Spin>& rhs) noexcept {
    data_ -= rhs.data();
    return *this;
  }

  auto& operator-=(const T& rhs) noexcept {
    data_ -= rhs;
    return *this;
  }

  T& data() noexcept { return data_; }
  const T& data() const noexcept { return data_; }

  size_t size() const noexcept { return data_.size(); }

  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  T data_;
};
/// @}

template <typename T, int Spin>
void SpinWeighted<T, Spin, true>::pup(PUP::er& p) noexcept {
  p | data_;
}

template <typename T, int Spin>
void SpinWeighted<T, Spin, false>::pup(PUP::er& p) noexcept {
  p | data_;
}

/// @{
/// \ingroup TypeTraitsGroup
/// \ingroup DataStructuresGroup
/// This is a `std::true_type` if the provided type is a `SpinWeighted` of any
/// type and spin, otherwise is a `std::false_type`.
template <typename T>
struct is_any_spin_weighted : std::false_type {};

template <typename T, int S>
struct is_any_spin_weighted<SpinWeighted<T, S>> : std::true_type {};
/// @}

template <typename T>
constexpr bool is_any_spin_weighted_v = is_any_spin_weighted<T>::value;

/// @{
/// \ingroup TypeTraitsGroup
/// \ingroup DataStructuresGroup
/// This is a `std::true_type` if the provided type `T` is a `SpinWeighted` of
/// `InternalType` and any spin, otherwise is a `std::false_type`.
template <typename InternalType, typename T>
struct is_spin_weighted_of : std::false_type {};

template <typename InternalType, int S>
struct is_spin_weighted_of<InternalType, SpinWeighted<InternalType, S>>
    : std::true_type {};
/// @}

template <typename InternalType, typename T>
constexpr bool is_spin_weighted_of_v =
    is_spin_weighted_of<InternalType, T>::value;

/// @{
/// \ingroup TypeTraitsGroup
/// \ingroup DataStructuresGroup
/// This is a `std::true_type` if the provided type `T1` is a `SpinWeighted` and
/// `T2` is a `SpinWeighted`, and both have the same internal type, but any
/// combination of spin weights.
template <typename T1, typename T2>
struct is_spin_weighted_of_same_type : std::false_type {};

template <typename T, int Spin1, int Spin2>
struct is_spin_weighted_of_same_type<SpinWeighted<T, Spin1>,
                                     SpinWeighted<T, Spin2>> : std::true_type {
};
/// @}

template <typename T1, typename T2>
constexpr bool is_spin_weighted_of_same_type_v =
    is_spin_weighted_of_same_type<T1, T2>::value;

/// @{
/// \brief Add two spin-weighted quantities if the types are compatible and
/// spins are the same. Un-weighted quantities are assumed to be spin 0.
// These overloads are designed to allow SpinWeighted to wrap Blaze expression
// templates to ensure efficient math operations, necessitating the
// `decltype(declval<T>() ...` syntax
template <typename T1, typename T2, int Spin>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T1>() + std::declval<T2>()), Spin>
    operator+(const SpinWeighted<T1, Spin>& lhs,
              const SpinWeighted<T2, Spin>& rhs) noexcept {
  return {lhs.data() + rhs.data()};
}
template <typename T>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T>() + std::declval<T>()), 0>
    operator+(const SpinWeighted<T, 0>& lhs, const T& rhs) noexcept {
  return {lhs.data() + rhs};
}
template <typename T>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<T>() + std::declval<get_vector_element_type_t<T>>()),
    0>
operator+(const SpinWeighted<T, 0>& lhs,
          const get_vector_element_type_t<T>& rhs) noexcept {
  return {lhs.data() + rhs};
}
template <typename T>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T>() + std::declval<T>()), 0>
    operator+(const T& lhs, const SpinWeighted<T, 0>& rhs) noexcept {
  return {lhs + rhs.data()};
}
template <typename T>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<get_vector_element_type_t<T>>() + std::declval<T>()),
    0>
operator+(const get_vector_element_type_t<T>& lhs,
          const SpinWeighted<T, 0>& rhs) noexcept {
  return {lhs + rhs.data()};
}
/// @}

/// @{
/// \brief Subtract two spin-weighted quantities if the types are compatible and
/// spins are the same. Un-weighted quantities are assumed to be spin 0.
// These overloads are designed to allow SpinWeighted to wrap Blaze expression
// templates to ensure efficient math operations, necessitating the
// `decltype(declval<T>() ...` syntax
template <typename T1, typename T2, int Spin>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T1>() - std::declval<T2>()), Spin>
    operator-(const SpinWeighted<T1, Spin>& lhs,
              const SpinWeighted<T2, Spin>& rhs) noexcept {
  return {lhs.data() - rhs.data()};
}
template <typename T>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T>() - std::declval<T>()), 0>
    operator-(const SpinWeighted<T, 0>& lhs, const T& rhs) noexcept {
  return {lhs.data() - rhs};
}
template <typename T>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<T>() - std::declval<get_vector_element_type_t<T>>()),
    0>
operator-(const SpinWeighted<T, 0>& lhs,
          const get_vector_element_type_t<T>& rhs) noexcept {
  return {lhs.data() - rhs};
}
template <typename T>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T>() - std::declval<T>()), 0>
    operator-(const T& lhs, const SpinWeighted<T, 0>& rhs) noexcept {
  return {lhs - rhs.data()};
}
template <typename T>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<get_vector_element_type_t<T>>() - std::declval<T>()),
    0>
operator-(const get_vector_element_type_t<T>& lhs,
          const SpinWeighted<T, 0>& rhs) noexcept {
  return {lhs - rhs.data()};
}
/// @}

/// Negation operator preserves spin
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<decltype(-std::declval<T>()), Spin>
operator-(const SpinWeighted<T, Spin>& operand) noexcept {
  return {-operand.data()};
}

/// Unary `+` operator preserves spin
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<decltype(+std::declval<T>()), Spin>
operator+(const SpinWeighted<T, Spin>& operand) noexcept {
  return {+operand.data()};
}

/// @{
/// \brief Multiply two spin-weighted quantities if the types are compatible and
/// add the spins. Un-weighted quantities are assumed to be spin 0.
// These overloads are designed to allow SpinWeighted to wrap Blaze expression
// templates to ensure efficient math operations, necessitating the
// `decltype(declval<T>() ...` syntax
template <typename T1, typename T2, int Spin1, int Spin2>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<T1>() * std::declval<T2>()), Spin1 + Spin2>
operator*(const SpinWeighted<T1, Spin1>& lhs,
          const SpinWeighted<T2, Spin2>& rhs) noexcept {
  return {lhs.data() * rhs.data()};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T>() * std::declval<T>()), Spin>
    operator*(const SpinWeighted<T, Spin>& lhs, const T& rhs) noexcept {
  return {lhs.data() * rhs};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<T>() * std::declval<get_vector_element_type_t<T>>()),
    Spin>
operator*(const SpinWeighted<T, Spin>& lhs,
          const get_vector_element_type_t<T>& rhs) noexcept {
  return {lhs.data() * rhs};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T>() * std::declval<T>()), Spin>
    operator*(const T& lhs, const SpinWeighted<T, Spin>& rhs) noexcept {
  return {lhs * rhs.data()};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<get_vector_element_type_t<T>>() * std::declval<T>()),
    Spin>
operator*(const get_vector_element_type_t<T>& lhs,
          const SpinWeighted<T, Spin>& rhs) noexcept {
  return {lhs * rhs.data()};
}
/// @}

/// @{
/// \brief Divide two spin-weighted quantities if the types are compatible and
/// subtract the spins. Un-weighted quantities are assumed to be spin 0.
// These overloads are designed to allow SpinWeighted to wrap Blaze expression
// templates to ensure efficient math operations, necessitating the
// `decltype(declval<T>() ...` syntax
template <typename T1, typename T2, int Spin1, int Spin2>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<T1>() / std::declval<T2>()), Spin1 - Spin2>
operator/(const SpinWeighted<T1, Spin1>& lhs,
          const SpinWeighted<T2, Spin2>& rhs) noexcept {
  return {lhs.data() / rhs.data()};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T>() / std::declval<T>()), Spin>
    operator/(const SpinWeighted<T, Spin>& lhs, const T& rhs) noexcept {
  return {lhs.data() / rhs};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<T>() / std::declval<get_vector_element_type_t<T>>()),
    Spin>
operator/(const SpinWeighted<T, Spin>& lhs,
          const get_vector_element_type_t<T>& rhs) noexcept {
  return {lhs.data() / rhs};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE
    SpinWeighted<decltype(std::declval<T>() / std::declval<T>()), -Spin>
    operator/(const T& lhs, const SpinWeighted<T, Spin>& rhs) noexcept {
  return {lhs / rhs.data()};
}
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<
    decltype(std::declval<get_vector_element_type_t<T>>() / std::declval<T>()),
    -Spin>
operator/(const get_vector_element_type_t<T>& lhs,
          const SpinWeighted<T, Spin>& rhs) noexcept {
  return {lhs / rhs.data()};
}
/// @}

/// conjugate the spin-weighted quantity, inverting the spin
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<decltype(conj(std::declval<T>())), -Spin>
conj(const SpinWeighted<T, Spin>& value) noexcept {
  return {conj(value.data())};
}

/// Take the exponential of the spin-weighted quantity; only valid for
/// spin-weight = 0
template <typename T>
SPECTRE_ALWAYS_INLINE SpinWeighted<decltype(exp(std::declval<T>())), 0> exp(
    const SpinWeighted<T, 0>& value) noexcept {
  return {exp(value.data())};
}

/// Take the square-root of the spin-weighted quantity; only valid for
/// spin-weight = 0
template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<decltype(sqrt(std::declval<T>())), 0> sqrt(
    const SpinWeighted<T, Spin>& value) noexcept {
  return {sqrt(value.data())};
}

/// @{
/// \brief Test equivalence of spin-weighted quantities if the types are
/// compatible and spins are the same. Un-weighted quantities are assumed to
/// be spin 0.
template <typename T1, typename T2, int Spin>
SPECTRE_ALWAYS_INLINE bool operator==(
    const SpinWeighted<T1, Spin>& lhs,
    const SpinWeighted<T2, Spin>& rhs) noexcept {
  return lhs.data() == rhs.data();
}
template <typename T>
SPECTRE_ALWAYS_INLINE bool operator==(const SpinWeighted<T, 0>& lhs,
                                      const T& rhs) noexcept {
  return lhs.data() == rhs;
}
template <typename T>
SPECTRE_ALWAYS_INLINE bool operator==(const T& lhs,
                                      const SpinWeighted<T, 0>& rhs) noexcept {
  return lhs == rhs.data();
}
/// @}

/// @{
/// \brief Test inequivalence of spin-weighted quantities if the types are
/// compatible and spins are the same. Un-weighted quantities are assumed to be
/// spin 0.
template <typename T1, typename T2, int Spin>
SPECTRE_ALWAYS_INLINE bool operator!=(
    const SpinWeighted<T1, Spin>& lhs,
    const SpinWeighted<T2, Spin>& rhs) noexcept {
  return not(lhs == rhs);
}
template <typename T>
SPECTRE_ALWAYS_INLINE bool operator!=(const SpinWeighted<T, 0>& lhs,
                                      const T& rhs) noexcept {
  return not(lhs == rhs);
}
template <typename T>
SPECTRE_ALWAYS_INLINE bool operator!=(const T& lhs,
                                      const SpinWeighted<T, 0>& rhs) noexcept {
  return not(lhs == rhs);
}
/// @}

/// \ingroup DataStructuresGroup
/// Make the input `view` a `const` view of the const data `spin_weighted`, at
/// offset `offset` and length `extent`.
///
/// \warning This DOES modify the (const) input `view`. The reason `view` is
/// taken by const pointer is to try to insist that the object to be a `const`
/// view is actually const. Of course, there are ways of subverting this
/// intended functionality and editing the data pointed into by `view` after
/// this function is called; doing so is highly discouraged and results in
/// undefined behavior.
template <typename SpinWeightedType,
          Requires<is_any_spin_weighted_v<SpinWeightedType> and
                   is_derived_of_vector_impl_v<
                       typename SpinWeightedType::value_type>> = nullptr>
void make_const_view(const gsl::not_null<const SpinWeightedType*> view,
                     const SpinWeightedType& spin_weighted, const size_t offset,
                     const size_t extent) noexcept {
  const_cast<SpinWeightedType*>(view.get())  // NOLINT
      ->set_data_ref(const_cast<             // NOLINT
                         typename SpinWeightedType::value_type::value_type*>(
                         spin_weighted.data().data()) +  // NOLINT
                         offset,
                     extent);
}

/// Stream operator simply forwards
template <typename T, int Spin>
std::ostream& operator<<(std::ostream& os,
                         const SpinWeighted<T, Spin>& d) noexcept {
  return os << d.data();
}

namespace MakeWithValueImpls {
template <int Spin, typename SpinWeightedType>
struct NumberOfPoints<SpinWeighted<SpinWeightedType, Spin>> {
  static SPECTRE_ALWAYS_INLINE size_t
  apply(const SpinWeighted<SpinWeightedType, Spin>& input) noexcept {
    return number_of_points(input.data());
  }
};

template <int Spin, typename SpinWeightedType>
struct MakeWithSize<SpinWeighted<SpinWeightedType, Spin>> {
  template <typename ValueType>
  static SPECTRE_ALWAYS_INLINE SpinWeighted<SpinWeightedType, Spin> apply(
      const size_t size, const ValueType value) noexcept {
    return SpinWeighted<SpinWeightedType, Spin>{
        make_with_value<SpinWeightedType>(size, value)};
  }
};
}  // namespace MakeWithValueImpls

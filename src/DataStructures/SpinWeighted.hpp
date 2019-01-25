// Distributed under the MIT License.
// See LICENSE.txt for details

#pragma once

#include "DataStructures/ComplexDataVector.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

// @{
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

  T& data() noexcept { return data_; }
  const T& data() const noexcept { return data_; }

 private:
  T data_;
};

template <typename T, int Spin>
struct SpinWeighted<T, Spin, true> {
  using value_type = T;
  constexpr static int spin = Spin;

  void set_data_ref(gsl::not_null<T*> rhs) noexcept { data_.set_data_ref(rhs); }

  template <typename ValueType>
  void set_data_ref(ValueType* const start, const size_t set_size) noexcept {
    data_.set_data_ref(start, set_size);
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

  T& data() noexcept { return data_; }
  const T& data() const noexcept { return data_; }

 private:
  T data_;
};
// @}

// @{
/// \ingroup TypeTraitsGroup
/// \ingroup DataStructuresGroup
/// This is a `std::true_type` if the provided type is a `SpinWeighted` of any
/// type and spin, otherwise is a `std::false_type`.
template <typename T>
struct is_any_spin_weighted : std::false_type {};

template <typename T, int S>
struct is_any_spin_weighted<SpinWeighted<T, S>> : std::true_type {};
// @}

template <typename T>
constexpr bool is_any_spin_weighted_v = is_any_spin_weighted<T>::value;

// @{
/// \ingroup TypeTraitsGroup
/// \ingroup DataStructuresGroup
/// This is a `std::true_type` if the provided type `T` is a `SpinWeighted` of
/// `InternalType` and any spin, otherwise is a `std::false_type`.
template <typename InternalType, typename T>
struct is_spin_weighted_of : std::false_type {};

template <typename InternalType, int S>
struct is_spin_weighted_of<InternalType, SpinWeighted<InternalType, S>>
    : std::true_type {};
// @}

template <typename InternalType, typename T>
constexpr bool is_spin_weighted_of_v =
    is_spin_weighted_of<InternalType, T>::value;

// @{
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
// @}

template <typename T1, typename T2>
constexpr bool is_spin_weighted_of_same_type_v =
    is_spin_weighted_of_same_type<T1, T2>::value;

// {@
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
// @}

// @{
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
// @}

// @{
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
// @}

// @{
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
// @}

template <typename T, int Spin>
SPECTRE_ALWAYS_INLINE SpinWeighted<decltype(conj(std::declval<T>())), -Spin>
conj(const SpinWeighted<T, Spin> value) noexcept {
  return {conj(value.data())};
}

// @{
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
// @}

// @{
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
// @}

namespace MakeWithValueImpls {
template <int Spin1, int Spin2, typename SpinWeightedType1,
          typename SpinWeightedType2>
struct MakeWithValueImpl<SpinWeighted<SpinWeightedType1, Spin1>,
                         SpinWeighted<SpinWeightedType2, Spin2>> {
  template <typename ValueType>
  static SPECTRE_ALWAYS_INLINE SpinWeighted<SpinWeightedType1, Spin1> apply(
      const SpinWeighted<SpinWeightedType2, Spin2>& input,
      const ValueType value) noexcept {
    return SpinWeighted<SpinWeightedType1, Spin1>{
        make_with_value<SpinWeightedType1>(input.data(), value)};
  }
};

template <int Spin, typename SpinWeightedType, typename MakeWithType>
struct MakeWithValueImpl<SpinWeighted<SpinWeightedType, Spin>, MakeWithType> {
  template <typename ValueType>
  static SPECTRE_ALWAYS_INLINE SpinWeighted<SpinWeightedType, Spin> apply(
      const MakeWithType& input, const ValueType value) noexcept {
    return SpinWeighted<SpinWeightedType, Spin>{
        make_with_value<SpinWeightedType>(input, value)};
  }
};
}  // namespace MakeWithValueImpls

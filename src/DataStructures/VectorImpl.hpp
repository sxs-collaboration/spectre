// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>  // for std::reference_wrapper
#include <initializer_list>
#include <memory>
#include <ostream>
#include <pup.h>
#include <type_traits>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeWithValue.hpp" // IWYU pragma: keep
#include "Utilities/PointerVector.hpp" // IWYU pragma: keep
#include "Utilities/PrintHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdArrayHelpers.hpp"

// IWYU doesn't like that we want PointerVector.hpp to expose Blaze and also
// have VectorImpl.hpp to expose PointerVector.hpp without including Blaze
// directly in VectorImpl.hpp
//
// IWYU pragma: no_include <blaze/math/dense/DenseVector.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecAddExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecDivExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecMultExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecSubExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecMapExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecScalarDivExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecScalarMultExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DenseVector.h>
// IWYU pragma: no_include <blaze/math/expressions/Vector.h>
// IWYU pragma: no_include <blaze/math/typetraits/IsVector.h>
// IWYU pragma: no_include <blaze/math/expressions/Forward.h>
// IWYU pragma: no_include <blaze/math/AlignmentFlag.h>
// IWYU pragma: no_include <blaze/math/PaddingFlag.h>
// IWYU pragma: no_include <blaze/math/traits/AddTrait.h>
// IWYU pragma: no_include <blaze/math/traits/DivTrait.h>
// IWYU pragma: no_include <blaze/math/traits/MultTrait.h>
// IWYU pragma: no_include <blaze/math/traits/SubTrait.h>
// IWYU pragma: no_include <blaze/system/TransposeFlag.h>
#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
// IWYU pragma: no_include <blaze/math/traits/UnaryMapTrait.h>
// IWYU pragma: no_include <blaze/math/traits/BinaryMapTrait.h>
#else
// IWYU pragma: no_include <blaze/math/traits/MapTrait.h>
#endif  // ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
// IWYU pragma: no_include <blaze/math/typetraits/TransposeFlag.h>

// IWYU pragma: no_forward_declare blaze::DenseVector
#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
// IWYU pragma: no_forward_declare blaze::UnaryMapTrait
// IWYU pragma: no_forward_declare blaze::BinaryMapTrait
#else
// IWYU pragma: no_forward_declare blaze::MapTrait
#endif  // ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
// IWYU pragma: no_forward_declare blaze::IsVector
// IWYU pragma: no_forward_declare blaze::TransposeFlag

/*!
 * \ingroup DataStructuresGroup
 * \brief Base class template for various DataVector and related types
 *
 * \details The `VectorImpl` class is the generic parent class for vectors
 * representing collections of related function values, such as `DataVector`s
 * for contiguous data over a computational domain.
 *
 * The `VectorImpl` does not itself define any particular mathematical
 * operations on the contained values. The `VectorImpl` template class and the
 * macros defined in `VectorImpl.hpp` assist in the construction of various
 * derived classes supporting a chosen set of mathematical operations.
 *
 * In addition, the equivalence ooperator `==` is inherited from the underlying
 * `PointerVector` type, and returns true if and only if the size and contents
 * of the two compared vectors are equivalent.
 *
 * Template parameters:
 * - `T` is the underlying stored type, e.g. `double`, `std::complex<double>`,
     `float`, etc.

 * - `VectorType` is the type that should be associated with the VectorImpl
 *    during mathematical computations. In most cases, inherited types should
 *    have themselves as the second template argument, e.g.
 *  ```
 *  class DataVector : VectorImpl<double, DataVector> {
 *  ```
 *  The second template parameter communicates arithmetic type restrictions to
 *  the underlying Blaze framework. For example, if `VectorType` is
 *  `DataVector`, then the underlying architecture will prevent the addition the
 *  vector type with `ModalVector`. Since `DataVector`s and `ModalVector`s
 *  represent data in different spaces, we wish to forbid several operations
 *  between them. This typing tracks the vector type through an expression and
 *  prevents accidental mixing in math expressions.
 *
 * \note
 * - If created with size 0, then  `data()` will return `nullptr`
 * - If `SPECTRE_DEBUG` or `SPECTRE_NAN_INIT` macros are defined, then the
 *   `VectorImpl` is default initialized to `signaling_NaN()`. Otherwise, the
 *   vector is filled with uninitialized memory for performance.
 * - Using an object of this type without explicit initialization is undefined
     behavior unless `SPECTRE_DEBUG` or `SPECTRE_NAN_INIT` is defined (see
     previous point).
 */
template <typename T, typename VectorType>
class VectorImpl
    : public PointerVector<T, blaze::unaligned, blaze::unpadded,
                           blaze::defaultTransposeFlag, VectorType> {
  /// \cond HIDDEN_SYMBOLS
  // the `static_assert` requires a member function around it, as it then
  // gains access to the fully defined type it is a member of, and therefore
  // can test the resulting class.
  static constexpr void private_asserts() noexcept {
    static_assert(
        std::is_nothrow_move_constructible<VectorImpl<T, VectorType>>::value,
        "Missing move semantics");
  }
  /// \endcond
 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using BaseType = PointerVector<T, blaze::unaligned, blaze::unpadded,
                                 blaze::defaultTransposeFlag, VectorType>;
  static constexpr bool transpose_flag = blaze::defaultTransposeFlag;

  using ElementType = T;
  using TransposeType = VectorImpl<T, VectorType>;
  using CompositeType = const VectorImpl<T, VectorType>&;
  using iterator = typename BaseType::Iterator;
  using const_iterator = typename BaseType::ConstIterator;

  using BaseType::operator[];
  using BaseType::begin;
  using BaseType::cbegin;
  using BaseType::cend;
  using BaseType::data;
  using BaseType::end;
  using BaseType::size;

  // @{
  /// Upcast to `BaseType`
  /// \attention
  /// upcast should only be used when implementing a derived vector type, not in
  /// calling code
  const BaseType& operator~() const noexcept {
    return static_cast<const BaseType&>(*this);
  }
  BaseType& operator~() noexcept { return static_cast<BaseType&>(*this); }
  // @}

  // @{
  /// Create with the given size and value. In debug mode, the vector is
  /// initialized to 'NaN' by default.
  ///
  /// - `set_size` set_size number of values
  /// - `value` the value to initialize each element
  explicit VectorImpl(size_t set_size) noexcept
      : owned_data_(set_size > 0 ? static_cast<value_type*>(
                                       malloc(set_size * sizeof(value_type)))
                                 : nullptr,
                    &free) {
#if defined(SPECTRE_DEBUG) || defined(SPECTRE_NAN_INIT)
    std::fill(owned_data_.get(), owned_data_.get() + set_size,
              std::numeric_limits<value_type>::signaling_NaN());
#endif  // SPECTRE_DEBUG
    reset_pointer_vector(set_size);
  }

  VectorImpl(size_t set_size, double value) noexcept
      : owned_data_(set_size > 0 ? static_cast<value_type*>(
                                       malloc(set_size * sizeof(value_type)))
                                 : nullptr,
                    &free) {
    std::fill(owned_data_.get(), owned_data_.get() + set_size, value);
    reset_pointer_vector(set_size);
  }
  // @}

  /// Create a non-owning VectorImpl that points to `start`
  VectorImpl(T* start, size_t set_size) noexcept
      : BaseType(start, set_size), owning_(false) {}

  /// Create from an initializer list of doubles. All elements in the
  /// `std::initializer_list` must have decimal points
  template <class U, Requires<cpp17::is_same_v<U, T>> = nullptr>
  VectorImpl(std::initializer_list<U> list) noexcept
      : owned_data_(list.size() > 0 ? static_cast<value_type*>(malloc(
                                          list.size() * sizeof(value_type)))
                                    : nullptr,
                    &free) {
    // Note: can't use memcpy with an initializer list.
    std::copy(list.begin(), list.end(), owned_data_.get());
    reset_pointer_vector(list.size());
  }

  /// Empty VectorImpl
  VectorImpl() = default;
  /// \cond HIDDEN_SYMBOLS
  ~VectorImpl() = default;

  // clang-tidy: calling a base constructor other than the copy constructor.
  // We reset the base class in reset_pointer_vector after calling its default
  // constructor
  VectorImpl(const VectorImpl<T, VectorType>& rhs) noexcept;

  VectorImpl(VectorImpl<T, VectorType>&& rhs) noexcept;

  // This is a converting constructor. clang-tidy complains that it's not
  // explicit, but we want it to allow conversion.
  // clang-tidy: mark as explicit (we want conversion to VectorImpl type)
  template <
      typename VT, bool VF,
      Requires<cpp17::is_same_v<typename VT::ResultType, VectorType>> = nullptr>
  VectorImpl(const blaze::DenseVector<VT, VF>& expression) noexcept;  // NOLINT

  template <typename VT, bool VF>
  VectorImpl& operator=(const blaze::DenseVector<VT, VF>& expression) noexcept;

  //  declaration for copy constructor
  VectorImpl& operator=(const VectorImpl<T, VectorType>& rhs) noexcept;

  // declaration for move constructor
  VectorImpl& operator=(VectorImpl<T, VectorType>&& rhs) noexcept;
  /// \endcond

  // The case of assigning a type apart from the same VectorImpl or a
  // `blaze::DenseVector` forwards the assignment to the `PointerVector` base
  // type. In the case of a single compatible value, this fills the vector with
  // that value.
  VectorImpl& operator=(const T& rhs) noexcept;

  // @{
  /// Set the VectorImpl to be a reference to another VectorImpl object
  void set_data_ref(gsl::not_null<VectorType*> rhs) noexcept {
    set_data_ref(rhs->data(), rhs->size());
  }

  void set_data_ref(T* const start, const size_t set_size) noexcept {
    owned_data_.reset();
    (~*this).reset(start, set_size);
    owning_ = false;
  }
  // @}

  /// Returns true if the class owns the data
  bool is_owning() const noexcept { return owning_; }

  /// Serialization for Charm++
  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 protected:
  std::unique_ptr<value_type[], decltype(&free)> owned_data_{nullptr, &free};
  bool owning_{true};

  SPECTRE_ALWAYS_INLINE void reset_pointer_vector(
      const size_t set_size) noexcept {
    this->reset(owned_data_.get(), set_size);
  }
};

#pragma GCC diagnostic push  // Incorrect GCC warning.
#pragma GCC diagnostic ignored "-Wextra"
template <typename T, typename VectorType>
VectorImpl<T, VectorType>::VectorImpl(
    const VectorImpl<T, VectorType>& rhs) noexcept
    : owned_data_(rhs.size() > 0 ? static_cast<value_type*>(
                                       malloc(rhs.size() * sizeof(value_type)))
                                 : nullptr,
                  &free) {
#pragma GCC diagnostic pop
  reset_pointer_vector(rhs.size());
  std::memcpy(data(), rhs.data(), size() * sizeof(value_type));
}

//  definition for copy constructor
template <typename T, typename VectorType>
VectorImpl<T, VectorType>& VectorImpl<T, VectorType>::operator=(
    const VectorImpl<T, VectorType>& rhs) noexcept {
  if (this != &rhs) {
    if (owning_) {
      if (size() != rhs.size()) {
        owned_data_.reset(rhs.size() > 0 ? static_cast<value_type*>(malloc(
                                               rhs.size() * sizeof(value_type)))
                                         : nullptr);
      }
      reset_pointer_vector(rhs.size());
    } else {
      ASSERT(rhs.size() == size(), "Must copy into same size, not "
                                       << rhs.size() << " into " << size());
    }
    std::memcpy(data(), rhs.data(), size() * sizeof(value_type));
  }
  return *this;
}

template <typename T, typename VectorType>
VectorImpl<T, VectorType>::VectorImpl(
    VectorImpl<T, VectorType>&& rhs) noexcept {
  owned_data_ = std::move(rhs.owned_data_);
  ~*this = ~rhs;  // PointerVector is trivially copyable
  owning_ = rhs.owning_;
  rhs.owning_ = true;
  rhs.reset();
}

// definition for move constructor
template <typename T, typename VectorType>
VectorImpl<T, VectorType>& VectorImpl<T, VectorType>::operator=(
    VectorImpl<T, VectorType>&& rhs) noexcept {
  if (this != &rhs) {
    if (owning_) {
      owned_data_ = std::move(rhs.owned_data_);
      ~*this = ~rhs; /* PointerVector is trivially copyable */
      owning_ = rhs.owning_;
    } else {
      ASSERT(rhs.size() == size(), "Must copy into same size, not "
                                       << rhs.size() << " into " << size());
      std::memcpy(data(), rhs.data(), size() * sizeof(value_type));
    }
    rhs.owning_ = true;
    rhs.reset();
  }
  return *this;
}

/// \cond HIDDEN_SYMBOLS
// This is a converting constructor. clang-tidy complains that it's not
// explicit, but we want it to allow conversion.
// clang-tidy: mark as explicit (we want conversion to VectorImpl)
template <typename T, typename VectorType>
template <typename VT, bool VF,
          Requires<cpp17::is_same_v<typename VT::ResultType, VectorType>>>
VectorImpl<T, VectorType>::VectorImpl(
    const blaze::DenseVector<VT, VF>& expression)  // NOLINT
    noexcept
    : owned_data_(static_cast<value_type*>(
                      malloc((~expression).size() * sizeof(value_type))),
                  &free) {
  static_assert(cpp17::is_same_v<typename VT::ResultType, VectorType>,
                "You are attempting to assign the result of an expression "
                "that is not consistent with the VectorImpl type you are "
                "assigning to.");
  reset_pointer_vector((~expression).size());
  ~*this = expression;
}

template <typename T, typename VectorType>
template <typename VT, bool VF>
VectorImpl<T, VectorType>& VectorImpl<T, VectorType>::operator=(
    const blaze::DenseVector<VT, VF>& expression) noexcept {
  static_assert(cpp17::is_same_v<typename VT::ResultType, VectorType>,
                "You are attempting to assign the result of an expression "
                "that is not consistent with the VectorImpl type you are "
                "assigning to.");
  if (owning_ and (~expression).size() != size()) {
    owned_data_.reset(static_cast<value_type*>(
        malloc((~expression).size() * sizeof(value_type))));
    reset_pointer_vector((~expression).size());
  } else if (not owning_) {
    ASSERT((~expression).size() == size(), "Must copy into same size, not "
                                               << (~expression).size()
                                               << " into " << size());
  }
  ~*this = expression;
  return *this;
}
/// \endcond

// The case of assigning a type apart from the same VectorImpl or a
// `blaze::DenseVector` forwards the assignment to the `PointerVector` base
// type. In the case of a single compatible value, this fills the vector with
// that value.
template <typename T, typename VectorType>
VectorImpl<T, VectorType>& VectorImpl<T, VectorType>::operator=(
    const T& rhs) noexcept {
  ~*this = rhs;
  return *this;
}

template <typename T, typename VectorType>
void VectorImpl<T, VectorType>::pup(PUP::er& p) noexcept {  // NOLINT
  auto my_size = size();
  p | my_size;
  if (my_size > 0) {
    if (p.isUnpacking()) {
      owning_ = true;
      owned_data_.reset(my_size > 0 ? static_cast<value_type*>(
                                          malloc(my_size * sizeof(value_type)))
                                    : nullptr);
      reset_pointer_vector(my_size);
    }
    PUParray(p, data(), size());
  }
}

/// Output operator for VectorImpl
template <typename T, typename VectorType>
std::ostream& operator<<(std::ostream& os,
                         const VectorImpl<T, VectorType>& d) noexcept {
  sequence_print_helper(os, d.begin(), d.end());
  return os;
}

/*!
 * \ingroup DataStructuresGroup
 * \brief Instructs Blaze to provide the appropriate vector result type after
 * math operations. This is accomplished by Declaring a type alias in structs
 * accessed by Blaze
 *
 * \param VECTORTYPE The vector type, which matches the type of the operation
 * result (e.g. `DataVector`)
 *
 * \param BLAZE_MATH_TRAIT The blaze trait/expression for which you want to
 * specify the return type (e.g. `AddTrait`).
 */
#define BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(VECTORTYPE, BLAZE_MATH_TRAIT) \
  template <>                                                             \
  struct BLAZE_MATH_TRAIT<VECTORTYPE, VECTORTYPE> {                       \
    using Type = VECTORTYPE;                                              \
  };                                                                      \
  template <>                                                             \
  struct BLAZE_MATH_TRAIT<VECTORTYPE, VECTORTYPE::value_type> {           \
    using Type = VECTORTYPE;                                              \
  };                                                                      \
  template <>                                                             \
  struct BLAZE_MATH_TRAIT<VECTORTYPE::value_type, VECTORTYPE> {           \
    using Type = VECTORTYPE;                                              \
  }

/*!
 * \ingroup DataStructuresGroup
 * \brief Instructs Blaze to provide the appropriate vector result type of an
 * operator between `VECTORTYPE` and `COMPATIBLE, where the operation is
 * represented by `BLAZE_MATH_TRAIT`
 *
 * \param VECTORTYPE The vector type, which matches the type of the operation
 * result (e.g. `DataVector`)
 *
 * \param COMPATIBLE the type for which you want math operations to work with
 * VECTORTYPE smoothly (e.g. `ComplexDataVector`)
 *
 * \param BLAZE_MATH_TRAIT The blaze trait for which you want declare the Type
 * field (e.g. `AddTrait`)
 */
#define BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(VECTORTYPE, COMPATIBLE, \
                                                       BLAZE_MATH_TRAIT)       \
  template <>                                                                  \
  struct BLAZE_MATH_TRAIT<VECTORTYPE, COMPATIBLE> {                            \
    using Type = VECTORTYPE;                                                   \
  };                                                                           \
  template <>                                                                  \
  struct BLAZE_MATH_TRAIT<COMPATIBLE, VECTORTYPE> {                            \
    using Type = VECTORTYPE;                                                   \
  }

/*!
 * \ingroup DataStructuresGroup
 * \brief Declares blaze Type fields which are needed by a contiguous data set
 *
 * \details Type declarations here are suitable for the original DataVector, but
 * this macro might need to be tweaked for other types of data, for instance
 * Fourier coefficients
 *
 * \param VECTORTYPE The vector type, which for the arithmetic operations is the
 * type of the operation result (e.g. `DataVector`)
 *
 */
#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
#define BLAZE_TRAIT_SPECIALIZE_TYPICAL_VECTOR_TRAITS(VECTORTYPE) \
  template <>                                                    \
  struct IsVector<VECTORTYPE> : std::true_type {};               \
  template <>                                                    \
  struct TransposeFlag<VECTORTYPE>                               \
      : BoolConstant<VECTORTYPE::transpose_flag> {};             \
  BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(VECTORTYPE, AddTrait);     \
  BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(VECTORTYPE, SubTrait);     \
  BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(VECTORTYPE, MultTrait);    \
  BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(VECTORTYPE, DivTrait);     \
  template <typename Operator>                                   \
  struct UnaryMapTrait<VECTORTYPE, Operator> {                   \
    using Type = VECTORTYPE;                                     \
  };                                                             \
  template <typename Operator>                                   \
  struct BinaryMapTrait<VECTORTYPE, VECTORTYPE, Operator> {      \
    using Type = VECTORTYPE;                                     \
  }
#else
#define VECTOR_BLAZE_TRAIT_SPEC(VECTORTYPE)                   \
  template <>                                                 \
  struct IsVector<VECTORTYPE> : std::true_type {};            \
  template <>                                                 \
  struct TransposeFlag<VECTORTYPE>                            \
      : BoolConstant<VECTORTYPE::transpose_flag> {};          \
  BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(VECTORTYPE, AddTrait);  \
  BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(VECTORTYPE, SubTrait);  \
  BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(VECTORTYPE, MultTrait); \
  BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(VECTORTYPE, DivTrait);  \
  template <typename Operator>                                \
  struct MapTrait<VECTORTYPE, Operator> {                     \
    using Type = VECTORTYPE;                                  \
  };                                                          \
  template <typename Operator>                                \
  struct MapTrait<VECTORTYPE, VECTORTYPE, Operator> {         \
    using Type = VECTORTYPE;                                  \
  }
#endif  // ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))

/*!
 * \ingroup DataStructuresGroup
 * \brief Declares the set of binary operations often supported for
 * `std::array<VECTORTYPE, size>`, for arbitrary `size`.
 *
 *  \param VECTORTYPE The vector type (e.g. `DataVector`)
 */
#define MAKE_STD_ARRAY_VECTOR_BINOPS(VECTORTYPE)                            \
  DEFINE_STD_ARRAY_BINOP(VECTORTYPE, VECTORTYPE::value_type,                \
                         VECTORTYPE, operator+, std::plus<>())              \
  DEFINE_STD_ARRAY_BINOP(VECTORTYPE, VECTORTYPE,                            \
                         VECTORTYPE::value_type, operator+, std::plus<>())  \
  DEFINE_STD_ARRAY_BINOP(VECTORTYPE, VECTORTYPE, VECTORTYPE, operator+,     \
                         std::plus<>())                                     \
                                                                            \
  DEFINE_STD_ARRAY_BINOP(VECTORTYPE, VECTORTYPE::value_type,                \
                         VECTORTYPE, operator-, std::minus<>())             \
  DEFINE_STD_ARRAY_BINOP(VECTORTYPE, VECTORTYPE,                            \
                         VECTORTYPE::value_type, operator-, std::minus<>()) \
  DEFINE_STD_ARRAY_BINOP(VECTORTYPE, VECTORTYPE, VECTORTYPE, operator-,     \
                         std::minus<>())                                    \
                                                                            \
  DEFINE_STD_ARRAY_INPLACE_BINOP(VECTORTYPE, VECTORTYPE, operator-=,        \
                                 std::minus<>())                            \
  DEFINE_STD_ARRAY_INPLACE_BINOP(                                           \
      VECTORTYPE, VECTORTYPE::value_type, operator-=, std::minus<>())       \
  DEFINE_STD_ARRAY_INPLACE_BINOP(VECTORTYPE, VECTORTYPE, operator+=,        \
                                 std::plus<>())                             \
  DEFINE_STD_ARRAY_INPLACE_BINOP(                                           \
      VECTORTYPE, VECTORTYPE::value_type, operator+=, std::plus<>())

/*!
 * \ingroup DataStructuresGroup
 * \brief Declares `MAKE_EXPRESSION_MATH_ASSIGN_PV` with all assignment
 * arithmetic operations
 *
 * \param VECTORTYPE The vector type (e.g. `DataVector`)
 */
#define MAKE_MATH_ASSIGN_EXPRESSION_ARITHMETIC(VECTORTYPE)  \
  MAKE_MATH_ASSIGN_EXPRESSION_POINTERVECTOR(+=, VECTORTYPE) \
  MAKE_MATH_ASSIGN_EXPRESSION_POINTERVECTOR(-=, VECTORTYPE) \
  MAKE_MATH_ASSIGN_EXPRESSION_POINTERVECTOR(*=, VECTORTYPE) \
  MAKE_MATH_ASSIGN_EXPRESSION_POINTERVECTOR(/=, VECTORTYPE)

/*!
 * \ingroup DataStructuresGroup
 * \brief Defines the MakeWithValueImpl apply specialization
 *
 * \details The MakeWithValueImpl<type,type>::apply(type,double) constructed
 * here produces an object with the same size as the type argument,
 * initialized with the value argument in every entry.
 *
 * \param VECTORTYPE The vector type (e.g. DataVector)
 */
#define MAKE_VECTOR_MAKEWITHVALUES(VECTORTYPE)           \
  namespace MakeWithValueImpls {                         \
  template <>                                            \
  struct MakeWithValueImpl<VECTORTYPE, VECTORTYPE> {     \
    static SPECTRE_ALWAYS_INLINE VECTORTYPE              \
    apply(const VECTORTYPE& input,                       \
          const VECTORTYPE::value_type value) noexcept { \
      return VECTORTYPE(input.size(), value);            \
    }                                                    \
  };                                                     \
  }  // namespace MakeWithValueImpls

// {@
/*!
 * \ingroup DataStructuresGroup
 * \brief Helper struct to determine the base storage type of a VectorImpl or
 * container of VectorImpl
 *
 * \details Extracts the storage type of a `VectorImpl`, a std::array of
 * `VectorImpl`, or a reference or pointer to a `VectorImpl`. In any of these
 * cases, the `type` member is defined as the `ElementType` of the `VectorImpl`
 * in question. If, instead, `VectorBaseType` is passed a numeric type, the
 * `type` member is defined as that numeric type. Examples:
 * - `VectorBaseType_t<DataVector>` is `double`
 * - `VectorBaseType_t<std::array<DataVector, 2>>` is `double`
 * - `VectorBaseType_t<std::complex<double>*>` is `std::complex<double>`
 * - `VectorBaseType_t<std::vector<double>>` is not defined.
 */
template <typename T, typename Enable = std::nullptr_t>
struct VectorBaseType {};
template <typename T>
struct VectorBaseType<
    T, Requires<std::is_arithmetic<T>::value or tt::is_a_v<std::complex, T>>> {
  using type = T;
};
template <typename T>
struct VectorBaseType<
    T,
    Requires<std::is_arithmetic<typename T::ResultType::ElementType>::value or
             tt::is_a_v<std::complex, typename T::ResultType::ElementType>>> {
  using type = typename T::ResultType::ElementType;
};
template <typename T>
struct VectorBaseType<
    T,
    Requires<
        not std::is_arithmetic<typename T::ResultType::ElementType>::value and
        not tt::is_a_v<std::complex, typename T::ResultType::ElementType>>> {
  using type =
      typename VectorBaseType<typename T::ResultType::ElementType>::type;
};
template <typename T>
struct VectorBaseType<T*, std::nullptr_t> {
  using type = typename VectorBaseType<T>::type;
};
template <typename T>
struct VectorBaseType<T&, std::nullptr_t> {
  using type = typename VectorBaseType<T>::type;
};
template <typename T, size_t S>
struct VectorBaseType<std::array<T, S>, std::nullptr_t> {
  using type = typename VectorBaseType<T>::type;
};
// @}

template <typename T>
using VectorBaseType_t = typename VectorBaseType<T>::type;

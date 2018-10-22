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
  // Upcast to `BaseType`
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
  template <class C, Requires<cpp17::is_same_v<C, T>> = nullptr>
  VectorImpl(std::initializer_list<C> list) noexcept
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
  VectorImpl<T, VectorType>& operator=(
      const blaze::DenseVector<VT, VF>& expression) noexcept;

  //  declaration for copy constructor
  VectorImpl<T, VectorType>& operator=(
      const VectorImpl<T, VectorType>& rhs) noexcept;

  // declaration for move constructor
  VectorImpl<T, VectorType>& operator=(
      VectorImpl<T, VectorType>&& rhs) noexcept;
  /// \endcond

  // create from a single stored value, which becomes the sole entry.
  VectorImpl<T, VectorType>& operator=(const T& rhs) noexcept;

  // @{
  /// Set the VectorImpl to be a reference to another VectorImpl object
  template <typename RhsVectorType>
  void set_data_ref(gsl::not_null<RhsVectorType*> rhs) noexcept {
    static_assert(
        cpp17::is_same_v<typename RhsVectorType::ResultType, VectorType>,
        "You are attempting to assign the pointer of a VectorImpl type "
        "that is not consistent with the VectorImpl type you are "
        "assigning to.");
    set_data_ref(rhs->data(), rhs->size());
  }

  void set_data_ref(T* start, size_t set_size) noexcept {
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

template <typename T, typename VectorType>
VectorImpl<T, VectorType>::VectorImpl(
    VectorImpl<T, VectorType>&& rhs) noexcept {
  owned_data_ = std::move(rhs.owned_data_);
  ~*this = ~rhs;  // PointerVector is trivially copyable
  owning_ = rhs.owning_;
  rhs.owning_ = true;
  rhs.reset();
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

//  declaration for copy constructor
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

// declaration for move constructor
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
/// \endcond

// create from a single stored value, which becomes the sole entry.
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

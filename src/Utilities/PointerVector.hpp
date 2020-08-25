// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class PointerVector

#pragma once

#include <cmath>
#include <functional>

// The Utilities/Blaze.hpp configures Blaze
#include "ErrorHandling/Assert.hpp"
#include "Utilities/Blaze.hpp"

#include <blaze/math/CustomVector.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Version.h>
#include <blaze/util/typetraits/RemoveConst.h>

// clang-tidy: do not use pointer arithmetic
#define SPECTRE_BLAZE_ALLOCATOR(_TYPE_T, _SIZE_V) \
  new _TYPE_T[_SIZE_V]  // NOLINT
#define SPECTRE_BLAZE_DEALLOCATOR blaze::ArrayDelete()

// Blaze version compatibility definitions:
// between Blaze 3.2 and 3.4, there have been several minor changes to type
// definitions. Here, we define the aliases to the appropriate tokens for the
// respective versions.
#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION < 6))
const bool blaze_unaligned = blaze::unaligned != 0;
const bool blaze_unpadded = blaze::unpadded != 0;
using AlignmentFlag_t = bool;
using PaddingFlag_t = bool;
template <typename T>
using BlazePow = blaze::UnaryPow<T>;
#elif ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION == 6))
const blaze::AlignmentFlag blaze_unaligned = blaze::AlignmentFlag::unaligned;
const bool blaze_unpadded = blaze::unpadded;
#define BLAZE_TEMPLATE template
using AlignmentFlag_t = blaze::AlignmentFlag;
using PaddingFlag_t = bool;
template <typename T>
using BlazePow = blaze::Bind2nd<blaze::Pow, T>;
#elif ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION >= 7))
const blaze::AlignmentFlag blaze_unaligned = blaze::AlignmentFlag::unaligned;
const blaze::PaddingFlag blaze_unpadded = blaze::PaddingFlag::unpadded;
#define BLAZE_TEMPLATE template
using AlignmentFlag_t = blaze::AlignmentFlag;
using PaddingFlag_t = blaze::PaddingFlag;
template <typename T>
using BlazePow = blaze::Bind2nd<blaze::Pow, T>;
#endif

template <bool B>
using blaze_enable_if_t = blaze::EnableIf_t<B>;
template <typename T>
using blaze_remove_const_t = blaze::RemoveConst_t<T>;
template <typename T>
using blaze_simd_trait_t = blaze::SIMDTrait_t<T>;
template <typename T>
using blaze_element_type_t = blaze::ElementType_t<T>;
template <typename T>
using blaze_result_type_t = blaze::ResultType_t<T>;
template <typename T1, typename T2>
using blaze_mult_trait_t = blaze::MultTrait_t<T1, T2>;
template <typename T1, typename T2>
using blaze_div_trait_t = blaze::DivTrait_t<T1, T2>;
template <typename T1, typename T2>
using blaze_cross_trait_t = blaze::CrossTrait_t<T1, T2>;
template <typename T>
using blaze_const_iterator_t = blaze::ConstIterator_t<T>;
template <typename T>
const bool blaze_is_numeric = blaze::IsNumeric_v<T>;
template <typename T>
const bool blaze_is_numeric_v = blaze_is_numeric<T>;

namespace blaze {
template <typename T>
BLAZE_ALWAYS_INLINE SIMDdouble step_function(const SIMDf64<T>& v) noexcept
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
  return _mm512_set_pd((~v).eval().value[7] < 0.0 ? 0.0 : 1.0,
                       (~v).eval().value[6] < 0.0 ? 0.0 : 1.0,
                       (~v).eval().value[5] < 0.0 ? 0.0 : 1.0,
                       (~v).eval().value[4] < 0.0 ? 0.0 : 1.0,
                       (~v).eval().value[3] < 0.0 ? 0.0 : 1.0,
                       (~v).eval().value[2] < 0.0 ? 0.0 : 1.0,
                       (~v).eval().value[1] < 0.0 ? 0.0 : 1.0,
                       (~v).eval().value[0] < 0.0 ? 0.0 : 1.0);
}
#elif BLAZE_AVX_MODE
{
  return _mm256_set_pd((~v).eval().value[3] < 0.0 ? 0.0 : 1.0,
                       (~v).eval().value[2] < 0.0 ? 0.0 : 1.0,
                       (~v).eval().value[1] < 0.0 ? 0.0 : 1.0,
                       (~v).eval().value[0] < 0.0 ? 0.0 : 1.0);
}
#elif BLAZE_SSE2_MODE
{
  return _mm_set_pd((~v).eval().value[1] < 0.0 ? 0.0 : 1.0,
                    (~v).eval().value[0] < 0.0 ? 0.0 : 1.0);
}
#else
{
  return SIMDdouble{(~v).value < 0.0 ? 0.0 : 1.0};
}
#endif

BLAZE_ALWAYS_INLINE double step_function(const double v) noexcept {
  return v < 0.0 ? 0.0 : 1.0;
}

struct StepFunction {
  explicit inline StepFunction() = default;

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) operator()(const T& a) const noexcept {
    return step_function(a);
  }

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) load(const T& a) const noexcept {
    BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK(T);
    return step_function(a);
  }
};
}  // namespace blaze

template <typename VT, bool TF>
// NOLINTNEXTLINE(readability-const-return-type)
BLAZE_ALWAYS_INLINE decltype(auto) step_function(
    const blaze::DenseVector<VT, TF>& vec) noexcept {
  return map(~vec, blaze::StepFunction{});
}

template <typename VT, bool TF>
BLAZE_ALWAYS_INLINE decltype(auto) StepFunction(
    const blaze::DenseVector<VT, TF>& vec) noexcept {
  return map(~vec, blaze::StepFunction{});
}

namespace blaze {
template <typename ST>
struct DivideScalarByVector {
 public:
  explicit inline DivideScalarByVector(ST scalar) : scalar_(scalar) {}

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) operator()(const T& a) const {
    return scalar_ / a;
  }

  template <typename T>
  static constexpr bool simdEnabled() {
    return blaze::HasSIMDDiv<T, ST>::value;
  }

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) load(const T& a) const {
    BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK(T);
    return set(scalar_) / a;
  }

 private:
  ST scalar_;
};

#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION < 6))
template <typename Scalar, typename VT, bool TF,
          typename = blaze_enable_if_t<blaze_is_numeric<Scalar>>>
BLAZE_ALWAYS_INLINE decltype(auto) operator/(
    Scalar scalar, const blaze::DenseVector<VT, TF>& vec) {
  return forEach(~vec, DivideScalarByVector<Scalar>(scalar));
}
#endif

template <typename ST>
struct AddScalar {
 public:
  explicit inline AddScalar(ST scalar) : scalar_(scalar) {}

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) operator()(const T& a) const {
    return a + scalar_;
  }

  template <typename T>
  static constexpr bool simdEnabled() {
    return blaze::HasSIMDAdd<T, ST>::value;
  }

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) load(const T& a) const {
    BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK(T);
    return a + set(scalar_);
  }

 private:
  ST scalar_;
};

#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION < 6))
template <typename VT, bool TF, typename Scalar,
          typename = blaze_enable_if_t<blaze_is_numeric<Scalar>>>
decltype(auto) operator+(const blaze::DenseVector<VT, TF>& vec, Scalar scalar) {
  return forEach(~vec, AddScalar<Scalar>(scalar));
}

template <typename Scalar, typename VT, bool TF,
          typename = blaze_enable_if_t<blaze_is_numeric<Scalar>>>
decltype(auto) operator+(Scalar scalar, const blaze::DenseVector<VT, TF>& vec) {
  return forEach(~vec, AddScalar<Scalar>(scalar));
}

template <typename VT, bool TF, typename Scalar,
          typename = blaze_enable_if_t<blaze_is_numeric<Scalar>>>
VT& operator+=(blaze::DenseVector<VT, TF>& vec, Scalar scalar) {
  (~vec) = (~vec) + scalar;
  return ~vec;
}
#endif

template <typename ST>
struct SubScalarRhs {
 public:
  explicit inline SubScalarRhs(ST scalar) : scalar_(scalar) {}

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) operator()(const T& a) const {
    return a - scalar_;
  }

  template <typename T>
  static constexpr bool simdEnabled() {
    return blaze::HasSIMDSub<T, ST>::value;
  }

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) load(const T& a) const {
    BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK(T);
    return a - set(scalar_);
  }

 private:
  ST scalar_;
};

template <typename ST>
struct SubScalarLhs {
 public:
  explicit inline SubScalarLhs(ST scalar) : scalar_(scalar) {}

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) operator()(const T& a) const {
    return scalar_ - a;
  }

  template <typename T>
  static constexpr bool simdEnabled() {
    return blaze::HasSIMDSub<T, ST>::value;
  }

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) load(const T& a) const {
    BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK(T);
    return set(scalar_) - a;
  }

 private:
  ST scalar_;
};

#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION < 6))
template <typename VT, bool TF, typename Scalar,
          typename = blaze_enable_if_t<blaze_is_numeric<Scalar>>>
decltype(auto) operator-(const blaze::DenseVector<VT, TF>& vec, Scalar scalar) {
  return forEach(~vec, SubScalarRhs<Scalar>(scalar));
}

template <typename VT, bool TF, typename Scalar,
          typename = blaze_enable_if_t<blaze_is_numeric<Scalar>>>
decltype(auto) operator-(Scalar scalar, const blaze::DenseVector<VT, TF>& vec) {
  return forEach(~vec, SubScalarLhs<Scalar>(scalar));
}

template <typename VT, bool TF, typename Scalar,
          typename = blaze_enable_if_t<blaze_is_numeric<Scalar>>>
VT& operator-=(blaze::DenseVector<VT, TF>& vec, Scalar scalar) {
  (~vec) = (~vec) - scalar;
  return ~vec;
}
#endif
}  // namespace blaze

namespace blaze {
// Enable support for reference wrappers with Blaze
template <typename T>
struct UnderlyingElement<std::reference_wrapper<T>> {
  using Type = typename UnderlyingElement<std::decay_t<T>>::Type;
};
}  // namespace blaze

/*!
 * \ingroup UtilitiesGroup
 * \brief A raw pointer endowed with expression template support via the Blaze
 * library
 *
 * PointerVector can be used instead of a raw pointer to pass around size
 * information and to be able to have the pointer array support expression
 * templates. The primary use case for PointerVector is inside the Data class
 * so that Data has support for expression templates but not incurring any
 * overhead for them.
 *
 * See the Blaze documentation for CustomVector for details on the template
 * parameters to PointerVector since CustomVector is what PointerVector is
 * modeled after.
 *
 * One additional feature that Blaze's CustomVector (currently) does not support
 * is the ability to change the result type so that CustomVector can be used for
 * the expression template backend for different vector types. PointerVector
 * allows this by passing the `ExprResultType` template parameter. For example,
 * `DataVector` sets `ExprResultType = DataVector`.
 */
template <typename Type, AlignmentFlag_t AF = blaze_unaligned,
          PaddingFlag_t PF = blaze_unpadded,
          bool TF = blaze::defaultTransposeFlag,
          typename ExprResultType =
              blaze::DynamicVector<blaze_remove_const_t<Type>, TF>>
struct PointerVector
    : public blaze::DenseVector<PointerVector<Type, AF, PF, TF, ExprResultType>,
                                TF> {
  /// \cond
 public:
  using This = PointerVector<Type, AF, PF, TF, ExprResultType>;
  using BaseType = blaze::DenseVector<This, TF>;
  using ResultType = ExprResultType;
  using TransposeType = PointerVector<Type, AF, PF, !TF, ExprResultType>;
  using ElementType = Type;
  using SIMDType = blaze_simd_trait_t<ElementType>;
  using ReturnType = const Type&;
  using CompositeType = const PointerVector&;

  using Reference = Type&;
  using ConstReference = const Type&;
  using Pointer = Type*;
  using ConstPointer = const Type*;

#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 6))
  using Iterator = blaze::DenseIterator<Type, AF != blaze_unaligned>;
  using ConstIterator = blaze::DenseIterator<const Type, AF != blaze_unaligned>;
#else
  using Iterator = blaze::DenseIterator<Type, AF>;
  using ConstIterator = blaze::DenseIterator<const Type, AF>;
#endif
  enum : bool { simdEnabled = blaze::IsVectorizable<Type>::value };
  enum : bool { smpAssignable = !blaze::IsSMPAssignable<Type>::value };

  PointerVector() = default;
  PointerVector(Type* ptr, size_t size) : v_(ptr), size_(size) {}
  PointerVector(const PointerVector& /*rhs*/) = default;
  PointerVector& operator=(const PointerVector& /*rhs*/) = default;
  PointerVector(PointerVector&& /*rhs*/) = default;
  PointerVector& operator=(PointerVector&& /*rhs*/) = default;
  ~PointerVector() = default;

  /*!\name Data access functions */
  //@{
  Type& operator[](const size_t i) noexcept {
    ASSERT(i < size(), "i = " << i << ", size = " << size());
    // clang-tidy: do not use pointer arithmetic
    return v_[i];  // NOLINT
  }
  const Type& operator[](const size_t i) const noexcept {
    ASSERT(i < size(), "i = " << i << ", size = " << size());
    // clang-tidy: do not use pointer arithmetic
    return v_[i];  // NOLINT
  }
  Reference at(size_t index);
  ConstReference at(size_t index) const;
  Pointer data() noexcept { return v_; }
  ConstPointer data() const noexcept { return v_; }
  Iterator begin() noexcept { return Iterator(v_); }
  ConstIterator begin() const noexcept { return ConstIterator(v_); }
  ConstIterator cbegin() const noexcept { return ConstIterator(v_); }
  Iterator end() noexcept {
    // clang-tidy: do not use pointer arithmetic
    return Iterator(v_ + size_);  // NOLINT
  }
  ConstIterator end() const noexcept {
    // clang-tidy: do not use pointer arithmetic
    return ConstIterator(v_ + size_);  // NOLINT
  }
  ConstIterator cend() const noexcept { return ConstIterator(v_ + size_); }
  //@}

  /*!\name Assignment operators */
  //@{
  PointerVector& operator=(const Type& rhs);
  PointerVector& operator=(std::initializer_list<Type> list);

  template <typename Other, size_t N>
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  PointerVector& operator=(const Other (&array)[N]);

  template <typename VT>
  PointerVector& operator=(const blaze::Vector<VT, TF>& rhs);
  template <typename VT>
  PointerVector& operator+=(const blaze::Vector<VT, TF>& rhs);
  template <typename VT>
  PointerVector& operator-=(const blaze::Vector<VT, TF>& rhs);
  template <typename VT>
  PointerVector& operator*=(const blaze::Vector<VT, TF>& rhs);
  template <typename VT>
  PointerVector& operator/=(const blaze::Vector<VT, TF>& rhs);
  template <typename VT>
  PointerVector& operator%=(const blaze::Vector<VT, TF>& rhs);

  template <typename Other>
  std::enable_if_t<blaze_is_numeric_v<Other>, This>& operator*=(Other rhs);

  template <typename Other>
  std::enable_if_t<blaze_is_numeric_v<Other>, This>& operator/=(Other rhs);
  //@}

  /*!\name Utility functions */
  //@{
  void clear() noexcept {
    size_ = 0;
    v_ = nullptr;
  }

  size_t spacing() const noexcept { return size_; }

  size_t size() const noexcept { return size_; }
  //@}

  /*!\name Resource management functions */
  //@{
  void reset() { clear(); }

  void reset(Type* ptr, size_t n) noexcept {
    v_ = ptr;
    size_ = n;
  }
  //@}

 private:
  template <typename VT>
  using VectorizedAssign = std::integral_constant<
      bool, blaze::useOptimizedKernels && simdEnabled && VT::simdEnabled &&
                blaze::IsSIMDCombinable<Type, blaze_element_type_t<VT>>::value>;

  template <typename VT>
  using VectorizedAddAssign = std::integral_constant<
      bool,
      blaze::useOptimizedKernels && simdEnabled && VT::simdEnabled &&
          blaze::IsSIMDCombinable<Type, blaze_element_type_t<VT>>::value &&
          blaze::HasSIMDAdd<Type, blaze_element_type_t<VT>>::value>;

  template <typename VT>
  using VectorizedSubAssign = std::integral_constant<
      bool,
      blaze::useOptimizedKernels && simdEnabled && VT::simdEnabled &&
          blaze::IsSIMDCombinable<Type, blaze_element_type_t<VT>>::value &&
          blaze::HasSIMDSub<Type, blaze_element_type_t<VT>>::value>;

  template <typename VT>
  using VectorizedMultAssign = std::integral_constant<
      bool,
      blaze::useOptimizedKernels && simdEnabled && VT::simdEnabled &&
          blaze::IsSIMDCombinable<Type, blaze_element_type_t<VT>>::value &&
          blaze::HasSIMDMult<Type, blaze_element_type_t<VT>>::value>;

  template <typename VT>
  using VectorizedDivAssign = std::integral_constant<
      bool,
      blaze::useOptimizedKernels && simdEnabled && VT::simdEnabled &&
          blaze::IsSIMDCombinable<Type, blaze_element_type_t<VT>>::value &&
          blaze::HasSIMDDiv<Type, blaze_element_type_t<VT>>::value>;

  //! The number of elements packed within a single SIMD element.
  enum : size_t { SIMDSIZE = blaze::SIMDTrait<ElementType>::size };

 public:
  /*!\name Expression template evaluation functions */
  //@{
  template <typename Other>
  bool canAlias(const Other* alias) const noexcept;
  template <typename Other>
  bool isAliased(const Other* alias) const noexcept;

  bool isAligned() const noexcept;
  bool canSMPAssign() const noexcept;

  BLAZE_ALWAYS_INLINE SIMDType load(size_t index) const noexcept;
  BLAZE_ALWAYS_INLINE SIMDType loada(size_t index) const noexcept;
  BLAZE_ALWAYS_INLINE SIMDType loadu(size_t index) const noexcept;

  BLAZE_ALWAYS_INLINE void store(size_t index, const SIMDType& value) noexcept;
  BLAZE_ALWAYS_INLINE void storea(size_t index, const SIMDType& value) noexcept;
  BLAZE_ALWAYS_INLINE void storeu(size_t index, const SIMDType& value) noexcept;
  BLAZE_ALWAYS_INLINE void stream(size_t index, const SIMDType& value) noexcept;

  template <typename VT>
  std::enable_if_t<not(This::template VectorizedAssign<VT>::value)> assign(
      const blaze::DenseVector<VT, TF>& rhs);

  template <typename VT>
  std::enable_if_t<(VectorizedAssign<VT>::value)> assign(
      const blaze::DenseVector<VT, TF>& rhs);

  template <typename VT>
  std::enable_if_t<not(This::template VectorizedAddAssign<VT>::value)>
  addAssign(const blaze::DenseVector<VT, TF>& rhs);

  template <typename VT>
  std::enable_if_t<(VectorizedAddAssign<VT>::value)> addAssign(
      const blaze::DenseVector<VT, TF>& rhs);

  template <typename VT>
  void addAssign(const blaze::SparseVector<VT, TF>& rhs);

  template <typename VT>
  std::enable_if_t<not(This::template VectorizedSubAssign<VT>::value)>
  subAssign(const blaze::DenseVector<VT, TF>& rhs);

  template <typename VT>
  std::enable_if_t<(VectorizedSubAssign<VT>::value)> subAssign(
      const blaze::DenseVector<VT, TF>& rhs);

  template <typename VT>
  void subAssign(const blaze::SparseVector<VT, TF>& rhs);

  template <typename VT>
  std::enable_if_t<not(PointerVector<Type, AF, PF, TF, ResultType>::
                           template VectorizedMultAssign<VT>::value)>
  multAssign(const blaze::DenseVector<VT, TF>& rhs);

  template <typename VT>
  std::enable_if_t<(VectorizedMultAssign<VT>::value)> multAssign(
      const blaze::DenseVector<VT, TF>& rhs);

  template <typename VT>
  void multAssign(const blaze::SparseVector<VT, TF>& rhs);

  template <typename VT>
  std::enable_if_t<not(This::template VectorizedDivAssign<VT>::value)>
  divAssign(const blaze::DenseVector<VT, TF>& rhs);

  template <typename VT>
  std::enable_if_t<(VectorizedDivAssign<VT>::value)> divAssign(
      const blaze::DenseVector<VT, TF>& rhs);
  //@}

 private:
  Type* v_ = nullptr;
  size_t size_ = 0;
  /// \endcond
};

/// \cond
template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
inline typename PointerVector<Type, AF, PF, TF, ExprResultType>::Reference
PointerVector<Type, AF, PF, TF, ExprResultType>::at(size_t index) {
  if (index >= size_) {
    BLAZE_THROW_OUT_OF_RANGE("Invalid vector access index");
  }
  return (*this)[index];
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
inline typename PointerVector<Type, AF, PF, TF, ExprResultType>::ConstReference
PointerVector<Type, AF, PF, TF, ExprResultType>::at(size_t index) const {
  if (index >= size_) {
    BLAZE_THROW_OUT_OF_RANGE("Invalid vector access index");
  }
  return (*this)[index];
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
inline PointerVector<Type, AF, PF, TF, ExprResultType>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator=(const Type& rhs) {
  for (size_t i = 0; i < size_; ++i) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] = rhs;  // NOLINT
  }
  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
inline PointerVector<Type, AF, PF, TF, ExprResultType>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator=(
    std::initializer_list<Type> list) {
  ASSERT(list.size() <= size_, "Invalid assignment to custom vector");
  std::fill(std::copy(list.begin(), list.end(), v_), v_ + size_, Type());
  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename Other, size_t N>
inline PointerVector<Type, AF, PF, TF, ExprResultType>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator=(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const Other (&array)[N]) {
  ASSERT(size_ == N, "Invalid array size");
  for (size_t i = 0UL; i < N; ++i) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] = array[i];  // NOLINT
  }
  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline PointerVector<Type, AF, PF, TF, ExprResultType>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator=(
    const blaze::Vector<VT, TF>& rhs) {
  ASSERT((~rhs).size() == size_,
         "Vector sizes do not match: " << size_ << " and " << (~rhs).size());
  blaze::smpAssign(*this, ~rhs);
  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline PointerVector<Type, AF, PF, TF, ExprResultType>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator+=(
    const blaze::Vector<VT, TF>& rhs) {
  ASSERT((~rhs).size() == size_,
         "Vector sizes do not match: " << size_ << " and " << (~rhs).size());
  blaze::smpAddAssign(*this, ~rhs);
  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline PointerVector<Type, AF, PF, TF, ExprResultType>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator-=(
    const blaze::Vector<VT, TF>& rhs) {
  ASSERT((~rhs).size() == size_,
         "Vector sizes do not match: " << size_ << " and " << (~rhs).size());
  blaze::smpSubAssign(*this, ~rhs);
  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline PointerVector<Type, AF, PF, TF, ExprResultType>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator*=(
    const blaze::Vector<VT, TF>& rhs) {
  BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG(VT, TF);
  BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION(blaze_result_type_t<VT>);

  using MultType = blaze_mult_trait_t<ResultType, blaze_result_type_t<VT>>;

  BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG(MultType, TF);
  BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION(MultType);

  ASSERT((~rhs).size() == size_,
         "Vector sizes do not match: " << size_ << " and " << (~rhs).size());
  blaze::smpMultAssign(*this, ~rhs);
  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline PointerVector<Type, AF, PF, TF, ExprResultType>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator/=(
    const blaze::Vector<VT, TF>& rhs) {
  BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG(VT, TF);
  BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION(blaze_result_type_t<VT>);

  using DivType = blaze_div_trait_t<ResultType, blaze_result_type_t<VT>>;

  BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG(DivType, TF);
  BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION(DivType);

  ASSERT((~rhs).size() == size_,
         "Vector sizes do not match: " << size_ << " and " << (~rhs).size());
  blaze::smpDivAssign(*this, ~rhs);
  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline PointerVector<Type, AF, PF, TF, ExprResultType>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator%=(
    const blaze::Vector<VT, TF>& rhs) {
  using blaze::assign;

  BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG(VT, TF);
  BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION(blaze_result_type_t<VT>);

  using CrossType = blaze_cross_trait_t<ResultType, blaze_result_type_t<VT>>;

  BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE(CrossType);
  BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG(CrossType, TF);
  BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION(CrossType);

  if (size_ != 3UL || (~rhs).size() != 3UL) {
    BLAZE_THROW_INVALID_ARGUMENT("Invalid vector size for cross product");
  }

  const CrossType tmp(*this % (~rhs));
  assign(*this, tmp);

  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename Other>
inline std::enable_if_t<blaze_is_numeric_v<Other>,
                        PointerVector<Type, AF, PF, TF, ExprResultType>>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator*=(Other rhs) {
  blaze::smpAssign(*this, (*this) * rhs);
  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename Other>
inline std::enable_if_t<blaze_is_numeric_v<Other>,
                        PointerVector<Type, AF, PF, TF, ExprResultType>>&
PointerVector<Type, AF, PF, TF, ExprResultType>::operator/=(Other rhs) {
  BLAZE_USER_ASSERT(rhs != Other(0), "Division by zero detected");
  blaze::smpAssign(*this, (*this) / rhs);
  return *this;
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename Other>
inline bool PointerVector<Type, AF, PF, TF, ExprResultType>::canAlias(
    const Other* alias) const noexcept {
  return static_cast<const void*>(this) == static_cast<const void*>(alias);
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename Other>
inline bool PointerVector<Type, AF, PF, TF, ExprResultType>::isAliased(
    const Other* alias) const noexcept {
  return static_cast<const void*>(this) == static_cast<const void*>(alias);
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
inline bool PointerVector<Type, AF, PF, TF, ExprResultType>::isAligned() const
    noexcept {
  return (AF || checkAlignment(v_));
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
inline bool PointerVector<Type, AF, PF, TF, ExprResultType>::canSMPAssign()
    const noexcept {
  return (size() > blaze::SMP_DVECASSIGN_THRESHOLD);
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
BLAZE_ALWAYS_INLINE
    typename PointerVector<Type, AF, PF, TF, ExprResultType>::SIMDType
    PointerVector<Type, AF, PF, TF, ExprResultType>::load(size_t index) const
    noexcept {
  if (AF != blaze_unaligned) {
    return loada(index);
  }
  return loadu(index);
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
BLAZE_ALWAYS_INLINE
    typename PointerVector<Type, AF, PF, TF, ExprResultType>::SIMDType
    PointerVector<Type, AF, PF, TF, ExprResultType>::loada(size_t index) const
    noexcept {
  using blaze::loada;

  BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);

  BLAZE_INTERNAL_ASSERT(index < size_, "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(index + SIMDSIZE <= size_,
                        "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(!AF || index % SIMDSIZE == 0UL,
                        "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(checkAlignment(v_ + index),
                        "Invalid vector access index");
  // clang-tidy: do not use pointer arithmetic
  return loada(v_ + index);  // NOLINT
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
BLAZE_ALWAYS_INLINE
    typename PointerVector<Type, AF, PF, TF, ExprResultType>::SIMDType
    PointerVector<Type, AF, PF, TF, ExprResultType>::loadu(size_t index) const
    noexcept {
  using blaze::loadu;

  BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);

  BLAZE_INTERNAL_ASSERT(index < size_, "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(index + SIMDSIZE <= size_,
                        "Invalid vector access index");
  // clang-tidy: do not use pointer arithmetic
  return loadu(v_ + index);  // NOLINT
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
BLAZE_ALWAYS_INLINE void PointerVector<Type, AF, PF, TF, ExprResultType>::store(
    size_t index, const SIMDType& value) noexcept {
  if (AF != blaze_unaligned) {
    storea(index, value);
  } else {
    storeu(index, value);
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
BLAZE_ALWAYS_INLINE void
PointerVector<Type, AF, PF, TF, ExprResultType>::storea(
    size_t index, const SIMDType& value) noexcept {
  using blaze::storea;

  BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);

  BLAZE_INTERNAL_ASSERT(index < size_, "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(index + SIMDSIZE <= size_,
                        "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(!AF || index % SIMDSIZE == 0UL,
                        "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(checkAlignment(v_ + index),
                        "Invalid vector access index");
  // clang-tidy: do not use pointer arithmetic
  storea(v_ + index, value);  // NOLINT
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
BLAZE_ALWAYS_INLINE void
PointerVector<Type, AF, PF, TF, ExprResultType>::storeu(
    size_t index, const SIMDType& value) noexcept {
  using blaze::storeu;

  BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);

  BLAZE_INTERNAL_ASSERT(index < size_, "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(index + SIMDSIZE <= size_,
                        "Invalid vector access index");
  // clang-tidy: do not use pointer arithmetic
  storeu(v_ + index, value);  // NOLINT
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
BLAZE_ALWAYS_INLINE void
PointerVector<Type, AF, PF, TF, ExprResultType>::stream(
    size_t index, const SIMDType& value) noexcept {
  using blaze::stream;

  BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);

  BLAZE_INTERNAL_ASSERT(index < size_, "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(index + SIMDSIZE <= size_,
                        "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(!AF || index % SIMDSIZE == 0UL,
                        "Invalid vector access index");
  BLAZE_INTERNAL_ASSERT(checkAlignment(v_ + index),
                        "Invalid vector access index");
  // clang-tidy: do not use pointer arithmetic
  stream(v_ + index, value);  // NOLINT
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline std::enable_if_t<not(PointerVector<Type, AF, PF, TF, ExprResultType>::
                                BLAZE_TEMPLATE VectorizedAssign<VT>::value)>
PointerVector<Type, AF, PF, TF, ExprResultType>::assign(
    const blaze::DenseVector<VT, TF>& rhs) {
  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const size_t ipos(size_ & size_t(-2));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % 2UL)) == ipos,
                        "Invalid end calculation");

  for (size_t i = 0UL; i < ipos; i += 2UL) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] = (~rhs)[i];              // NOLINT
    v_[i + 1UL] = (~rhs)[i + 1UL];  // NOLINT
  }
  if (ipos < (~rhs).size()) {
    // clang-tidy: do not use pointer arithmetic
    v_[ipos] = (~rhs)[ipos];  // NOLINT
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline std::enable_if_t<(PointerVector<Type, AF, PF, TF, ExprResultType>::
                             BLAZE_TEMPLATE VectorizedAssign<VT>::value)>
PointerVector<Type, AF, PF, TF, ExprResultType>::assign(
    const blaze::DenseVector<VT, TF>& rhs) {
  BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);

  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const size_t ipos(size_ & size_t(-SIMDSIZE));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % SIMDSIZE)) == ipos,
                        "Invalid end calculation");

  if (AF && blaze::useStreaming &&
      size_ > (blaze::cacheSize / (sizeof(Type) * 3UL)) &&
      !(~rhs).isAliased(this)) {
    size_t i(0UL);

    for (; i < ipos; i += SIMDSIZE) {
      stream(i, (~rhs).load(i));
    }
    for (; i < size_; ++i) {
      // clang-tidy: do not use pointer arithmetic
      v_[i] = (~rhs)[i];  // NOLINT
    }
  } else {
    const size_t i4way(size_ & size_t(-SIMDSIZE * 4));
    BLAZE_INTERNAL_ASSERT((size_ - (size_ % (SIMDSIZE * 4UL))) == i4way,
                          "Invalid end calculation");
    BLAZE_INTERNAL_ASSERT(i4way <= ipos, "Invalid end calculation");

    size_t i(0UL);
    blaze_const_iterator_t<VT> it((~rhs).begin());

    for (; i < i4way; i += SIMDSIZE * 4UL) {
      store(i, it.load());
      it += SIMDSIZE;
      store(i + SIMDSIZE, it.load());
      it += SIMDSIZE;
      store(i + SIMDSIZE * 2UL, it.load());
      it += SIMDSIZE;
      store(i + SIMDSIZE * 3UL, it.load());
      it += SIMDSIZE;
    }
    for (; i < ipos; i += SIMDSIZE, it += SIMDSIZE) {
      store(i, it.load());
    }
    for (; i < size_; ++i, ++it) {
      // clang-tidy: do not use pointer arithmetic
      v_[i] = *it;  // NOLINT
    }
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline std::enable_if_t<not(PointerVector<Type, AF, PF, TF, ExprResultType>::
                                BLAZE_TEMPLATE VectorizedAddAssign<VT>::value)>
PointerVector<Type, AF, PF, TF, ExprResultType>::addAssign(
    const blaze::DenseVector<VT, TF>& rhs) {
  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const size_t ipos(size_ & size_t(-2));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % 2UL)) == ipos,
                        "Invalid end calculation");

  for (size_t i = 0UL; i < ipos; i += 2UL) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] += (~rhs)[i];              // NOLINT
    v_[i + 1UL] += (~rhs)[i + 1UL];  // NOLINT
  }
  if (ipos < (~rhs).size()) {
    // clang-tidy: do not use pointer arithmetic
    v_[ipos] += (~rhs)[ipos];  // NOLINT
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline std::enable_if_t<(PointerVector<Type, AF, PF, TF, ExprResultType>::
                             BLAZE_TEMPLATE VectorizedAddAssign<VT>::value)>
PointerVector<Type, AF, PF, TF, ExprResultType>::addAssign(
    const blaze::DenseVector<VT, TF>& rhs) {
  BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);

  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const size_t ipos(size_ & size_t(-SIMDSIZE));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % SIMDSIZE)) == ipos,
                        "Invalid end calculation");

  const size_t i4way(size_ & size_t(-SIMDSIZE * 4));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % (SIMDSIZE * 4UL))) == i4way,
                        "Invalid end calculation");
  BLAZE_INTERNAL_ASSERT(i4way <= ipos, "Invalid end calculation");

  size_t i(0UL);
  blaze_const_iterator_t<VT> it((~rhs).begin());

  for (; i < i4way; i += SIMDSIZE * 4UL) {
    store(i, load(i) + it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE, load(i + SIMDSIZE) + it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE * 2UL, load(i + SIMDSIZE * 2UL) + it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE * 3UL, load(i + SIMDSIZE * 3UL) + it.load());
    it += SIMDSIZE;
  }
  for (; i < ipos; i += SIMDSIZE, it += SIMDSIZE) {
    store(i, load(i) + it.load());
  }
  for (; i < size_; ++i, ++it) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] += *it;  // NOLINT
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline void PointerVector<Type, AF, PF, TF, ExprResultType>::addAssign(
    const blaze::SparseVector<VT, TF>& rhs) {
  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  for (blaze_const_iterator_t<VT> element = (~rhs).begin();
       element != (~rhs).end(); ++element) {
    v_[element->index()] += element->value();
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline std::enable_if_t<not(PointerVector<Type, AF, PF, TF, ExprResultType>::
                                BLAZE_TEMPLATE VectorizedSubAssign<VT>::value)>
PointerVector<Type, AF, PF, TF, ExprResultType>::subAssign(
    const blaze::DenseVector<VT, TF>& rhs) {
  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const size_t ipos(size_ & size_t(-2));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % 2UL)) == ipos,
                        "Invalid end calculation");

  for (size_t i = 0UL; i < ipos; i += 2UL) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] -= (~rhs)[i];              // NOLINT
    v_[i + 1UL] -= (~rhs)[i + 1UL];  // NOLINT
  }
  if (ipos < (~rhs).size()) {
    // clang-tidy: do not use pointer arithmetic
    v_[ipos] -= (~rhs)[ipos];  // NOLINT
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline std::enable_if_t<(PointerVector<Type, AF, PF, TF, ExprResultType>::
                             BLAZE_TEMPLATE VectorizedSubAssign<VT>::value)>
PointerVector<Type, AF, PF, TF, ExprResultType>::subAssign(
    const blaze::DenseVector<VT, TF>& rhs) {
  BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);

  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const size_t ipos(size_ & size_t(-SIMDSIZE));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % SIMDSIZE)) == ipos,
                        "Invalid end calculation");

  const size_t i4way(size_ & size_t(-SIMDSIZE * 4));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % (SIMDSIZE * 4UL))) == i4way,
                        "Invalid end calculation");
  BLAZE_INTERNAL_ASSERT(i4way <= ipos, "Invalid end calculation");

  size_t i(0UL);
  blaze_const_iterator_t<VT> it((~rhs).begin());

  for (; i < i4way; i += SIMDSIZE * 4UL) {
    store(i, load(i) - it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE, load(i + SIMDSIZE) - it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE * 2UL, load(i + SIMDSIZE * 2UL) - it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE * 3UL, load(i + SIMDSIZE * 3UL) - it.load());
    it += SIMDSIZE;
  }
  for (; i < ipos; i += SIMDSIZE, it += SIMDSIZE) {
    store(i, load(i) - it.load());
  }
  for (; i < size_; ++i, ++it) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] -= *it;  // NOLINT
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline void PointerVector<Type, AF, PF, TF, ExprResultType>::subAssign(
    const blaze::SparseVector<VT, TF>& rhs) {
  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  for (blaze_const_iterator_t<VT> element = (~rhs).begin();
       element != (~rhs).end(); ++element) {
    v_[element->index()] -= element->value();
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline std::enable_if_t<not(PointerVector<Type, AF, PF, TF, ExprResultType>::
                                BLAZE_TEMPLATE VectorizedMultAssign<VT>::value)>
PointerVector<Type, AF, PF, TF, ExprResultType>::multAssign(
    const blaze::DenseVector<VT, TF>& rhs) {
  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const size_t ipos(size_ & size_t(-2));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % 2UL)) == ipos,
                        "Invalid end calculation");

  for (size_t i = 0UL; i < ipos; i += 2UL) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] *= (~rhs)[i];              // NOLINT
    v_[i + 1UL] *= (~rhs)[i + 1UL];  // NOLINT
  }
  if (ipos < (~rhs).size()) {
    // clang-tidy: do not use pointer arithmetic
    v_[ipos] *= (~rhs)[ipos];  // NOLINT
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline std::enable_if_t<(PointerVector<Type, AF, PF, TF, ExprResultType>::
                             BLAZE_TEMPLATE VectorizedMultAssign<VT>::value)>
PointerVector<Type, AF, PF, TF, ExprResultType>::multAssign(
    const blaze::DenseVector<VT, TF>& rhs) {
  BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);

  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const size_t ipos(size_ & size_t(-SIMDSIZE));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % SIMDSIZE)) == ipos,
                        "Invalid end calculation");

  const size_t i4way(size_ & size_t(-SIMDSIZE * 4));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % (SIMDSIZE * 4UL))) == i4way,
                        "Invalid end calculation");
  BLAZE_INTERNAL_ASSERT(i4way <= ipos, "Invalid end calculation");

  size_t i(0UL);
  blaze_const_iterator_t<VT> it((~rhs).begin());

  for (; i < i4way; i += SIMDSIZE * 4UL) {
    store(i, load(i) * it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE, load(i + SIMDSIZE) * it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE * 2UL, load(i + SIMDSIZE * 2UL) * it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE * 3UL, load(i + SIMDSIZE * 3UL) * it.load());
    it += SIMDSIZE;
  }
  for (; i < ipos; i += SIMDSIZE, it += SIMDSIZE) {
    store(i, load(i) * it.load());  // LCOV_EXCL_LINE
  }
  for (; i < size_; ++i, ++it) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] *= *it;  // NOLINT
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline void PointerVector<Type, AF, PF, TF, ExprResultType>::multAssign(
    const blaze::SparseVector<VT, TF>& rhs) {
  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const blaze::DynamicVector<Type, TF> tmp(serial(*this));
  reset();
  for (blaze_const_iterator_t<VT> element = (~rhs).begin();
       element != (~rhs).end(); ++element) {
    v_[element->index()] = tmp[element->index()] * element->value();
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline std::enable_if_t<not(PointerVector<Type, AF, PF, TF, ExprResultType>::
                                BLAZE_TEMPLATE VectorizedDivAssign<VT>::value)>
PointerVector<Type, AF, PF, TF, ExprResultType>::divAssign(
    const blaze::DenseVector<VT, TF>& rhs) {
  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const size_t ipos(size_ & size_t(-2));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % 2UL)) == ipos,
                        "Invalid end calculation");

  for (size_t i = 0UL; i < ipos; i += 2UL) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] /= (~rhs)[i];              // NOLINT
    v_[i + 1UL] /= (~rhs)[i + 1UL];  // NOLINT
  }
  if (ipos < (~rhs).size()) {
    // clang-tidy: do not use pointer arithmetic
    v_[ipos] /= (~rhs)[ipos];  // NOLINT
  }
}

template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
          typename ExprResultType>
template <typename VT>
inline std::enable_if_t<(PointerVector<Type, AF, PF, TF, ExprResultType>::
                             BLAZE_TEMPLATE VectorizedDivAssign<VT>::value)>
PointerVector<Type, AF, PF, TF, ExprResultType>::divAssign(
    const blaze::DenseVector<VT, TF>& rhs) {
  BLAZE_CONSTRAINT_MUST_BE_VECTORIZABLE_TYPE(Type);

  BLAZE_INTERNAL_ASSERT(size_ == (~rhs).size(), "Invalid vector sizes");

  const size_t ipos(size_ & size_t(-SIMDSIZE));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % SIMDSIZE)) == ipos,
                        "Invalid end calculation");

  const size_t i4way(size_ & size_t(-SIMDSIZE * 4));
  BLAZE_INTERNAL_ASSERT((size_ - (size_ % (SIMDSIZE * 4UL))) == i4way,
                        "Invalid end calculation");
  BLAZE_INTERNAL_ASSERT(i4way <= ipos, "Invalid end calculation");

  size_t i(0UL);
  blaze_const_iterator_t<VT> it((~rhs).begin());

  for (; i < i4way; i += SIMDSIZE * 4UL) {
    store(i, load(i) / it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE, load(i + SIMDSIZE) / it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE * 2UL, load(i + SIMDSIZE * 2UL) / it.load());
    it += SIMDSIZE;
    store(i + SIMDSIZE * 3UL, load(i + SIMDSIZE * 3UL) / it.load());
    it += SIMDSIZE;
  }
  for (; i < ipos; i += SIMDSIZE, it += SIMDSIZE) {
    store(i, load(i) / it.load());  // LCOV_EXCL_LINE
  }
  for (; i < size_; ++i, ++it) {
    // clang-tidy: do not use pointer arithmetic
    v_[i] /= *it;  // NOLINT
  }
}
/// \endcond

#define CHECK_FOR_SIZE_ZERO(SCALAR_TYPE)                                    \
  template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,   \
            typename ExprResultType>                                        \
  inline bool operator==(                                                   \
      const PointerVector<Type, AF, PF, TF, ExprResultType>& a,             \
      const SCALAR_TYPE& b) noexcept {                                      \
    ASSERT(a.size() != 0,                                                   \
           "Comparing an empty vector to a value usually indicates a bug, " \
           "and so is disallowed.");                                        \
    return static_cast<const typename PointerVector<                        \
               Type, AF, PF, TF, ExprResultType>::BaseType&>(a) == b;       \
  }                                                                         \
                                                                            \
  template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,   \
            typename ExprResultType>                                        \
  inline bool operator==(                                                   \
      const SCALAR_TYPE& a,                                                 \
      const PointerVector<Type, AF, PF, TF, ExprResultType>& b) noexcept {  \
    return b == a;                                                          \
  }                                                                         \
                                                                            \
  template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,   \
            typename ExprResultType>                                        \
  inline bool operator!=(                                                   \
      const PointerVector<Type, AF, PF, TF, ExprResultType>& a,             \
      const SCALAR_TYPE& b) noexcept {                                      \
    return not(a == b);                                                     \
  }                                                                         \
                                                                            \
  template <typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,   \
            typename ExprResultType>                                        \
  inline bool operator!=(                                                   \
      const SCALAR_TYPE& a,                                                 \
      const PointerVector<Type, AF, PF, TF, ExprResultType>& b) noexcept {  \
    return not(a == b);                                                     \
  }

CHECK_FOR_SIZE_ZERO(double)
CHECK_FOR_SIZE_ZERO(std::complex<double>)
#undef CHECK_FOR_SIZE_ZERO

// There is a bug either in Blaze or in vector intrinsics implementation in GCC
// that results in _mm_set1_epi64 not being callable with an `unsigned long`.
// The way to work around this is to use a forwarding reference (which is super
// aggressive and matches everything), convert the exponent to a double, and
// then call the double pow.
template <
    typename Type, AlignmentFlag_t AF, PaddingFlag_t PF, bool TF,
    typename ExprResultType, typename T,
    typename = std::enable_if_t<std::is_fundamental<std::decay_t<T>>::value>>
decltype(auto) pow(const PointerVector<Type, AF, PF, TF, ExprResultType>& t,
                   T&& exponent) noexcept {
  using ReturnType =
      const blaze::DVecMapExpr<PointerVector<Type, AF, PF, TF, ExprResultType>,
                               BlazePow<double>, TF>;
#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION >= 6))
  return ReturnType(
      t, BlazePow<double>{blaze::Pow{}, static_cast<double>(exponent)});
#else
  return ReturnType(t, BlazePow<double>{static_cast<double>(exponent)});
#endif
}

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <type_traits>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace Parallel {
/*!
 * \ingroup ParallelGroup
 * \brief The array index used for indexing Chare Arrays, mostly an
 * implementation detail
 *
 * The implementation is generic and can handle custom array indices. This
 * replaces the generated, hard-coded Charm++ array indices with a template,
 * allowing a single implementation to be used for different array indices.
 *
 * \details Charm++ allocates memory for `CkArrayIndex`. The size can be
 * configured (in the Charm++ configuration) and defaults to the size of three
 * integers. We place the `Index` into this buffer using placement `new`. Then,
 * `CkArrayIndex::data()` can be safely reinterpreted as an `Index*`.
 */
template <class Index>
struct ArrayIndex
#if defined(SPECTRE_USE_CHARM)
    : public CkArrayIndex
#endif
{
  static_assert(std::is_standard_layout_v<Index> and std::is_trivial_v<Index>,
                "The array index type must be a POD, plain-old-data");
  // clang-tidy: suspicious use of sizeof
  static_assert(sizeof(Index) <= 2 * sizeof(int),  // NOLINT
                "The maximum index size is 64 bits or 8 bytes. Your Index type "
                "is too large.");
  // clang-tidy: suspicious use of sizeof
  static_assert(sizeof(Index) % sizeof(int) == 0,  // NOLINT
                "The Charm++ array Index type must be exactly a multiple of "
                "the size of an integer, but the user-provided one is not.");
#if defined(SPECTRE_USE_CHARM)
  static_assert(
      alignof(Index) == alignof(decltype(index)),
      "Incorrect alignment of Charm++ array Index type. The "
      "alignment must match the alignment of the internal Charm++ type");
#endif
  static_assert(not tt::is_a_v<ArrayIndex, Index>,
                "The Index type passed to ArrayIndex cannot be an ArrayIndex");

  // Use placement new to ensure that the custom index object is placed in the
  // memory reserved for it in the base class
  // clang-tidy: mark explicit: it's a conversion constructor
  ArrayIndex(const Index& array_index)  // NOLINT
#if defined(SPECTRE_USE_CHARM)
      : array_index_(new (index) Index(array_index)) {  // NOLINT
    // clang-tidy: suspicious use of sizeof
    nInts = sizeof(array_index) / sizeof(int);  // NOLINT
  }
#elif defined(SPECTRE_USE_FINDUS)
      : array_index_(0) {
    std::memcpy(&array_index_, &array_index, sizeof(array_index_));
  }
#endif

  // clang-tidy: mark explicit: it's a conversion constructor
  ArrayIndex(const CkArrayIndex& array_index)  // NOLINT
#if defined(SPECTRE_USE_CHARM)
      : CkArrayIndex(array_index),
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        array_index_(reinterpret_cast<Index*>(CkArrayIndex::data())) {
    ASSERT(CkArrayIndex::nInts * sizeof(int) == sizeof(Index),
           "The CkArrayIndex::nInts does not match the size of the custom "
           "array index class.");
  }
#elif defined(SPECTRE_USE_FINDUS)
      : array_index_(0) {
    static_assert(sizeof(array_index_) == sizeof(CkArrayIndex));
    std::memcpy(&array_index_, &array_index, sizeof(array_index_));
  }
#endif

  ArrayIndex(const ArrayIndex& rhs) = delete;
  ArrayIndex& operator=(const ArrayIndex& rhs) = delete;

  ArrayIndex(ArrayIndex&& /*rhs*/) = delete;
  ArrayIndex& operator=(ArrayIndex&& /*rhs*/) = delete;
  ~ArrayIndex() = default;

  const Index& get_index() const {
#if defined(SPECTRE_USE_CHARM)
    return *array_index_;
#elif defined(SPECTRE_USE_FINDUS)
    return array_index_;
#endif
  }

 private:
#if defined(SPECTRE_USE_CHARM)
  Index* array_index_ = nullptr;
#elif defined(SPECTRE_USE_FINDUS)
  std::uint64_t array_index_ = 0;
#endif
};

using ArrayIndex1D = ArrayIndex<CkIndex1D>;
using ArrayIndex2D = ArrayIndex<CkIndex2D>;
using ArrayIndex3D = ArrayIndex<CkIndex3D>;
using ArrayIndex4D = ArrayIndex<CkIndex4D>;
using ArrayIndex5D = ArrayIndex<CkIndex5D>;
using ArrayIndex6D = ArrayIndex<CkIndex6D>;
}  // namespace Parallel

// These namespaces have silly names because we are subverting the charm++
// utilities that prepend "CkArrayIndex" to the name of the array index. See
// comments in Parallel/Algorithms/AlgorithmArray.ci for further explanation.

namespace CkArrayIndexSpectreIndex_detail {
template <typename Index>
using ArrayIndex = ::Parallel::ArrayIndex<Index>;
}  // namespace CkArrayIndexSpectreIndex_detail

namespace SpectreIndex_detail {
template <typename Index>
using ArrayIndex = Index;
}  // namespace SpectreIndex_detail

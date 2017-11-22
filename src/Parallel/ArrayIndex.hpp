// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <type_traits>

namespace Parallel {
/*!
 * \ingroup ParallelGroup
 * \brief The array index used for indexing Chare Arrays, mostly an
 * implementation detail
 *
 * The implementation is generic and can handle custom array indices. This
 * replaces the generated, hard-coded Charm++ array indices with a template,
 * allowing a single implementation to be used for different array indices.
 */
template <class Index>
struct ArrayIndex : public CkArrayIndex {
  static_assert(std::is_pod<Index>::value,
                "The array index type must be a POD, plain-old-data");
  // clang-tidy: suspicious use of sizeof
  static_assert(sizeof(Index) / sizeof(int) <= 3,  // NOLINT
                "The default Charm++ CK_ARRAYINDEX_MAXLEN is 3. If you have "
                "changed this at Charm++ configuration time then please update "
                "the static_assert, otherwise your Index type is too large.");
  // clang-tidy: suspicious use of sizeof
  static_assert(sizeof(Index) % sizeof(int) == 0,  // NOLINT
                "The Charm++ array Index type must be exactly a multiple of "
                "the size of an integer, but the user-provided one is not.");
  static_assert(
      alignof(Index) == alignof(decltype(index)),
      "Incorrect alignment of Charm++ array Index type. The "
      "alignment must match the alignment of the internal Charm++ type");

  // Use placement new to ensure that the custom index object is placed in the
  // memory reserved for it in the base class
  // clang-tidy: mark explicit: it's a conversion constructor
  ArrayIndex(const Index& array_index)  // NOLINT
                                        // clang-tidy: do not use unions
      : array_index_(new (index) Index(array_index)) {  // NOLINT
    // clang-tidy: suspicious use of sizeof
    nInts = sizeof(array_index) / sizeof(int);  // NOLINT
  }

  ArrayIndex(const ArrayIndex& rhs) = delete;
  ArrayIndex& operator=(const ArrayIndex& rhs) = delete;

  ArrayIndex(ArrayIndex&& /*rhs*/) noexcept = delete;
  ArrayIndex& operator=(ArrayIndex&& /*rhs*/) noexcept = delete;
  ~ArrayIndex() = default;

 private:
  Index* array_index_ = nullptr;
};

using ArrayIndex1D = ArrayIndex<CkIndex1D>;
using ArrayIndex2D = ArrayIndex<CkIndex2D>;
using ArrayIndex3D = ArrayIndex<CkIndex3D>;
using ArrayIndex4D = ArrayIndex<CkIndex4D>;
using ArrayIndex5D = ArrayIndex<CkIndex5D>;
using ArrayIndex6D = ArrayIndex<CkIndex6D>;
}  // namespace Parallel

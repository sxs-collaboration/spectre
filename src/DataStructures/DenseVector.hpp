// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DenseVector.

#pragma once

#include <pup.h>  // IWYU pragma: keep

#include "Options/Options.hpp"
#include "Utilities/Blaze.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TypeTraits.hpp"

#include <blaze/math/DynamicVector.h>
#include <blaze/system/Version.h>

/*!
 * \ingroup DataStructuresGroup
 * \brief A dynamically sized vector of arbitrary type.
 *
 * \details Use this vector type to represent a generic collection of numbers
 * without assigning any particular meaning to them. It supports all common
 * vector operations such as elementwise arithmetic, dot-products between
 * vectors and multiplication by a DenseMatrix.
 *
 * \note This is a thin wrapper around `blaze::DynamicVector`. Please refer to
 * the [Blaze documentation](https://bitbucket.org/blaze-lib/blaze/wiki/Home)
 * for information on how to use it.
 *
 * Refer to the \ref DataStructuresGroup documentation for a list of other
 * available vector and matrix types. In particular, to represent the values of
 * a function on the computational domain use DataVector instead.
 */
template <typename T, bool TF = blaze::defaultTransposeFlag>
class DenseVector : public blaze::DynamicVector<T, TF> {
 public:
  // Inherit constructors
  using blaze::DynamicVector<T, TF>::DynamicVector;

  /// Charm++ serialization
  // clang-tidy: runtime-references
  void pup(PUP::er& p) {  // NOLINT
    auto size = blaze::DynamicVector<T, TF>::size();
    p | size;
    if (p.isUnpacking()) {
      blaze::DynamicVector<T, TF>::resize(size);
    }
    if (std::is_fundamental_v<T>) {
      PUParray(p, blaze::DynamicVector<T, TF>::data(), size);
    } else {
      for (auto& element : *this) {
        p | element;
      }
    }
  }
};

namespace MakeWithValueImpls {
template <bool TF>
struct NumberOfPoints<DenseVector<double, TF>> {
  static SPECTRE_ALWAYS_INLINE size_t
  apply(const DenseVector<double, TF>& input) {
    return input.size();
  }
};

template <bool TF>
struct MakeWithSize<DenseVector<double, TF>> {
  /// \brief Returns a `DenseVector` with the given size, with each element
  /// equal to `value`.
  static SPECTRE_ALWAYS_INLINE DenseVector<double, TF> apply(
      const size_t size, const double value) {
    return DenseVector<double, TF>(size, value);
  }
};
}  // namespace MakeWithValueImpls

template <typename T, bool TF>
struct Options::create_from_yaml<DenseVector<T, TF>> {
  template <typename Metavariables>
  static DenseVector<T, TF> create(const Options::Option& options) {
    const auto data = options.parse_as<std::vector<T>>();
    DenseVector<T, TF> result(data.size());
    std::copy(std::begin(data), std::end(data), result.begin());
    return result;
  }
};

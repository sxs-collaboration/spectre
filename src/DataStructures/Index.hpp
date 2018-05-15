// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Index.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <ostream>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep
namespace PUP {
class er;
}  // namespace PUP

/// \ingroup DataStructuresGroup
/// An integer multi-index.
///
/// \tparam Dim the number of integers in the Index.
template <size_t Dim>
class Index {
 public:
  /// Construct with each element set to the same value.
  explicit Index(const size_t i0 = std::numeric_limits<size_t>::max()) noexcept
      : indices_(make_array<Dim>(i0)) {}

  /// Construct specifying value in each dimension
  template <typename... I, Requires<(sizeof...(I) > 1)> = nullptr>
  explicit Index(I... i) noexcept
      : indices_(make_array(static_cast<size_t>(i)...)) {
    static_assert(cpp17::conjunction_v<tt::is_integer<I>...>,
                  "You must pass in a set of size_t's to Index.");
    static_assert(Dim == sizeof...(I),
                  "The number of indices given to Index must be the same as "
                  "the dimensionality of the Index.");
  }

  explicit Index(std::array<size_t, Dim> i) noexcept : indices_(std::move(i)) {}

  size_t operator[](const size_t d) const noexcept {
    return gsl::at(indices_, d);
  }
  size_t& operator[](const size_t d) noexcept { return gsl::at(indices_, d); }

  typename std::array<size_t, Dim>::iterator begin() {
    return indices_.begin();
  }
  typename std::array<size_t, Dim>::const_iterator begin() const {
    return indices_.begin();
  }

  typename std::array<size_t, Dim>::iterator end() { return indices_.end(); }
  typename std::array<size_t, Dim>::const_iterator end() const {
    return indices_.end();
  }

  size_t size() const noexcept { return Dim; }

  /// The product of the indices.
  /// If Dim = 0, the product is defined as 1.
  template <int N = Dim, Requires<(N > 0)> = nullptr>
  constexpr size_t product() const noexcept {
    return indices_[N - 1] * product<N - 1>();
  }
  /// \cond
  // Specialization for N = 0 to stop recursion
  template <int N = Dim, Requires<(N == 0)> = nullptr>
  constexpr size_t product() const noexcept {
    return 1;
  }
  /// \endcond

  /// Return a smaller Index with the d-th element removed.
  ///
  /// \param d the element to remove.
  template <size_t N = Dim, Requires<(N > 0)> = nullptr>
  Index<Dim - 1> slice_away(const size_t d) const noexcept {
    ASSERT(d < Dim,
           "Can't slice dimension " << d << " from an Index<" << Dim << ">");
    std::array<size_t, Dim - 1> t{};
    for (size_t i = 0; i < Dim; ++i) {
      if (i < d) {
        gsl::at(t, i) = gsl::at(indices_, i);
      } else if (i > d) {
        gsl::at(t, i - 1) = gsl::at(indices_, i);
      }
    }
    return Index<Dim - 1>(t);
  }

  /// \cond
  // clang-tidy: runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT
  /// \endcond

  template <size_t N>
  friend std::ostream& operator<<(std::ostream& os,  // NOLINT
                                  const Index<N>& i);

  const size_t* data() const noexcept { return indices_.data(); }
  size_t* data() noexcept { return indices_.data(); }

  const std::array<size_t, Dim>& indices() const noexcept { return indices_; }

 private:
  std::array<size_t, Dim> indices_;
};

/// \ingroup DataStructuresGroup
/// Get the collapsed index into a 1D array of the data corresponding to this
/// Index. Note that the first dimension of the Index varies fastest when
/// computing the collapsed index.
template <size_t N>
size_t collapsed_index(const Index<N>& index, const Index<N>& extents) noexcept;

template <size_t N>
std::ostream& operator<<(std::ostream& os, const Index<N>& i);

/// \cond HIDDEN_SYMBOLS
template <size_t N>
bool operator==(const Index<N>& lhs, const Index<N>& rhs) noexcept;

template <size_t N>
bool operator!=(const Index<N>& lhs, const Index<N>& rhs) noexcept;
/// \endcond

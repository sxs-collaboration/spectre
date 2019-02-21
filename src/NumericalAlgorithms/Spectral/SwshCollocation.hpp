// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdlib>
#include <memory>
#include <sharp_cxx.h>

#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
namespace Swsh {

/// \ingroup SwshGroup
/// \brief Convenience function for determining the number of spin-weighted
/// spherical harmonic collocation values that are stored for a given `l_max`
/// for a libsharp-compatible set of collocation points.
constexpr SPECTRE_ALWAYS_INLINE size_t
number_of_swsh_collocation_points(const size_t l_max) noexcept {
  return (l_max + 1) * (2 * l_max + 1);
}

// In the static caching mechanism, we permit an l_max up to this macro
// value. Higher l_max values may still be created manually using the
// `Collocation` constructor. If l_max's are used several times at higher value,
// consider increasing this value, but only after memory costs have been
// evaluated.
constexpr size_t collocation_maximum_l_max = 200;

namespace detail {
// Helping functor to appropriately delete a stored `sharp_geom_info**`
struct DestroySharpGeometry {
  void operator()(sharp_geom_info* to_delete) noexcept {
    sharp_destroy_geom_info(to_delete);
  }
};
}  // namespace detail

/// A container for reporting a single collocation point for libsharp compatible
/// data structures
struct LibsharpCollocationPoint {
  size_t offset;
  double theta;
  double phi;
};

/// \brief A wrapper class for the spherical harmonic library collocation data
///
/// \details The currently chosen library for spin-weighted spherical harmonic
/// transforms is libsharp. The `Collocation` class stores the
/// libsharp `sharp_geom_info` object, which contains data about
/// 1. The angular collocation points used in spin-weighted spherical harmonic
/// transforms
/// 2. The memory representation of double-type values at those collocation
/// points
///
/// \tparam Representation the ComplexRepresentation, either
/// `ComplexRepresentation::Interleaved` or
/// `ComplexRepresentation::RealsThenImags` compatible with the generated
/// `Collocation` - this is necessary because the stored libsharp type contains
/// memory stride information.
template <ComplexRepresentation Representation>
class Collocation {
 public:
  class CollocationConstIterator {
   public:
    /// create a new iterator. defaults to the start of the supplied object
    explicit CollocationConstIterator(
        const gsl::not_null<const Collocation<Representation>*> collocation,
        const size_t start_index = 0) noexcept
        : index_{start_index}, collocation_{collocation} {}

    /// recovers the data at the collocation point using a
    /// `LibsharpCollocationPoint`, which stores the vector `offset` of the
    /// location of the `theta`, `phi` point in libsharp compatible data
    LibsharpCollocationPoint operator*() const noexcept {
      return LibsharpCollocationPoint{index_, collocation_->theta(index_),
                                      collocation_->phi(index_)};
    }
    /// advance the iterator by one position (prefix)
    CollocationConstIterator& operator++() noexcept {
      ++index_;
      return *this;
    };
    /// advance the iterator by one position (postfix)
    // clang-tidy wants this to return a const iterator
    CollocationConstIterator operator++(int)noexcept {  // NOLINT
      return CollocationConstIterator(collocation_, index_++);
    }

    /// retreat the iterator by one position (prefix)
    CollocationConstIterator& operator--() noexcept {
      --index_;
      return *this;
    };
    /// retreat the iterator by one position (prefix)
    // clang-tidy wants this to return a const iterator
    CollocationConstIterator operator--(int)noexcept {  // NOLINT
      return CollocationConstIterator(collocation_, index_--);
    }

    // @{
    /// (In)Equivalence checks both the object and index for the iterator
    bool operator==(const CollocationConstIterator& rhs) const noexcept {
      return index_ == rhs.index_ and collocation_ == rhs.collocation_;
    }
    bool operator!=(const CollocationConstIterator& rhs) const noexcept {
      return not(*this == rhs);
    }
    // @}

   private:
    size_t index_;
    const Collocation<Representation>* const collocation_;
  };

  /// The representation of the block of complex values, which sets the stride
  /// inside the libsharp type
  static constexpr ComplexRepresentation complex_representation =
      Representation;

  /// \brief Generates the libsharp collocation information and stores it in
  /// `geom_info_`.
  ///
  /// \note If you will potentially use the same `l_max` collocation set more
  /// than once, it is probably better to use the
  /// `precomputed_spherical_harmonic_collocation` function
  explicit Collocation(size_t l_max) noexcept;

  /// default constructor required for iterator use
  ~Collocation() = default;
  Collocation() = default;
  Collocation(const Collocation&) = default;
  Collocation(Collocation&&) = default;
  Collocation& operator=(Collocation&) = default;
  Collocation& operator=(Collocation&&) = default;

  /// retrieve the `sharp_geom_info*` stored. This should largely be used only
  /// for passing to other libsharp functions. Otherwise, access elements
  /// through iterator or access functions.
  sharp_geom_info* get_sharp_geom_info() const noexcept {
    return geom_info_.get();
  }

  /// Retrieve the \f$\theta\f$ value for a given index in a libsharp-compatible
  /// array
  double theta(size_t offset) const noexcept;
  /// Retrieve the \f$\phi\f$ value for a given index in a libsharp-compatible
  /// array
  double phi(size_t offset) const noexcept;

  constexpr size_t l_max() const noexcept { return l_max_; }

  /// Compute the number of entries the libsharp-compatible data structure
  /// should have
  constexpr size_t size() const noexcept {
    return (l_max_ + 1) * (2 * l_max_ + 1);
  }

  // @{
  /// Get a bidirectional iterator to the start of the grid. `operator*` for
  /// that iterator gives a `LibsharpCollocationPoint` with members `offset`,
  /// `theta`, and `phi` with operator *
  Collocation<Representation>::CollocationConstIterator begin() const noexcept {
    return CollocationConstIterator{make_not_null(this), 0};
  }
  Collocation<Representation>::CollocationConstIterator cbegin() const
      noexcept {
    return begin();
  }
  // @}
  // @{
  /// Get a bidirectional iterator to the end of the grid. `operator*` for
  /// that iterator gives a `LibsharpCollocationPoint` with members `offset`,
  /// `theta`, and `phi` with operator *
  Collocation<Representation>::CollocationConstIterator end() const noexcept {
    return CollocationConstIterator{make_not_null(this), size()};
  }
  Collocation<Representation>::CollocationConstIterator cend() const noexcept {
    return end();
  }
  // @}

 private:
  // an extra pointer layer is required for the peculiar way that libsharp
  // constructs these values.
  size_t l_max_ = 0;
  std::unique_ptr<sharp_geom_info, detail::DestroySharpGeometry> geom_info_;
};

/// \brief precomputation function for those collocation grids that are
/// requested
///
/// \details keeps a compile-time structure which acts as a thread-safe lookup
/// table for all l_max values that have been requested so far during execution,
/// so that the libsharp generation need not be re-run. If it has been
/// generated, it's returned by reference. Otherwise, the new grid is generated
/// and put in the lookup table before it is returned by reference.
template <ComplexRepresentation Representation>
const Collocation<Representation>& precomputed_collocation(
    size_t l_max) noexcept;
}  // namespace Swsh
}  // namespace Spectral

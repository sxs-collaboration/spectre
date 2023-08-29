// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <sharp_cxx.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshSettings.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
namespace Swsh {

/// \ingroup SwshGroup
/// \brief Convenience function for determining the number of spin-weighted
/// spherical harmonic collocation values that are stored for a given `l_max`
/// for a libsharp-compatible set of collocation points.
constexpr SPECTRE_ALWAYS_INLINE size_t
number_of_swsh_collocation_points(const size_t l_max) {
  return (l_max + 1) * (2 * l_max + 1);
}

/// \ingroup SwshGroup
/// \brief Returns the number of spin-weighted spherical harmonic collocation
/// values in \f$\theta\f$ for a libsharp-compatible set of collocation
/// points.
///
/// \details The full number of collocation points is the product of the number
/// of \f$\theta\f$ points and the number of \f$\phi\f$ points (a 'rectangular'
/// grid).
constexpr SPECTRE_ALWAYS_INLINE size_t
number_of_swsh_theta_collocation_points(const size_t l_max) {
  return (l_max + 1);
}

/// \ingroup SwshGroup
/// \brief Returns the number of spin-weighted spherical harmonic collocation
/// values in \f$\phi\f$ for a libsharp-compatible set of collocation
/// points.
///
/// \details The full number of collocation points is the product of the number
/// of \f$\theta\f$ points and the number of \f$\phi\f$ points (a 'rectangular'
/// grid).
constexpr SPECTRE_ALWAYS_INLINE size_t
number_of_swsh_phi_collocation_points(const size_t l_max) {
  return (2 * l_max + 1);
}

/// \ingroup SwshGroup
/// \brief Obtain the three-dimensional mesh associated with a
/// libsharp-compatible sequence of spherical nodal shells.
/// \warning This is to be used only for operations in the radial direction! The
/// angular collocation points should not be regarded as having a useful product
/// representation for e.g. taking derivatives in just the theta direction.
/// Instead, use spin-weighted utilities for all angular operations.
SPECTRE_ALWAYS_INLINE Mesh<3> swsh_volume_mesh_for_radial_operations(
    const size_t l_max, const size_t number_of_radial_points) {
  return Mesh<3>{{{number_of_swsh_phi_collocation_points(l_max),
                   number_of_swsh_theta_collocation_points(l_max),
                   number_of_radial_points}},
                 Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}

namespace detail {
// Helping functor to appropriately delete a stored `sharp_geom_info**`
struct DestroySharpGeometry {
  void operator()(sharp_geom_info* to_delete) {
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
/// transforms is libsharp. The `CollocationMetadata` class stores the
/// libsharp `sharp_geom_info` object, which contains data about
/// 1. The angular collocation points used in spin-weighted spherical harmonic
/// transforms
/// 2. The memory representation of double-type values at those collocation
/// points
///
/// \tparam Representation the ComplexRepresentation, either
/// `ComplexRepresentation::Interleaved` or
/// `ComplexRepresentation::RealsThenImags` compatible with the generated
/// `CollocationMetadata` - this is necessary because the stored libsharp type
/// contains memory stride information.
template <ComplexRepresentation Representation>
class CollocationMetadata {
 public:
  class CollocationConstIterator {
   public:
    /// create a new iterator. defaults to the start of the supplied object
    explicit CollocationConstIterator(
        const gsl::not_null<const CollocationMetadata<Representation>*>
            collocation,
        const size_t start_index = 0)
        : index_{start_index}, collocation_{collocation} {}

    /// recovers the data at the collocation point using a
    /// `LibsharpCollocationPoint`, which stores the vector `offset` of the
    /// location of the `theta`, `phi` point in libsharp compatible data
    LibsharpCollocationPoint operator*() const {
      return LibsharpCollocationPoint{index_, collocation_->theta(index_),
                                      collocation_->phi(index_)};
    }
    /// advance the iterator by one position (prefix)
    CollocationConstIterator& operator++() {
      ++index_;
      return *this;
    };
    /// advance the iterator by one position (postfix)
    // clang-tidy wants this to return a const iterator
    CollocationConstIterator operator++(int) {  // NOLINT
      return CollocationConstIterator(collocation_, index_++);
    }

    /// retreat the iterator by one position (prefix)
    CollocationConstIterator& operator--() {
      --index_;
      return *this;
    };
    /// retreat the iterator by one position (prefix)
    // clang-tidy wants this to return a const iterator
    CollocationConstIterator operator--(int) {  // NOLINT
      return CollocationConstIterator(collocation_, index_--);
    }

    /// @{
    /// (In)Equivalence checks both the object and index for the iterator
    bool operator==(const CollocationConstIterator& rhs) const {
      return index_ == rhs.index_ and collocation_ == rhs.collocation_;
    }
    bool operator!=(const CollocationConstIterator& rhs) const {
      return not(*this == rhs);
    }
    /// @}

   private:
    size_t index_;
    const CollocationMetadata<Representation>* const collocation_;
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
  explicit CollocationMetadata(size_t l_max);

  /// default constructor required for iterator use
  ~CollocationMetadata() = default;
  CollocationMetadata() = default;
  CollocationMetadata(const CollocationMetadata&) = default;
  CollocationMetadata(CollocationMetadata&&) = default;
  CollocationMetadata& operator=(CollocationMetadata&) = default;
  CollocationMetadata& operator=(CollocationMetadata&&) = default;

  /// retrieve the `sharp_geom_info*` stored. This should largely be used only
  /// for passing to other libsharp functions. Otherwise, access elements
  /// through iterator or access functions.
  sharp_geom_info* get_sharp_geom_info() const { return geom_info_.get(); }

  /// Retrieve the \f$\theta\f$ value for a given index in a libsharp-compatible
  /// array
  double theta(size_t offset) const;
  /// Retrieve the \f$\phi\f$ value for a given index in a libsharp-compatible
  /// array
  double phi(size_t offset) const;

  constexpr size_t l_max() const { return l_max_; }

  /// Compute the number of entries the libsharp-compatible data structure
  /// should have
  constexpr size_t size() const { return (l_max_ + 1) * (2 * l_max_ + 1); }

  /// @{
  /// Get a bidirectional iterator to the start of the grid. `operator*` for
  /// that iterator gives a `LibsharpCollocationPoint` with members `offset`,
  /// `theta`, and `phi`
  CollocationMetadata<Representation>::CollocationConstIterator begin() const {
    return CollocationConstIterator{make_not_null(this), 0};
  }
  CollocationMetadata<Representation>::CollocationConstIterator cbegin() const {
    return begin();
  }
  /// @}
  /// @{
  /// Get a bidirectional iterator to the end of the grid. `operator*` for
  /// that iterator gives a `LibsharpCollocationPoint` with members `offset`,
  /// `theta`, and `phi`
  CollocationMetadata<Representation>::CollocationConstIterator end() const {
    return CollocationConstIterator{make_not_null(this), size()};
  }
  CollocationMetadata<Representation>::CollocationConstIterator cend() const {
    return end();
  }
  /// @}

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
const CollocationMetadata<Representation>& cached_collocation_metadata(
    size_t l_max);

/// \brief Store the libsharp-compatible collocation grid and corresponding
/// unit-sphere cartesian grid in the supplied buffers.
void create_angular_and_cartesian_coordinates(
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_coordinates,
    size_t l_max);
}  // namespace Swsh
}  // namespace Spectral

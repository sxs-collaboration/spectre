// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes representing tensor indices

#pragma once

#include <ostream>
#include <type_traits>

#include "ErrorHandling/Error.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TypeTraits.hpp"

namespace brigand {
template <class...>
struct list;
}  // namespace brigand

/// \ingroup TensorGroup
/// Whether a \ref SpacetimeIndex "TensorIndexType" is covariant,
/// contravariant, or Euclidean
enum class UpLo {
  /// Contravariant, or Upper index
  Up,
  /// Covariant, or Lower index
  Lo,
  /// Euclidean index, upper and lower are not distinguished
  Euclidean
};

/// \ingroup TensorGroup
/// The opposite index position (swapping Up and Lo)
constexpr UpLo opposite(UpLo ul) noexcept {
  switch (ul) {
    case UpLo::Up:
      return UpLo::Lo;
    case UpLo::Lo:
      return UpLo::Up;
    case UpLo::Euclidean:
      return UpLo::Euclidean;
    default:
      return ul;
  }
}

/// \cond HIDDEN_SYMBOLS
inline std::ostream& operator<<(std::ostream& os, const UpLo& ul) {
  switch (ul) {
    case UpLo::Up:
      return os << "Up";
    case UpLo::Lo:
      return os << "Lo";
    case UpLo::Euclidean:
      return os << "Euclidean";
    default:
      ERROR("Invalid UpLo");
  }
}
/// \endcond

/// \ingroup TensorGroup
/// Indicates the ::Frame that a \ref SpacetimeIndex "TensorIndexType"
/// is in.
namespace Frame {
/// \ingroup TensorGroup
/// Marks a Frame as being "physical" in the sense that it is meaningful to
/// evaluate an analytic solution in that frame.
struct FrameIsPhysical {};

struct Logical {};
struct Grid {};
struct Inertial : FrameIsPhysical {};
struct Distorted {};
/// Represents an index that is not in a known frame, e.g. some internal
/// intermediate frame that is irrelevant to the interface.
struct NoFrame {};

/// Represents a spherical-coordinate frame that is associated with a
/// Cartesian frame, e.g. \f$(r,\theta,\phi)\f$ associated with the Inertial
/// frame, as used on an apparent horizon or a wave-extraction surface.
template <typename CartesianFrame>
struct Spherical {};

/// \ingroup TensorGroup
/// Returns std::true_type if the frame is "physical" in the sense that it is
/// meaningful to evaluate an analytic solution in that frame.
/// \example
/// \snippet Test_Tensor.cpp is_frame_physical
template <typename CheckFrame>
using is_frame_physical =
    std::integral_constant<bool,
                           std::is_base_of<FrameIsPhysical, CheckFrame>::value>;

/// \ingroup TensorGroup
/// Returns true if the frame is "physical" in the sense that it is
/// meaningful to evaluate an analytic solution in that frame.
/// \example
/// \snippet Test_Tensor.cpp is_frame_physical
template <typename CheckFrame>
constexpr bool is_frame_physical_v = is_frame_physical<CheckFrame>::value;

/// \cond HIDDEN_SYMBOLS
inline std::ostream& operator<<(std::ostream& os,
                                const Frame::Logical& /*meta*/) {
  return os << "Logical";
}
inline std::ostream& operator<<(std::ostream& os, const Frame::Grid& /*meta*/) {
  return os << "Grid";
}
inline std::ostream& operator<<(std::ostream& os,
                                const Frame::Inertial& /*meta*/) {
  return os << "Inertial";
}
inline std::ostream& operator<<(std::ostream& os,
                                const Frame::Distorted& /*meta*/) {
  return os << "Distorted";
}
inline std::ostream& operator<<(std::ostream& os,
                                const Frame::NoFrame& /*meta*/) {
  return os << "NoFrame";
}
/// \endcond
}  // namespace Frame

/// \ingroup TensorGroup
/// Indicates whether the \ref SpacetimeIndex "TensorIndexType" is
/// Spatial or Spacetime
enum class IndexType : char {
  /// The \ref SpatialIndex "TensorIndexType" is purely spatial
  Spatial,
  /// The \ref SpacetimeIndex "TensorIndexType" is a spacetime index
  Spacetime
};

/// \cond HIDDEN_SYMBOLS
inline std::ostream& operator<<(std::ostream& os, const IndexType& index_type) {
  return os << (index_type == IndexType::Spatial ? "Spatial" : "Spacetime");
}
/// \endcond

namespace Tensor_detail {
/// \ingroup TensorGroup
/// A ::TensorIndexType holds information about what type of index an
/// index of a Tensor is. It holds the information about the number of
/// spatial dimensions, whether the index is covariant, contravariant,
/// or Euclidean (::UpLo), the ::Frame the index is in, and whether
/// the Index is Spatial or Spacetime.
/// \tparam SpatialDim the spatial dimensionality of the TensorIndex
/// \tparam Ul an UpLo describing index position
/// \tparam Fr the Frame the TensorIndex is in
/// \tparam Index either IndexType::Spatial or IndexType::Spacetime
template <size_t SpatialDim, UpLo Ul, typename Fr, IndexType Index>
struct TensorIndexType {
  static_assert(SpatialDim > 0,
                "Cannot have a spatial dimensionality less than 1 (one) in a "
                "TensorIndexType");
  static_assert(not(Ul == UpLo::Euclidean and Index == IndexType::Spacetime),
                "Spacetime should not be treated as Euclidean");
  using value_type = decltype(SpatialDim);
  static constexpr value_type dim =
      Index == IndexType::Spatial ? SpatialDim
                                  : SpatialDim + static_cast<value_type>(1);
  // value is here just so that some generic metafunctions can retrieve the dim
  // easily
  static constexpr value_type value = dim;
  static constexpr UpLo ul = Ul;
  using Frame = Fr;
  static constexpr IndexType index_type = Index;
};
}  // namespace Tensor_detail

/// \ingroup TensorGroup
/// A SpatialIndex holds information about the number of spatial
/// dimensions, whether the index is covariant, contravariant, or
/// Euclidean (::UpLo), and the ::Frame the index is in.  \tparam
/// SpatialDim the spatial dimensionality of the \ref SpatialIndex
/// "TensorIndexType"
/// \tparam Ul an UpLo describing index position
/// \tparam Fr the ::Frame the \ref SpatialIndex "TensorIndexType"
/// is in
template <size_t SpatialDim, UpLo Ul, typename Fr>
using SpatialIndex =
    Tensor_detail::TensorIndexType<SpatialDim, Ul, Fr, IndexType::Spatial>;

/// \ingroup TensorGroup
/// A SpacetimeIndex holds information about the number of spatial
/// dimensions, whether the index is covariant or contravariant (::UpLo), and
/// the ::Frame the index is in.
///
/// \tparam SpatialDim the spatial dimensionality of the \ref
/// SpacetimeIndex "TensorIndexType"
/// \tparam Ul either UpLo::Up or UpLo::Lo for contra or covariant
/// \tparam Fr the ::Frame the \ref SpacetimeIndex "TensorIndexType"
/// is in
template <size_t SpatialDim, UpLo Ul, typename Fr>
using SpacetimeIndex =
    Tensor_detail::TensorIndexType<SpatialDim, Ul, Fr, IndexType::Spacetime>;

namespace tt {
// @{
/// \ingroup TensorGroup TypeTraits
/// Inherits from std::true_type if T is a \ref SpacetimeIndex
/// "TensorIndexType"
/// \tparam T the type to check
template <typename T>
struct is_tensor_index_type : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <size_t SpatialDim, UpLo Ul, typename Fr, IndexType Index>
struct is_tensor_index_type<
    Tensor_detail::TensorIndexType<SpatialDim, Ul, Fr, Index>>
    : std::true_type {};
/// \endcond
/// \see is_tensor_index_type
template <typename T>
using is_tensor_index_type_t = typename is_tensor_index_type<T>::type;
// @}
}  // namespace tt

/// \ingroup TensorGroup
/// Change the \ref SpacetimeIndex "TensorIndexType" to be covariant
/// if it's contravariant and vice-versa.  Euclidean indices are left
/// unchanged.
///
/// Here is an example of how to use ::change_index_up_lo
/// \snippet Test_Tensor.cpp change_up_lo
///
/// \tparam Index the \ref SpacetimeIndex "TensorIndexType" to change
template <typename Index>
using change_index_up_lo = Tensor_detail::TensorIndexType<
    Index::index_type == IndexType::Spatial ? Index::value : Index::value - 1,
    opposite(Index::ul), typename Index::Frame, Index::index_type>;

/// \ingroup TensorGroup
/// Whether contracting a pair of indices makes sense
template <typename IndexA, typename IndexB>
constexpr bool can_contract_v =
    IndexA::index_type == IndexB::index_type and
    IndexA::dim == IndexB::dim and
    cpp17::is_same_v<typename IndexA::Frame, typename IndexB::Frame> and
    (IndexA::ul != IndexB::ul or IndexA::ul == UpLo::Euclidean);

template <typename... Ts>
using index_list = brigand::list<Ts...>;

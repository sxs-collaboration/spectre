// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes representing tensor indices

#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <type_traits>

#include "Utilities/Literals.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \ingroup TensorGroup
/// Whether a \ref SpacetimeIndex "TensorIndexType" is covariant or
/// contravariant
enum class UpLo {
  /// Contravariant, or Upper index
  Up,
  /// Covariant, or Lower index
  Lo
};

/// \cond HIDDEN_SYMBOLS
inline std::ostream& operator<<(std::ostream& os, const UpLo& ul) {
  return os << (ul == UpLo::Up ? "Up"s : "Lo"s);
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

/// \ingroup TensorGroup
/// The frame-dependent prefix used when constructing the string returned by the
/// name function of a tag.
///
/// For Frame::Inertial it is the empty string, otherwise, it is the name of
/// the Frame followed by an underscore (as the name will be used in I/O).
/// \example
/// \snippet Hydro/Test_Tags.cpp prefix_example
template <typename Fr>
inline std::string prefix() noexcept;

/// \cond HIDDEN_SYMBOLS
template <>
inline std::string prefix<Frame::Logical>() noexcept {
  return "Logical_";
}

template <>
inline std::string prefix<Frame::Grid>() noexcept {
  return "Grid_";
}

template <>
inline std::string prefix<Frame::Inertial>() noexcept {
  return "";
}

template <>
inline std::string prefix<Frame::Distorted>() noexcept {
  return "Distorted_";
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
/// A ::TensorIndexType holds information about what type of index an index of a
/// Tensor is. It holds the information about the number of spatial dimensions,
/// whether the index is covariant or contravariant (::UpLo), the ::Frame the
/// index is in, and whether the Index is Spatial or Spacetime.
/// \tparam SpatialDim the spatial dimensionality of the TensorIndex
/// \tparam Ul either UpLo::Up or UpLo::Lo for contra or covariant
/// \tparam Fr the Frame the TensorIndex is in
/// \tparam Index either IndexType::Spatial or IndexType::Spacetime
template <size_t SpatialDim, UpLo Ul, typename Fr, IndexType Index>
struct TensorIndexType {
  static_assert(SpatialDim > 0,
                "Cannot have a spatial dimensionality less than 1 (one) in a "
                "TensorIndexType");
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
/// dimensions, whether the index is covariant or contravariant (::UpLo), and
/// the ::Frame the index is in.
/// \tparam SpatialDim the spatial dimensionality of the \ref
/// SpatialIndex "TensorIndexType"
/// \tparam Ul either UpLo::Up or UpLo::Lo for contra or covariant
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
/// \ingroup TensorGroup TypeTraitsGroup
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
/// if it's contravariant and vice-versa
///
/// Here is an example of how to use ::change_index_up_lo
/// \snippet Test_Tensor.cpp change_up_lo
///
/// \tparam Index the \ref SpacetimeIndex "TensorIndexType" to change
template <typename Index>
using change_index_up_lo = Tensor_detail::TensorIndexType<
    Index::index_type == IndexType::Spatial ? Index::value : Index::value - 1,
    Index::ul == UpLo::Up ? UpLo::Lo : UpLo::Up, typename Index::Frame,
    Index::index_type>;

template <typename... Ts>
using index_list = tmpl::list<Ts...>;

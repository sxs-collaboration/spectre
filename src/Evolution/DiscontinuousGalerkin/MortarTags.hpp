// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionalIdMap;
template <size_t Dim>
class Mesh;
namespace evolution::dg {
template <size_t Dim>
class BoundaryMessage;
template <size_t Dim>
class MortarData;
}  // namespace evolution::dg
namespace Spectral {
enum class ChildSize : uint8_t;
using MortarSize = ChildSize;
}  // namespace Spectral
class TimeStepId;
namespace TimeSteppers {
template <typename LocalData, typename RemoteData, typename CouplingResult>
class BoundaryHistory;
}  // namespace TimeSteppers
/// \endcond

/// %Tags used for DG evolution scheme.
namespace evolution::dg::Tags {
/// Data on mortars, indexed by (Direction, ElementId) pairs
///
/// The `Dim` is the volume dimension, not the face dimension.
template <size_t Dim>
struct MortarData : db::SimpleTag {
  using type = DirectionalIdMap<Dim, evolution::dg::MortarData<Dim>>;
};

/// History of the data on mortars, indexed by (Direction, ElementId) pairs, and
/// used by the linear multistep local time stepping code.
///
/// The `Dim` is the volume dimension, not the face dimension.
///
/// `CouplingResult` is the result of calling a functor of type `Coupling` used
/// in `TimeSteppers::BoundaryHistory`. It is also the result of
/// `LtsTimeStepper::compute_boundary_delta()`, which again has a `Coupling`
/// template parameter.
template <size_t Dim, typename CouplingResult>
struct MortarDataHistory : db::SimpleTag {
  using type = DirectionalIdMap<
      Dim, TimeSteppers::BoundaryHistory<::evolution::dg::MortarData<Dim>,
                                         ::evolution::dg::MortarData<Dim>,
                                         CouplingResult>>;
};

/// Mesh on the mortars, indexed by (Direction, ElementId) pairs
///
/// The `Dim` is the volume dimension, not the face dimension.
template <size_t Dim>
struct MortarMesh : db::SimpleTag {
  using type = DirectionalIdMap<Dim, Mesh<Dim - 1>>;
};

/// Size of a mortar, relative to the element face.  That is, the part
/// of the face that it covers.
///
/// The `Dim` is the volume dimension, not the face dimension.
template <size_t Dim>
struct MortarSize : db::SimpleTag {
  using type = DirectionalIdMap<Dim, std::array<Spectral::MortarSize, Dim - 1>>;
};

/// The next temporal id at which to receive data on the specified mortar.
///
/// The `Dim` is the volume dimension, not the face dimension.
template <size_t Dim>
struct MortarNextTemporalId : db::SimpleTag {
  using type = DirectionalIdMap<Dim, TimeStepId>;
};

/// \brief The BoundaryMessage received from the inbox
///
/// We must store the `std::unique_ptr` in the DataBox so the memory persists in
/// case data was sent from another node
/// \tparam Dim The volume dimension, not the face dimension
template <size_t Dim>
struct BoundaryMessageFromInbox : db::SimpleTag {
  using type = DirectionalIdMap<Dim, std::unique_ptr<BoundaryMessage<Dim>>>;
};
}  // namespace evolution::dg::Tags

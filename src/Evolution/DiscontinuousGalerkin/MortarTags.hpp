// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"  // for MortarSize
#include "Time/BoundaryHistory.hpp"
#include "Time/TimeStepId.hpp"

/// %Tags used for DG evolution scheme.
namespace evolution::dg::Tags {
/// Data on mortars, indexed by (Direction, ElementId) pairs
///
/// The `Dim` is the volume dimension, not the face dimension.
template <size_t Dim>
struct MortarData : db::SimpleTag {
  using Key = std::pair<Direction<Dim>, ElementId<Dim>>;
  using type =
      std::unordered_map<Key, evolution::dg::MortarData<Dim>, boost::hash<Key>>;
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
  using Key = std::pair<Direction<Dim>, ElementId<Dim>>;
  using type =
      std::unordered_map<Key,
                         TimeSteppers::BoundaryHistory<
                             ::evolution::dg::MortarData<Dim>,
                             ::evolution::dg::MortarData<Dim>, CouplingResult>,
                         boost::hash<Key>>;
};

/// Mesh on the mortars, indexed by (Direction, ElementId) pairs
///
/// The `Dim` is the volume dimension, not the face dimension.
template <size_t Dim>
struct MortarMesh : db::SimpleTag {
  using Key = std::pair<Direction<Dim>, ElementId<Dim>>;
  using type = std::unordered_map<Key, Mesh<Dim - 1>, boost::hash<Key>>;
};

/// Size of a mortar, relative to the element face.  That is, the part
/// of the face that it covers.
///
/// The `Dim` is the volume dimension, not the face dimension.
template <size_t Dim>
struct MortarSize : db::SimpleTag {
  using Key = std::pair<Direction<Dim>, ElementId<Dim>>;
  using type =
      std::unordered_map<Key, std::array<Spectral::MortarSize, Dim - 1>,
                         boost::hash<Key>>;
};

/// The next temporal id at which to receive data on the specified mortar.
///
/// The `Dim` is the volume dimension, not the face dimension.
template <size_t Dim>
struct MortarNextTemporalId : db::SimpleTag {
  using Key = std::pair<Direction<Dim>, ElementId<Dim>>;
  using type = std::unordered_map<Key, TimeStepId, boost::hash<Key>>;
};

/// \brief The BoundaryMessage received from the inbox
///
/// We must store the `std::unique_ptr` in the DataBox so the memory persists in
/// case data was sent from another node
/// \tparam Dim The volume dimension, not the face dimension
template <size_t Dim>
struct BoundaryMessageFromInbox : db::SimpleTag {
  using Key = std::pair<Direction<Dim>, ElementId<Dim>>;
  using type = std::unordered_map<Key, std::unique_ptr<BoundaryMessage<Dim>>,
                                  boost::hash<Key>>;
};
}  // namespace evolution::dg::Tags

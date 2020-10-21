// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Utilities/TMPL.hpp"

/// %Tags used on the interior faces of the elements
namespace evolution::dg::Tags::InternalFace {
/// The magnitude of the unnormalized normal covector to the interface
struct MagnitudeOfNormal : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The normal covector to the interface
template <size_t Dim>
struct NormalCovector : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

/// The normal covector and its magnitude for all internal faces of an element.
///
/// The combined tag is used to make the allocations be in a Variables
///
/// We use a `std::optional` to keep track of whether or not these values are
/// up-to-date.
template <size_t Dim>
struct NormalCovectorAndMagnitude : db::SimpleTag {
  using type = DirectionMap<
      Dim, std::optional<
               Variables<tmpl::list<MagnitudeOfNormal, NormalCovector<Dim>>>>>;
};
}  // namespace evolution::dg::Tags::InternalFace

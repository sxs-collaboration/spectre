// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

namespace evolution::dg::subcell::fd::Tags {
/// \brief The inverse Jacobian from the element logical frame to the grid frame
/// at the cell centers.
///
/// Specifically, \f$\partial x^{\bar{i}} / \partial x^i\f$, where \f$\bar{i}\f$
/// denotes the element logical frame and \f$i\f$ denotes the grid frame.
///
/// \note stored as a `std::optional` so we can reset it when switching to DG
/// and reduce the memory footprint.
template <size_t Dim>
struct InverseJacobianLogicalToGrid : db::SimpleTag {
  static std::string name() { return "InverseJacobian(Logical,Grid)"; }
  using type = std::optional<
      ::InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Grid>>;
};

/// \brief The determinant of the inverse Jacobian from the element logical
/// frame to the grid frame at the cell centers.
///
/// \note stored as a `std::optional` so we can reset it when switching to DG
/// and reduce the memory footprint.
struct DetInverseJacobianLogicalToGrid : db::SimpleTag {
  static std::string name() { return "Det(InverseJacobian(Logical,Grid))"; }
  using type = std::optional<Scalar<DataVector>>;
};
}  // namespace evolution::dg::subcell::fd::Tags

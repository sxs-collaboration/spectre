// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Utilities/GetOutput.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg::subcell::Tags {
/// The coordinates in a given frame.
template <size_t Dim, typename Frame>
struct Coordinates : db::SimpleTag {
  static std::string name() { return get_output(Frame{}) + "Coordinates"; }
  using type = tnsr::I<DataVector, Dim, Frame>;
};

/// The logical coordinates on the subcell grid
template <size_t VolumeDim>
struct LogicalCoordinatesCompute
    : Coordinates<VolumeDim, Frame::ElementLogical>,
      db::ComputeTag {
  using base = Coordinates<VolumeDim, Frame::ElementLogical>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<Mesh<VolumeDim>>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<return_type*>, const ::Mesh<VolumeDim>&)>(
      &logical_coordinates<VolumeDim>);
};
}  // namespace evolution::dg::subcell::Tags

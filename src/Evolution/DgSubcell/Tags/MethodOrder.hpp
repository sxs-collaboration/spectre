// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <size_t Dim>
class Mesh;
namespace evolution::dg::subcell {
class SubcellOptions;
}  // namespace evolution::dg::subcell
/// \endcond

namespace evolution::dg::subcell::Tags {
/// @{
/// The order of the numerical method used.
///
/// \note This is intended to be used during observations.
template <size_t Dim>
struct MethodOrder : db::SimpleTag {
  using type = std::optional<tnsr::I<DataVector, Dim, Frame::ElementLogical>>;
};

template <size_t Dim>
struct MethodOrderCompute : db::ComputeTag, MethodOrder<Dim> {
  using base = MethodOrder<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::domain::Tags::Mesh<Dim>, Mesh<Dim>, ActiveGrid,
                 ReconstructionOrder<Dim>, SubcellOptions<Dim>>;
  static void function(
      gsl::not_null<return_type*> method_order, const ::Mesh<Dim>& dg_mesh,
      const ::Mesh<Dim>& subcell_mesh, subcell::ActiveGrid active_grid,
      const std::optional<tnsr::I<DataVector, Dim, Frame::ElementLogical>>&
          reconstruction_order,
      const subcell::SubcellOptions& subcell_options);
};
/// @}
}  // namespace evolution::dg::subcell::Tags

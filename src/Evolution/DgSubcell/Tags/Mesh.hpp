// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/NumberOfSubcellPoints.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::Tags {
/// @{
/// The mesh on the subcells
template <size_t VolumeDim>
struct Mesh : db::SimpleTag {
  static std::string name() { return "Subcell(Mesh)"; }
  using type = ::Mesh<VolumeDim>;
};

template <size_t VolumeDim>
struct MeshCompute : Mesh<VolumeDim>, db::ComputeTag {
  using base = Mesh<VolumeDim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::domain::Tags::Mesh<VolumeDim>,
                 evolution::dg::Tags::NumberOfSubcellPointsCoefficients>;
  static void function(gsl::not_null<return_type*> subcell_mesh,
                       const ::Mesh<VolumeDim>& dg_mesh,
                       const std::array<int, 3>& number_of_points_coefficients);
};
/// @}
}  // namespace evolution::dg::subcell::Tags

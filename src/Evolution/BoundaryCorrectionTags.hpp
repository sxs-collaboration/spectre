// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/OptionTags.hpp"

namespace evolution {
namespace OptionTags {
/// The boundary correction used for coupling the local PDE system solution to
/// solutions from neighboring elements or applying boundary conditions.
///
/// In the finite volume/difference and discontinuous Galerkin literature this
/// is often referred to as the "numerical flux". We avoid that nomenclature
/// because in the discontinuous Galerkin and finite volume case it is not the
/// flux that is modified, but the integrand of the boundary integral.
template <typename System>
struct BoundaryCorrection {
  using type = std::unique_ptr<typename System::boundary_correction_base>;
  using group = SpatialDiscretization::OptionTags::SpatialDiscretizationGroup;
  static constexpr Options::String help = "The boundary correction to use.";
};
}  // namespace OptionTags

namespace Tags {
/// The boundary correction used for coupling together neighboring cells or
/// applying boundary conditions.
///
/// In the finite volume/difference and discontinuous Galerkin literature this
/// is ofter referred to as the "numerical flux". We avoid that nomenclature
/// because in the discontinuous Galerkin and finite volume case it is not the
/// flux that is modified, but the integrand of the boundary integral.
template <typename System>
struct BoundaryCorrection : db::SimpleTag {
  using type = std::unique_ptr<typename System::boundary_correction_base>;

  using option_tags = tmpl::list<OptionTags::BoundaryCorrection<System>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& boundary_correction) noexcept {
    return boundary_correction->get_clone();
  }
};
}  // namespace Tags
}  // namespace evolution

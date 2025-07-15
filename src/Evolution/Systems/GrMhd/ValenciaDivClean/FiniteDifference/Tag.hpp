// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellSolver.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Options/String.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd {
/// Option tags for reconstruction
namespace OptionTags {
/// \brief Option tag for the reconstructor
struct Reconstructor {
  using type = std::unique_ptr<fd::Reconstructor>;

  static constexpr Options::String help = {"The reconstruction scheme to use."};
  using group = evolution::dg::subcell::OptionTags::SubcellSolverGroup;
};
}  // namespace OptionTags

/// %Tags for reconstruction
namespace Tags {
/// \brief Tag for the reconstructor
struct Reconstructor : db::SimpleTag {
  using type = std::unique_ptr<fd::Reconstructor>;
  using option_tags =
      tmpl::list<OptionTags::Reconstructor,
                 ::evolution::dg::subcell::OptionTags::SubcellOptions>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const type& reconstructor,
      const ::evolution::dg::subcell::SubcellOptions& subcell_options) {
    if (static_cast<int>(subcell_options.finite_difference_derivative_order()) <
            0 and
        not reconstructor->supports_adaptive_order()) {
      ERROR_NO_TRACE(
          "Cannot use adaptive finite difference derivative order with "
          "specified reconstructor.");
    }
    if ((static_cast<int>(
             subcell_options.finite_difference_derivative_order()) /
             2 -
         1) > static_cast<int>(reconstructor->ghost_zone_size())) {
      ERROR_NO_TRACE(
          "The derivative order chosen ("
          << subcell_options.finite_difference_derivative_order()
          << ") requires more ghost zones ("
          << (static_cast<int>(
                  subcell_options.finite_difference_derivative_order()) /
                  2 -
              1)
          << ") than the reconstructor sends, "
          << reconstructor->ghost_zone_size());
    }
    return reconstructor->get_clone();
  }
};
}  // namespace Tags
}  // namespace NewtonianEuler::fd

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DgSubcell/Tags/SubcellSolver.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Reconstructor.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::fd {
namespace OptionTags {
/*!
 * \brief Holds the subcell reconstructor in the input file
 */
template <size_t Dim>
struct Reconstructor {
  using type = std::unique_ptr<fd::Reconstructor<Dim>>;

  static constexpr Options::String help = {"The reconstruction scheme to use."};
  using group = evolution::dg::subcell::OptionTags::SubcellSolverGroup;
};
}  // namespace OptionTags

namespace Tags {
/*!
 * \brief Tag for the reconstructor
 */
template <size_t Dim>
struct Reconstructor : db::SimpleTag {
  using type = std::unique_ptr<fd::Reconstructor<Dim>>;
  using option_tags = tmpl::list<OptionTags::Reconstructor<Dim>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& reconstructor) noexcept {
    return reconstructor->get_clone();
  }
};
}  // namespace Tags
}  // namespace ScalarAdvection::fd

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::subcell {
/*!
 * \brief Zeros out the MHD time derivatives in the elements next to a DG-only
 * block that themselves are not DG-only elements.
 */
struct ZeroMhdTimeDerivatives {
  using return_tags = tmpl::list<::Tags::Variables<
      db::wrap_tags_in<::Tags::dt, typename System::variables_tag::tags_list>>>;
  using argument_tags =
      tmpl::list<domain::Tags::Element<3>,
                 evolution::dg::subcell::Tags::SubcellOptions<3>>;

  template <class DtTagsList>
  static void apply(
      const gsl::not_null<Variables<DtTagsList>*> dt_variables,
      const Element<3>& element,
      const evolution::dg::subcell::SubcellOptions& subcell_options) {
    const bool bordering_dg_block = alg::any_of(
        element.neighbors(),
        [&subcell_options](const auto& direction_and_neighbor) {
          const size_t first_block_id =
              direction_and_neighbor.second.ids().begin()->block_id();
          return std::binary_search(subcell_options.only_dg_block_ids().begin(),
                                    subcell_options.only_dg_block_ids().end(),
                                    first_block_id);
        });
    const bool in_dg_only_zone = std::binary_search(
        subcell_options.only_dg_block_ids().begin(),
        subcell_options.only_dg_block_ids().end(), element.id().block_id());
    if (bordering_dg_block and not in_dg_only_zone) {
      tmpl::for_each<
          typename grmhd::ValenciaDivClean::System::variables_tag::tags_list>(
          [&dt_variables]<class Tag>(tmpl::type_<Tag> /*meta*/) {
            auto& var = get<::Tags::dt<Tag>>(*dt_variables);
            for (size_t i = 0; i < var.size(); ++i) {
              var[i] = 0.0;
            }
          });
    }
  }
};
}  // namespace grmhd::GhValenciaDivClean::subcell

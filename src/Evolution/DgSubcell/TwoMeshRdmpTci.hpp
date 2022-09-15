// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/Variables.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
/*!
 * \brief Troubled cell indicator using a relaxed discrete maximum principle,
 * comparing the solution on two grids at the same point in time.
 *
 * Checks that the subcell solution \f$\underline{u}\f$ and the DG solution
 * \f$u\f$ satisfy
 *
 * \f{align*}{ \min(u)-\delta \le \underline{u} \le \max(u)+\delta \f}
 *
 * where
 *
 * \f{align*}{ \delta = \max\left[\delta_0, \epsilon(\max(u) - \min(u))\right]
 * \f}
 *
 * where \f$\delta_0\f$ and \f$\epsilon\f$ are constants controlling the maximum
 * absolute and relative change allowed when projecting the DG solution to the
 * subcell grid. We currently specify one value of \f$\delta_0\f$ and
 * \f$\epsilon\f$ for all variables, but this could be generalized to choosing
 * the allowed variation in a variable-specific manner.
 *
 * If all checks are passed and cell is not troubled, returns an integer `0`.
 * Otherwise returns 1-based index of the tag in the input Variables that fails
 * the check. For instance, if we have
 *
 *  - `Variables<tmpl::list<DgVar1, DgVar2, DgVar3>>` for `dg_evolved_vars`
 *  - `Variables<tmpl::list<SubVar1, SubVar2, SubVar3>>` for
 *    `subcell_evolved_vars`
 *
 * as inputs and TCI flags the second pair `DgVar2` and `SubVar2` not satisfying
 * two-mesh RDMP criteria, returned value is `2` since the second pair of tags
 * failed the check.
 *
 * \note Once a single pair of tags fails to satisfy the check, checks for the
 * remaining part of the input variables are skipped. In the example above, for
 * instance if the second pair (`DgVar2`,`SubVar2`) is flagged, the third pair
 * (`DgVar3`,`SubVar3`) is ignored and not checked.
 *
 */
template <typename... DgEvolvedVarsTags, typename... SubcellEvolvedVarsTags>
int two_mesh_rdmp_tci(
    const Variables<tmpl::list<DgEvolvedVarsTags...>>& dg_evolved_vars,
    const Variables<tmpl::list<SubcellEvolvedVarsTags...>>&
        subcell_evolved_vars,
    const double rdmp_delta0, const double rdmp_epsilon) {
  static_assert(sizeof...(DgEvolvedVarsTags) ==
                sizeof...(SubcellEvolvedVarsTags));
  ASSERT(rdmp_delta0 > 0.0, "The RDMP delta0 parameter must be positive.");
  ASSERT(rdmp_epsilon > 0.0, "The RDMP epsilon parameter must be positive.");

  bool cell_is_troubled = false;
  int tci_status = 0;
  size_t tag_index = 0;

  tmpl::for_each<
      tmpl::list<tmpl::list<DgEvolvedVarsTags, SubcellEvolvedVarsTags>...>>(
      [&cell_is_troubled, &tag_index, &dg_evolved_vars, rdmp_delta0,
       rdmp_epsilon, &subcell_evolved_vars, &tci_status](auto tag_v) {
        if (cell_is_troubled) {
          return;
        }

        using tags_list = tmpl::type_from<decltype(tag_v)>;
        using tag = tmpl::front<tags_list>;
        using inactive_tag = tmpl::back<tags_list>;
        const auto& dg_var = get<tag>(dg_evolved_vars);
        const auto& subcell_var = get<inactive_tag>(subcell_evolved_vars);

        for (auto dg_it = dg_var.begin(), subcell_it = subcell_var.begin();
             dg_it != dg_var.end() and subcell_it != subcell_var.end();
             (void)++dg_it, (void)++subcell_it) {
          ASSERT(not cell_is_troubled,
                 "If a cell has already been marked as troubled during the "
                 "two mesh RDMP TCI, we should not be continuing to check "
                 "other variables.");
          using std::max;
          using std::min;

          const double max_dg = max(*dg_it);
          const double min_dg = min(*dg_it);
          const double max_subcell = max(*subcell_it);
          const double min_subcell = min(*subcell_it);
          const double delta =
              max(rdmp_delta0, rdmp_epsilon * (max_dg - min_dg));
          cell_is_troubled =
              max_subcell > max_dg + delta or min_subcell < min_dg - delta;
          if (cell_is_troubled) {
            tci_status = static_cast<int>(tag_index + 1);
            return;
          }
          ++tag_index;
        }
      });
  return tci_status;
}
}  // namespace evolution::dg::subcell

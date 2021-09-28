// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
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
 * \f{align*}{
 * \min(u)-\delta \le \underline{u} \le \max(u)+\delta
 * \f}
 *
 * where
 *
 * \f{align*}{
 * \delta = \max\left[\delta_0, \epsilon(\max(u) - \min(u))\right]
 * \f}
 *
 * where \f$\delta_0\f$ and \f$\epsilon\f$ are constants controlling the maximum
 * absolute and relative change allowed when projecting the DG solution to the
 * subcell grid. We currently specify one value of \f$\delta_0\f$ and
 * \f$\epsilon\f$ for all variables, but this could be generalized to choosing
 * the allowed variation in a variable-specific manner.
 */
template <typename... EvolvedVarsTags>
bool two_mesh_rdmp_tci(
    const Variables<tmpl::list<EvolvedVarsTags...>>& dg_evolved_vars,
    const Variables<tmpl::list<Tags::Inactive<EvolvedVarsTags>...>>&
        subcell_evolved_vars,
    const double rdmp_delta0, const double rdmp_epsilon) {
  ASSERT(rdmp_delta0 > 0.0, "The RDMP delta0 parameter must be positive.");
  ASSERT(rdmp_epsilon > 0.0, "The RDMP epsilon parameter must be positive.");
  bool cell_is_troubled = false;
  tmpl::for_each<tmpl::list<EvolvedVarsTags...>>(
      [&cell_is_troubled, &dg_evolved_vars, rdmp_delta0, rdmp_epsilon,
       &subcell_evolved_vars](auto tag_v) {
        if (cell_is_troubled) {
          return;
        }

        using tag = tmpl::type_from<decltype(tag_v)>;
        using inactive_tag = Tags::Inactive<tag>;
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
            return;
          }
        }
      });
  return cell_is_troubled;
}
}  // namespace evolution::dg::subcell

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Compute the bulk contribution to the operator represented by the
 * `OperatorTag` applied to the `VarsTag`.
 *
 * This invokable computes \f$A(u)=-\partial_i F^i(u) + S(u)\f$, where \f$F^i\f$
 * and \f$S\f$ are the fluxes and sources of the system of first-order PDEs,
 * respectively. They are defined such that \f$A(u(x))=f(x)\f$ is the full
 * system of equations, with \f$f(x)\f$ representing sources that are
 * independent of the variables \f$u\f$. In a DG setting, boundary contributions
 * can be added to \f$A(u)\f$ to build the full DG operator.
 *
 * We generally build the operator \f$A(u)\f$ to perform elliptic solver
 * iterations. This invokable can be used to build operators for both the linear
 * solver and the nonlinear solver iterations by providing the appropriate
 * `OperatorTag` and `VarsTag`. For example, the `OperatorTag` for the linear
 * solver would typically be `LinearSolver::Tags::OperatorAppliedTo`. It is the
 * elliptic equivalent to the time derivative in evolution schemes. Instead of
 * using it to evolve the variables, the iterative linear solver uses it to
 * compute a better approximation to the equation \f$A(u)=f\f$ (as detailed
 * above).
 *
 * With:
 * - `operator_tag` = `db::add_tag_prefix<OperatorTag, VarsTag>`
 * - `fluxes_tag` = `db::add_tag_prefix<::Tags::Flux, VarsTag,
 * tmpl::size_t<Dim>, Frame::Inertial>`
 * - `div_fluxes_tag` = `db::add_tag_prefix<Tags::div, fluxes_tag>`
 * - `sources_tag` = `db::add_tag_prefix<::Tags::Source, VarsTag>`
 *
 * Uses:
 * - DataBox:
 *   - `div_fluxes_tag`
 *   - `sources_tag`
 *
 * DataBox changes:
 * - Modifies:
 *   - `operator_tag`
 */
template <size_t Dim, template <typename> class OperatorTag, typename VarsTag>
struct FirstOrderOperator {
 private:
  using operator_tag = db::add_tag_prefix<OperatorTag, VarsTag>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, VarsTag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  using sources_tag = db::add_tag_prefix<::Tags::Source, VarsTag>;

 public:
  using return_tags = tmpl::list<operator_tag>;
  using argument_tags = tmpl::list<div_fluxes_tag, sources_tag>;
  static void apply(const gsl::not_null<db::item_type<operator_tag>*>
                        operator_applied_to_vars,
                    const db::const_item_type<div_fluxes_tag>& div_fluxes,
                    const db::const_item_type<sources_tag>& sources) noexcept {
    *operator_applied_to_vars = sources - div_fluxes;
  }
};

}  // namespace elliptic

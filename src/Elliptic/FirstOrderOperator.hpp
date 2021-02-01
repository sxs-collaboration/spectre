// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {

namespace first_order_operator_detail {

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields>
struct FirstOrderFluxesImpl;

template <size_t Dim, typename... PrimalFields, typename... AuxiliaryFields>
struct FirstOrderFluxesImpl<Dim, tmpl::list<PrimalFields...>,
                            tmpl::list<AuxiliaryFields...>> {
  template <typename VarsTags, typename FluxesComputer, typename... FluxesArgs>
  static constexpr void apply(
      const gsl::not_null<Variables<db::wrap_tags_in<
          ::Tags::Flux, VarsTags, tmpl::size_t<Dim>, Frame::Inertial>>*>
          fluxes,
      const Variables<VarsTags>& vars, const FluxesComputer& fluxes_computer,
      const FluxesArgs&... fluxes_args) noexcept {
    // Compute fluxes for primal fields
    fluxes_computer.apply(
        make_not_null(&get<::Tags::Flux<PrimalFields, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(*fluxes))...,
        fluxes_args..., get<AuxiliaryFields>(vars)...);
    // Compute fluxes for auxiliary fields
    fluxes_computer.apply(
        make_not_null(&get<::Tags::Flux<AuxiliaryFields, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(*fluxes))...,
        fluxes_args..., get<PrimalFields>(vars)...);
  }
};

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputer>
struct FirstOrderSourcesImpl;

template <size_t Dim, typename... PrimalFields, typename... AuxiliaryFields,
          typename SourcesComputer>
struct FirstOrderSourcesImpl<Dim, tmpl::list<PrimalFields...>,
                             tmpl::list<AuxiliaryFields...>, SourcesComputer> {
  template <typename VarsTags, typename... SourcesArgs>
  static constexpr void apply(
      const gsl::not_null<
          Variables<db::wrap_tags_in<::Tags::Source, VarsTags>>*>
          sources,
      const Variables<VarsTags>& vars,
      const Variables<db::wrap_tags_in<
          ::Tags::Flux, VarsTags, tmpl::size_t<Dim>, Frame::Inertial>>& fluxes,
      const SourcesArgs&... sources_args) noexcept {
    // Compute primal sources. They may depend on the auxiliary variables in
    // addition to the primal variables. However, we pass the fluxes instead of
    // the auxiliary variables to the source computer as an optimization so they
    // don't have to be re-computed.
    sources->initialize(vars.number_of_grid_points(), 0.);
    SourcesComputer::apply(
        make_not_null(&get<::Tags::Source<PrimalFields>>(*sources))...,
        sources_args..., get<PrimalFields>(vars)...,
        get<::Tags::Flux<PrimalFields, tmpl::size_t<Dim>, Frame::Inertial>>(
            fluxes)...);
    // Set auxiliary field sources to the auxiliary field values to begin with.
    // This is the standard choice, since the auxiliary equations define the
    // auxiliary variables. Source computers can add to these sources.
    tmpl::for_each<tmpl::list<AuxiliaryFields...>>(
        [&sources, &vars](const auto auxiliary_field_tag_v) noexcept {
          using auxiliary_field_tag =
              tmpl::type_from<decltype(auxiliary_field_tag_v)>;
          get<::Tags::Source<auxiliary_field_tag>>(*sources) =
              get<auxiliary_field_tag>(vars);
        });
    SourcesComputer::apply(
        make_not_null(&get<::Tags::Source<AuxiliaryFields>>(*sources))...,
        sources_args..., get<PrimalFields>(vars)...);
  }
};

}  // namespace first_order_operator_detail

// @{
/*!
 * \brief Compute the fluxes \f$F^i(u)\f$ for the first-order formulation of
 * elliptic systems.
 *
 * \see `elliptic::first_order_operator`
 */
template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename VarsTags, typename FluxesComputer, typename... FluxesArgs>
void first_order_fluxes(
    gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::Flux, VarsTags, tmpl::size_t<Dim>, Frame::Inertial>>*>
        fluxes,
    const Variables<VarsTags>& vars, const FluxesComputer& fluxes_computer,
    const FluxesArgs&... fluxes_args) noexcept {
  first_order_operator_detail::FirstOrderFluxesImpl<
      Dim, PrimalFields, AuxiliaryFields>::apply(std::move(fluxes), vars,
                                                 fluxes_computer,
                                                 fluxes_args...);
}

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename VarsTags, typename FluxesComputer, typename... FluxesArgs>
auto first_order_fluxes(const Variables<VarsTags>& vars,
                        const FluxesComputer& fluxes_computer,
                        const FluxesArgs&... fluxes_args) noexcept {
  Variables<db::wrap_tags_in<::Tags::Flux, VarsTags, tmpl::size_t<Dim>,
                             Frame::Inertial>>
      fluxes{vars.number_of_grid_points()};
  first_order_operator_detail::FirstOrderFluxesImpl<
      Dim, PrimalFields, AuxiliaryFields>::apply(make_not_null(&fluxes), vars,
                                                 fluxes_computer,
                                                 fluxes_args...);
  return fluxes;
}
// @}

// @{
/*!
 * \brief Compute the sources \f$S(u)\f$ for the first-order formulation of
 * elliptic systems.
 *
 * This function takes the `fluxes` as an argument in addition to the variables
 * as an optimization. The fluxes will generally be computed before the sources
 * anyway, so we pass them to the source computers to avoid having to re-compute
 * them for source-terms that have the same form as the fluxes.
 *
 * \see `elliptic::first_order_operator`
 */
template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputer, typename VarsTags, typename... SourcesArgs>
void first_order_sources(
    gsl::not_null<Variables<db::wrap_tags_in<::Tags::Source, VarsTags>>*>
        sources,
    const Variables<VarsTags>& vars,
    const Variables<db::wrap_tags_in<::Tags::Flux, VarsTags, tmpl::size_t<Dim>,
                                     Frame::Inertial>>& fluxes,
    const SourcesArgs&... sources_args) noexcept {
  first_order_operator_detail::FirstOrderSourcesImpl<
      Dim, PrimalFields, AuxiliaryFields,
      SourcesComputer>::apply(std::move(sources), vars, fluxes,
                              sources_args...);
}

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputer, typename VarsTags, typename... SourcesArgs>
auto first_order_sources(
    const Variables<VarsTags>& vars,
    const Variables<db::wrap_tags_in<::Tags::Flux, VarsTags, tmpl::size_t<Dim>,
                                     Frame::Inertial>>& fluxes,
    const SourcesArgs&... sources_args) noexcept {
  Variables<db::wrap_tags_in<::Tags::Source, VarsTags>> sources{
      vars.number_of_grid_points()};
  first_order_operator_detail::FirstOrderSourcesImpl<
      Dim, PrimalFields, AuxiliaryFields,
      SourcesComputer>::apply(make_not_null(&sources), vars, fluxes,
                              sources_args...);
  return sources;
}
// @}

/*!
 * \brief Compute the bulk contribution to the operator represented by the
 * `OperatorTags`.
 *
 * This function computes \f$A(u)=-\partial_i F^i(u) + S(u)\f$, where \f$F^i\f$
 * and \f$S\f$ are the fluxes and sources of the system of first-order PDEs,
 * respectively. They are defined such that \f$A(u(x))=f(x)\f$ is the full
 * system of equations, with \f$f(x)\f$ representing sources that are
 * independent of the variables \f$u\f$. In a DG setting, boundary contributions
 * can be added to \f$A(u)\f$ to build the full DG operator.
 */
template <typename OperatorTags, typename DivFluxesTags, typename SourcesTags>
void first_order_operator(
    const gsl::not_null<Variables<OperatorTags>*> operator_applied_to_vars,
    const Variables<DivFluxesTags>& div_fluxes,
    const Variables<SourcesTags>& sources) noexcept {
  *operator_applied_to_vars = sources - div_fluxes;
}

/*!
 * \brief Mutating DataBox invokable to compute the bulk contribution to the
 * operator represented by the `OperatorTag` applied to the `VarsTag`
 *
 * \see `elliptic::first_order_operator`
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
  static constexpr auto apply =
      &first_order_operator<typename operator_tag::tags_list,
                            typename div_fluxes_tag::tags_list,
                            typename sources_tag::tags_list>;
};

}  // namespace elliptic

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \ref protocols related to elliptic systems
namespace elliptic::protocols {

namespace FirstOrderSystem_detail {
template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename PrimalFluxes, typename AuxiliaryFluxes>
struct test_primal_and_auxiliary_fields_and_fluxes;
template <size_t Dim, typename... PrimalFields, typename... AuxiliaryFields,
          typename... PrimalFluxes, typename... AuxiliaryFluxes>
struct test_primal_and_auxiliary_fields_and_fluxes<
    Dim, tmpl::list<PrimalFields...>, tmpl::list<AuxiliaryFields...>,
    tmpl::list<PrimalFluxes...>, tmpl::list<AuxiliaryFluxes...>>
    : std::true_type {
  static_assert(sizeof...(PrimalFields) == sizeof...(AuxiliaryFields) and
                    sizeof...(PrimalFluxes) == sizeof...(PrimalFields) and
                    sizeof...(AuxiliaryFluxes) == sizeof...(PrimalFields),
                "The system must have the same number of primal and auxiliary "
                "fields and fluxes.");
  static_assert(
      ((tmpl::size<typename AuxiliaryFields::type::index_list>::value ==
        tmpl::size<typename PrimalFields::type::index_list>::value + 1) and
       ...),
      "Auxiliary fields and primal fields must correspond to each "
      "other. In particular, each auxiliary field must have one "
      "index more than its corresponding primal field.");
  static_assert(
      ((tmpl::size<typename PrimalFluxes::type::index_list>::value ==
        tmpl::size<typename PrimalFields::type::index_list>::value + 1) and
       ...) and
          (std::is_same_v<tmpl::front<typename PrimalFluxes::type::index_list>,
                          SpatialIndex<Dim, UpLo::Up, Frame::Inertial>> and
           ...),
      "Primal fluxes and primal fields must correspond to each "
      "other. In particular, each primal flux must have one "
      "index more than its corresponding primal field and an upper-spatial "
      "first index.");
  static_assert(
      ((tmpl::size<typename AuxiliaryFluxes::type::index_list>::value ==
        tmpl::size<typename AuxiliaryFields::type::index_list>::value + 1) and
       ...) and
          (std::is_same_v<
               tmpl::front<typename AuxiliaryFluxes::type::index_list>,
               SpatialIndex<Dim, UpLo::Up, Frame::Inertial>> and
           ...),
      "Auxiliary fluxes and auxiliary fields must correspond to each "
      "other. In particular, each auxiliary flux must have one "
      "index more than its corresponding auxiliary field and an upper-spatial "
      "first index.");
};
}  // namespace FirstOrderSystem_detail

/*!
 * \brief A system of elliptic equations in first-order "flux" formulation
 *
 * Classes conforming to this protocol represent a set of elliptic partial
 * differential equations in first-order "flux" formulation:
 *
 * \f{equation}
 * -\partial_i F^i_\alpha + S_\alpha = f_\alpha(x)
 * \f}
 *
 * in terms of fluxes \f$F_\alpha^i\f$, sources \f$S_\alpha\f$ and fixed-sources
 * \f$f_\alpha(x)\f$. It resembles closely formulations of hyperbolic
 * conservation laws but allows the fluxes \f$F_\alpha^i\f$ to be higher-rank
 * tensor fields. The fluxes and sources are functionals of the "primal" system
 * variables \f$u_A(x)\f$ and their corresponding "auxiliary" variables
 * \f$v_A(x)\f$. The fixed-sourced \f$f_\alpha(x)\f$ are independent of the
 * system variables. We enumerate the variables with uppercase letters such that
 * \f$v_A\f$ is the auxiliary variable corresponding to \f$u_A\f$. Greek letters
 * enumerate _all_ variables. In documentation related to particular elliptic
 * systems we generally use the canonical system-specific symbols for the fields
 * in place of these indices. See the `Poisson::FirstOrderSystem` and the
 * `Elasticity::FirstOrderSystem` for examples.
 *
 * Conforming classes must have these static member variables:
 *
 * - `size_t volume_dim`: The number of spatial dimensions.
 *
 * Conforming classes must have these type aliases:
 *
 * - `primal_fields`: A list of tags representing the primal fields. These are
 *   the fields we solve for, e.g. \f$u\f$ for a Poisson equation.
 *
 * - `auxiliary_fields`: A list of tags representing the auxiliary fields, which
 *   are typically gradients of the primal fields, e.g. \f$v_i = \partial_i u\f$
 *   for a Poisson equation. These must follow the order of the `primal_fields`.
 *   Specifically, each auxiliary field must have one rank higher than its
 *   corresponding primal field.
 *
 * - `primal_fluxes`: A list of tags representing the primal fluxes
 *   \f$F_{u_A}^i\f$. These are typically some linear combination of the
 *   auxiliary fields with raised indices, e.g. \f$v^i = g^{ij}v_j\f$ for a
 *   curved-space Poisson equation on a background metric \f$g_{ij}\f$. They
 *   must have the same rank as the `auxiliary_fields` but with an upper-spatial
 *   first index, because their divergence defines the elliptic equation.
 *
 * - `auxiliary_fluxes`: A list of tags representing the auxiliary fluxes
 *   \f$F_{v_A}^i\f$, e.g. \f$\delta^i_j u\f$ for a Poisson equation. These must
 *   have one rank higher than the `auxiliary_fields` and have an upper-spatial
 *   first index because their divergence defines the auxiliary fields.
 *
 * - `background_fields`: A list of tags representing the variable-independent
 *   background fields in the equations. Examples are a background metric,
 *   associated fixed geometry quantities such as Christoffel symbols or the
 *   Ricci scalar, or any other fixed field that determines the problem to be
 *   solved such as matter sources in the Einstein constraint equations.
 *
 * - `inv_metric_tag`: The tag that defines the background geometry, i.e. the
 *   the geometry that the elliptic equations are formulated on. This is the
 *   metric responsible for normalizing one-forms, such as face normals.
 *
 * - `fluxes_computer`: A class that defines the primal and auxiliary fluxes
 *   \f$F_\alpha^i\f$. Must have an `argument_tags` type alias and two `apply`
 *   function overloads: One that computes the primal fluxes and another that
 *   computes the auxiliary fluxes. The first `apply` function takes these
 *   arguments in this order:
 *
 *   1. The `primal_fluxes` as not-null pointer
 *   2. The `argument_tags`
 *   3. The `auxiliary_fields`
 *
 *   The second `apply` function takes these arguments in this order:
 *
 *   1. The `auxiliary_fluxes` as not-null pointer
 *   2. The `argument_tags`
 *   3. The `primal_fields`
 *
 *   The functions can assume the output buffers are already correctly sized,
 *   but no guarantee is made on the values that the buffers hold at input. The
 *   class must have an additional `volume_tags` type alias that lists the
 *   subset of `argument_tags` that will be retrieved directly from the DataBox,
 *   instead of retrieving it from the face of an element.
 *
 * - `sources_computer`: A class that defines the primal and auxiliary sources
 *   \f$S_\alpha\f$. Must have an `argument_tags` type alias and two `apply`
 *   function overloads: One that adds the primal sources and another that adds
 *   the auxiliary sources to the equations. The first `apply` function takes
 *   these arguments in this order:
 *
 *   1. The types of the `primal_fields` as not-null pointer. These are the
 *      primal equations.
 *   2. The `argument_tags`
 *   3. The `primal_fields`
 *   4. The `primal_fluxes`
 *
 *   The second `apply` function takes these arguments in this order:
 *
 *   1. The types of the `auxiliary_fields` as not-null pointer. These are the
 *      auxiliary equations.
 *   2. The `argument_tags`
 *   3. The `primal_fields`
 *
 *   The functions are expected to _add_ the sources \f$S_\alpha\f$ to the
 *   output buffers.
 *
 * - `boundary_conditions_base`: A base class representing the supported
 *   boundary conditions. Boundary conditions can be factory-created from this
 *   base class. Currently this should be a specialization of
 *   `elliptic::BoundaryConditions::BoundaryCondition`.
 */
struct FirstOrderSystem {
  template <typename ConformingType>
  struct test {
    static constexpr size_t volume_dim = ConformingType::volume_dim;

    using primal_fields = typename ConformingType::primal_fields;
    using auxiliary_fields = typename ConformingType::auxiliary_fields;
    using primal_fluxes = typename ConformingType::primal_fluxes;
    using auxiliary_fluxes = typename ConformingType::auxiliary_fluxes;
    static_assert(
        FirstOrderSystem_detail::test_primal_and_auxiliary_fields_and_fluxes<
            volume_dim, primal_fields, auxiliary_fields, primal_fluxes,
            auxiliary_fluxes>::value);

    using background_fields = typename ConformingType::background_fields;
    static_assert(tt::is_a_v<tmpl::list, background_fields>);
    using inv_metric_tag = typename ConformingType::inv_metric_tag;
    static_assert(std::is_same_v<inv_metric_tag, void> or
                  tmpl::list_contains_v<background_fields, inv_metric_tag>);

    using fluxes_computer = typename ConformingType::fluxes_computer;
    using sources_computer = typename ConformingType::sources_computer;
    using fluxes_argument_tags = typename fluxes_computer::argument_tags;
    using sources_argument_tags = typename sources_computer::argument_tags;
    static_assert(tt::is_a_v<tmpl::list, fluxes_argument_tags>);
    static_assert(tt::is_a_v<tmpl::list, sources_argument_tags>);

    using boundary_conditions_base =
        typename ConformingType::boundary_conditions_base;
    static_assert(
        std::is_base_of_v<domain::BoundaryConditions::BoundaryCondition,
                          boundary_conditions_base>);
  };
};

}  // namespace elliptic::protocols

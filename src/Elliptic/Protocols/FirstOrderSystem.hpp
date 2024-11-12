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
template <size_t Dim, typename PrimalFields, typename PrimalFluxes>
struct test_fields_and_fluxes;
template <size_t Dim, typename... PrimalFields, typename... PrimalFluxes>
struct test_fields_and_fluxes<Dim, tmpl::list<PrimalFields...>,
                              tmpl::list<PrimalFluxes...>> : std::true_type {
  static_assert(sizeof...(PrimalFields) == sizeof...(PrimalFluxes),
                "The system must have the same number of fields and fluxes.");
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
 * \f$f_\alpha(x)\f$ \cite Fischer2021voj.
 * It resembles closely formulations of hyperbolic
 * conservation laws but allows the fluxes \f$F_\alpha^i\f$ to be higher-rank
 * tensor fields. The fluxes and sources are functionals of the system
 * variables \f$u_\alpha(x)\f$ and their derivatives. The fixed-sources
 * \f$f_\alpha(x)\f$ are independent of the system variables. See the
 * `Poisson::FirstOrderSystem` and the `Elasticity::FirstOrderSystem` for
 * examples.
 *
 * Note that this formulation has been simplified since \cite Fischer2021voj :
 * We assume that the fluxes are linear in the fields and their derivatives and
 * removed the notion of "auxiliary variables" from the formulation altogether.
 * In the language of \cite Fischer2021voj we always just choose the partial
 * derivatives of the fields as auxiliary variables.
 *
 * Conforming classes must have these static member variables:
 *
 * - `size_t volume_dim`: The number of spatial dimensions.
 *
 * Conforming classes must have these type aliases:
 *
 * - `primal_fields`: A list of tags representing the primal fields. These are
 *   the fields we solve for, e.g. \f$u\f$ for a Poisson equation.
 *   (we may rename this to just "fields" since we removed the notion of
 *   "auxiliary fields")
 *
 * - `primal_fluxes`: A list of tags representing the primal fluxes
 *   \f$F_\alpha^i\f$. These are typically some linear combination of the
 *   derivatives of the system fields with raised indices, e.g. \f$v^i = g^{ij}
 *   \partial_j u\f$ for a curved-space Poisson equation on a background metric
 *   \f$g_{ij}\f$. They must have an upper-spatial first index, because their
 *   divergence defines the elliptic equation.
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
 * - `fluxes_computer`: A class that defines the fluxes \f$F_\alpha^i\f$. Must
 *   have an `argument_tags` type alias and an `apply` function that takes these
 *   arguments in this order:
 *
 *   1. The `primal_fluxes` as not-null pointer
 *   2. The `argument_tags`
 *   3. If `is_discontinuous` is `true` (see below):
 *      const ElementId<Dim>& element_id
 *   4. The `primal_fields`
 *   5. The partial derivatives of the `primal_fields`
 *
 *   The function can assume the output buffers are already correctly sized,
 *   but no guarantee is made on the values that the buffers hold at input.
 *
 *   The `fluxes_computer` must also have an `apply` function overload that is
 *   evaluated on faces of DG elements. It computes the same fluxes
 *   \f$F_\alpha^i\f$, but with the field derivatives replaced by the the face
 *   normal times the fields, and with the non-principal (non-derivative) terms
 *   set to zero. Having this separate function is an optimization to take
 *   advantage of the face normal remaining constant throughout the solve, so it
 *   can be "baked in" to the flux. The function takes these arguments in this
 *   order:
 *
 *   1. The `primal_fluxes` as not-null pointer
 *   2. The `argument_tags`
 *   3. If `is_discontinuous` is `true` (see below):
 *      const ElementId<Dim>& element_id
 *   4. The `const tnsr::i<DataVector, Dim>& face_normal` ($n_i$)
 *   5. The `const tnsr::I<DataVector, Dim>& face_normal_vector` ($n^i$)
 *   6. The `primal_fields`
 *
 *   The `fluxes_computer` class must also have the following additional
 *   type aliases and static member variables:
 *
 *   - `volume_tags`: the subset of `argument_tags` that will be retrieved
 *     directly from the DataBox, instead of retrieving it from the face of an
 *     element, when fluxes are applied on a face.
 *   - `const_global_cache_tags`: the subset of `argument_tags` that can be
 *     retrieved from _any_ element's DataBox, because they are stored in the
 *     global cache.
 *   - `bool is_trivial`: a boolean indicating whether the fluxes are simply
 *     the spatial metric, as is the case for the Poisson equation. Some
 *     computations can be skipped in this case.
 *   - `bool is_discontinuous`: a boolean indicating whether the fluxes are
 *     potentially discontinuous across element boundaries. This is `true` for
 *     systems where the equations on both sides of the boundary are different,
 *     e.g. elasticity with different materials on either side of the boundary.
 *     An additional `element_id` argument is passed to the `apply` functions in
 *     this case, which identifies on which side of an element boundary the
 *     fluxes are being evaluated.
 *
 * - `sources_computer`: A class that defines the sources \f$S_\alpha\f$. Must
 *   have an `argument_tags` type alias and an `apply` function that adds the
 *   sources to the equations. It takes these arguments in this order:
 *
 *   1. The types of the `primal_fields` as not-null pointer. These are the
 *      primal equations.
 *   2. The `argument_tags`
 *   3. The `primal_fields`
 *   4. The `primal_fluxes`
 *
 *   The function is expected to _add_ the sources \f$S_\alpha\f$ to the
 *   output buffers. It must also have the following alias:
 *
 * - `const_global_cache_tags`: the subset of `argument_tags` that can be
 *     retrieved from _any_ element's DataBox, because they are stored in the
 *     global cache.
 *
 *   The `sources_computer` may also be `void`, in which case \f$S_\alpha=0\f$.
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
    using primal_fluxes = typename ConformingType::primal_fluxes;
    static_assert(FirstOrderSystem_detail::test_fields_and_fluxes<
                  volume_dim, primal_fields, primal_fluxes>::value);

    using background_fields = typename ConformingType::background_fields;
    static_assert(tt::is_a_v<tmpl::list, background_fields>);
    using inv_metric_tag = typename ConformingType::inv_metric_tag;
    static_assert(std::is_same_v<inv_metric_tag, void> or
                  tmpl::list_contains_v<background_fields, inv_metric_tag>);

    using fluxes_computer = typename ConformingType::fluxes_computer;
    using sources_computer = typename ConformingType::sources_computer;
    static_assert(
        tt::is_a_v<tmpl::list, typename fluxes_computer::argument_tags>);

    using boundary_conditions_base =
        typename ConformingType::boundary_conditions_base;
    static_assert(
        std::is_base_of_v<domain::BoundaryConditions::BoundaryCondition,
                          boundary_conditions_base>);
  };
};

}  // namespace elliptic::protocols

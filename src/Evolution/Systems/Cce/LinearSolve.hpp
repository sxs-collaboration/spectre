// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"

/// \cond
class ComplexDataVector;
class Matrix;
/// \endcond

namespace Cce {

/*!
 * \brief Provides access to a lazily cached integration matrix for the \f$Q\f$
 * and \f$W\f$ equations in CCE hypersurface evaluation.
 *
 * \details The provided matrix acts on the integrand collocation points and
 * solves the equation,
 *
 * \f[
 * (1 - y) \partial_y f + 2 f = g,
 * \f]
 *
 * for \f$f\f$ given integrand \f$g\f$.
 */
const Matrix& precomputed_cce_q_integrator(
    size_t number_of_radial_grid_points) noexcept;

/*!
 * \brief A utility function for evaluating the \f$Q\f$ and \f$W\f$ hypersurface
 * integrals during CCE evolution.
 *
 * \details Computes and returns by `not_null` pointer the solution to the
 * equation
 *
 * \f[
 * (1 - y) \partial_y f + 2 f = A + (1 - y) B,
 * \f]
 *
 * where \f$A\f$ is provided as `pole_of_integrand` and \f$B\f$ is provided as
 * `regular_integrand`. The value `one_minus_y` is required for determining the
 * integrand and `l_max` is required to determine the shape of the spin-weighted
 * spherical harmonic mesh.
 */
void radial_integrate_cce_pole_equations(
    gsl::not_null<ComplexDataVector*> integral_result,
    const ComplexDataVector& pole_of_integrand,
    const ComplexDataVector& regular_integrand,
    const ComplexDataVector& boundary, const ComplexDataVector& one_minus_y,
    size_t l_max, size_t number_of_radial_points) noexcept;

namespace detail {
// needed because the standard transpose utility cannot create an arbitrary
// ordering of blocks of data. This returns by pointer the configuration useful
// for the linear solve step for H integration
void transpose_to_reals_then_imags_radial_stripes(
    gsl::not_null<DataVector*> result, const ComplexDataVector& input,
    size_t number_of_radial_points, size_t number_of_angular_points) noexcept;
}  // namespace detail

/// @{
/*!
 * \brief Computational structs for evaluating the hypersurface integrals during
 * CCE evolution. These are compatible with use in `db::mutate_apply`.
 *
 * \details
 * The integral evaluated and the corresponding inputs required depend on the
 * CCE quantity being computed. In any of these, the only mutated tag is `Tag`,
 * where the result of the integration is placed. The supported `Tag`s act in
 * the following ways:
 * - If the `Tag` is `Tags::BondiBeta` or `Tags::BondiU`, the integral to be
 * evaluated is simply \f[ \partial_y f = A, \f] where \f$A\f$ is retrieved with
 * `Tags::Integrand<Tag>`.
 * - If the `Tag` is `Tags::BondiQ` or `Tags::BondiW`, the integral to be
 * evaluated is \f[ (1 - y) \partial_y f + 2 f = A + (1 - y) B, \f] where
 * \f$A\f$ is retrieved with `Tags::PoleOfIntegrand<Tag>` and \f$B\f$ is
 * retrieved with `Tags::RegularIntegrand<Tag>`.
 * - If `Tag` is `Tags::BondiH`, the integral to be evaluated is:
 *
 * \f[
 * (1 - y) \partial_y f + L f + L^\prime \bar{f} = A + (1 - y) B,
 * \f]
 *
 * for \f$f\f$, where \f$A\f$ is retrieved with `Tags::PoleOfIntegrand<Tag>`,
 * \f$B\f$ is retrieved with `Tags::RegularIntegrand<Tag>`, \f$L\f$ is retrieved
 * with `Tags::LinearFactor<Tag>`, and \f$L^\prime\f$ is retrieved with
 * `Tags::LinearFactorForConjugate<Tag>`. The presence of \f$L\f$ and
 * \f$L^\prime\f$ ensure that the only current method we have for evaluating the
 * \f$H\f$ hypersurface equation is a direct linear solve, rather than the
 * spectral matrix multiplications which are available for the other integrals.
 *
 * In each case, the boundary value at the world tube for the integration is
 * retrieved from `BoundaryPrefix<Tag>`.
 *
 * Additional type aliases `boundary_tags` and `integrand_tags` are provided for
 * template processing of the required input tags necessary for these functions.
 * These type aliases are `tmpl::list`s with the subsets of `argument_tags` from
 * specific other parts of the CCE computation. Because they play different
 * roles, and have different extents, it is better for tag management to give
 * separated lists for the dependencies.
 */
template <template <typename> class BoundaryPrefix, typename Tag>
struct RadialIntegrateBondi {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tag>>;
  using integrand_tags = tmpl::list<Tags::Integrand<Tag>>;

  using return_tags = tmpl::list<Tag>;
  using argument_tags =
      tmpl::append<integrand_tags, boundary_tags,
                   tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>>;
  static void apply(
      gsl::not_null<
          Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>*>
          integral_result,
      const Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>&
          integrand,
      const Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>&
          boundary,
      size_t l_max, size_t number_of_radial_points) noexcept;
};

template <template <typename> class BoundaryPrefix>
struct RadialIntegrateBondi<BoundaryPrefix, Tags::BondiQ> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiQ>>;
  using integrand_tags = tmpl::list<Tags::PoleOfIntegrand<Tags::BondiQ>,
                                    Tags::RegularIntegrand<Tags::BondiQ>>;
  using integration_independent_tags = tmpl::list<Tags::OneMinusY>;

  using return_tags = tmpl::list<Tags::BondiQ>;
  using argument_tags =
      tmpl::append<integrand_tags, boundary_tags, integration_independent_tags,
                   tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          integral_result,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& pole_of_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& regular_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      size_t l_max, size_t number_of_radial_points) noexcept;
};

template <template <typename> class BoundaryPrefix>
struct RadialIntegrateBondi<BoundaryPrefix, Tags::BondiW> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiW>>;
  using integrand_tags = tmpl::list<Tags::PoleOfIntegrand<Tags::BondiW>,
                                    Tags::RegularIntegrand<Tags::BondiW>>;
  using integration_independent_tags = tmpl::list<Tags::OneMinusY>;

  using return_tags = tmpl::list<Tags::BondiW>;
  using argument_tags =
      tmpl::append<integrand_tags, boundary_tags, integration_independent_tags,
                   tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          integral_result,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& pole_of_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& regular_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      size_t l_max, size_t number_of_radial_points) noexcept;
};

template <template <typename> class BoundaryPrefix>
struct RadialIntegrateBondi<BoundaryPrefix, Tags::BondiH> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiH>>;
  using integrand_tags =
      tmpl::list<Tags::PoleOfIntegrand<Tags::BondiH>,
                 Tags::RegularIntegrand<Tags::BondiH>,
                 Tags::LinearFactor<Tags::BondiH>,
                 Tags::LinearFactorForConjugate<Tags::BondiH>>;
  using integration_independent_tags = tmpl::list<Tags::OneMinusY>;

  using return_tags = tmpl::list<Tags::BondiH>;
  using argument_tags =
      tmpl::append<integrand_tags, boundary_tags, integration_independent_tags,
                   tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>>;
  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          integral_result,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& pole_of_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& regular_integrand,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& linear_factor,
      const Scalar<SpinWeighted<ComplexDataVector, 4>>&
          linear_factor_of_conjugate,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
      size_t l_max, size_t number_of_radial_points) noexcept;
};
/// @}
}  // namespace Cce

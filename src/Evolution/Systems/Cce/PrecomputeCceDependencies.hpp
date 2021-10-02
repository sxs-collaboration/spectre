// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/VectorAlgebra.hpp"

namespace Cce {

namespace detail {
// A convenience function for computing the spin-weighted derivatives of \f$R\f$
// divided by \f$R\f$, which appears often in Jacobians to transform between
// Bondi coordinates and the numerical coordinates used in CCE.
template <typename DerivKind>
void angular_derivative_of_r_divided_by_r_impl(
    gsl::not_null<
        SpinWeighted<ComplexDataVector,
                     Spectral::Swsh::Tags::derivative_spin_weight<DerivKind>>*>
        d_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r, size_t l_max,
    size_t number_of_radial_points);

}  // namespace detail

/*!
 * \brief A set of procedures for computing the set of inputs to the CCE
 * integrand computations that can be computed before any of the intermediate
 * integrands are evaluated.
 *
 * \details The template specializations of this template are
 * compatible with acting as a the mutator in a \ref DataBoxGroup
 * `db::mutate_apply` operation. For flexibility in defining the \ref
 * DataBoxGroup structure, the tags for `Tensor`s used in these functions are
 * also organized into type lists:
 * -  type alias `integration_independent_tags`: with a subset of
 * `Cce::pre_computation_tags`, used for both input and output.
 * -  type alias `boundary_values`: with a subset of
 *   `Cce::pre_computation_boundary_tags`, used only for input.
 * - type alias `pre_swsh_derivatives` containing hypersurface quantities. For
 * this struct, it will only ever contain `Cce::Tags::BondiJ`, and is used as
 * input.
 *
 * The `BoundaryPrefix` tag allows easy switching between the
 * regularity-preserving version and standard CCE
 *
 */
template <template <typename> class BoundaryPrefix, typename Tag>
struct PrecomputeCceDependencies;

/// Computes \f$1 - y\f$ for the CCE system.
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::OneMinusY> {
  using boundary_tags = tmpl::list<>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::OneMinusY>;
  using argument_tags = tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          one_minus_y,
      const size_t l_max, const size_t number_of_radial_points) {
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    const DataVector one_minus_y_collocation =
        1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);
    // iterate through the angular 'chunks' and set them to their 1-y value
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      ComplexDataVector angular_view{
          get(*one_minus_y).data().data() + number_of_angular_points * i,
          number_of_angular_points};
      angular_view = one_minus_y_collocation[i];
    }
  }
};

/// Computes a volume version of Bondi radius of the worldtube \f$R\f$ from its
/// boundary value (by repeating it over the radial dimension)
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::BondiR> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiR>>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::BondiR>;
  using argument_tags =
      tmpl::append<boundary_tags, tmpl::list<Tags::NumberOfRadialPoints>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const size_t number_of_radial_points) {
    fill_with_n_copies(make_not_null(&get(*r).data()), get(boundary_r).data(),
                       number_of_radial_points);
  }
};

/// Computes \f$\partial_u R / R\f$ from its boundary value (by repeating it
/// over the radial dimension).
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::DuRDividedByR> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::DuRDividedByR>>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::DuRDividedByR>;
  using argument_tags =
      tmpl::append<boundary_tags, tmpl::list<Tags::NumberOfRadialPoints>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          du_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>&
          boundary_du_r_divided_by_r,
      const size_t number_of_radial_points) {
    fill_with_n_copies(make_not_null(&get(*du_r_divided_by_r).data()),
                       get(boundary_du_r_divided_by_r).data(),
                       number_of_radial_points);
  }
};

/// Computes \f$\eth R / R\f$ by differentiating and repeating the boundary
/// value of \f$R\f$.
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::EthRDividedByR> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiR>>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::EthRDividedByR>;
  using argument_tags =
      tmpl::append<boundary_tags,
                   tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
          eth_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const size_t l_max, const size_t number_of_radial_points) {
    detail::angular_derivative_of_r_divided_by_r_impl<
        Spectral::Swsh::Tags::Eth>(make_not_null(&get(*eth_r_divided_by_r)),
                                   get(boundary_r), l_max,
                                   number_of_radial_points);
  }
};

/// Computes \f$\eth \eth R / R\f$ by differentiating and repeating the boundary
/// value of \f$R\f$.
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::EthEthRDividedByR> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiR>>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::EthEthRDividedByR>;
  using argument_tags =
      tmpl::append<boundary_tags,
                   tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          eth_eth_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const size_t l_max, const size_t number_of_radial_points) {
    detail::angular_derivative_of_r_divided_by_r_impl<
        Spectral::Swsh::Tags::EthEth>(
        make_not_null(&get(*eth_eth_r_divided_by_r)), get(boundary_r), l_max,
        number_of_radial_points);
  }
};

/// Computes \f$\eth \bar{\eth} R / R\f$ by differentiating and repeating the
/// boundary value of \f$R\f$.
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::EthEthbarRDividedByR> {
  using boundary_tags = tmpl::list<BoundaryPrefix<Tags::BondiR>>;
  using pre_swsh_derivative_tags = tmpl::list<>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::EthEthbarRDividedByR>;
  using argument_tags =
      tmpl::append<boundary_tags,
                   tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          eth_ethbar_r_divided_by_r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_r,
      const size_t l_max, const size_t number_of_radial_points) {
    detail::angular_derivative_of_r_divided_by_r_impl<
        Spectral::Swsh::Tags::EthEthbar>(
        make_not_null(&get(*eth_ethbar_r_divided_by_r)), get(boundary_r), l_max,
        number_of_radial_points);
  }
};

/// Computes \f$K = \sqrt{1 + J \bar{J}}\f$.
template <template <typename> class BoundaryPrefix>
struct PrecomputeCceDependencies<BoundaryPrefix, Tags::BondiK> {
  using boundary_tags = tmpl::list<>;
  using pre_swsh_derivative_tags = tmpl::list<Tags::BondiJ>;
  using integration_independent_tags = tmpl::list<>;

  using return_tags = tmpl::list<Tags::BondiK>;
  using argument_tags = tmpl::push_front<pre_swsh_derivative_tags, Tags::LMax>;

  static void apply(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> k,
      const size_t /*l_max*/,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& j) {
    get(*k).data() = sqrt(1.0 + get(j).data() * conj(get(j)).data());
  }
};

/*!
 * \brief Convenience routine for computing all of the CCE inputs to integrand
 * computation that do not depend on intermediate integrand results. It should
 * be executed before moving through the hierarchy of integrands.
 *
 * \details Provided a \ref DataBoxGroup with the appropriate tags (including
 * `Cce::pre_computation_boundary_tags`, `Cce::pre_computation_tags`,
 * `Cce::Tags::BondiJ` and `Tags::LMax`), this function will
 * apply all of the necessary mutations to update the
 * `Cce::pre_computation_tags` to their correct values for the current values
 * for the remaining (input) tags.
 *
 * The `BoundaryPrefix` template template parameter is to be passed a prefix
 * tag associated with the boundary value prefix used in the computation (e.g.
 * `Cce::Tags::BoundaryValue`), and allows easy switching between the
 * regularity-preserving version and standard CCE.
 */
template <template <typename> class BoundaryPrefix, typename DataBoxType>
void mutate_all_precompute_cce_dependencies(
    const gsl::not_null<DataBoxType*> box) {
  tmpl::for_each<pre_computation_tags>([&box](auto x) {
    using integration_independent_tag = typename decltype(x)::type;
    using mutation =
        PrecomputeCceDependencies<BoundaryPrefix, integration_independent_tag>;
    db::mutate_apply<mutation>(box);
  });
}
}  // namespace Cce

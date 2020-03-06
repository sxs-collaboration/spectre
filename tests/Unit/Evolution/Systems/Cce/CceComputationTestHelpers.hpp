// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/VectorAlgebra.hpp"
#include "tests/Unit/TestingFramework.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {
namespace TestHelpers {

// For representing a primitive series of powers in inverse r for diagnostic
// computations
template <typename Tag>
struct RadialPolyCoefficientsFor : db::SimpleTag, db::PrefixTag {
  using type = Scalar<ComplexModalVector>;
  using tag = Tag;
};

// For representing the angular function in a separable quantity.
template <typename Tag>
struct AngularCollocationsFor : db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

// shortcut method for evaluating a frequently used radial quantity in CCE
void volume_one_minus_y(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> one_minus_y,
    size_t l_max) noexcept;

// explicit power method avoids behavior of Blaze to occasionally FPE on
// powers of complex that operate fine when repeated multiplication is used
// instead.
ComplexDataVector power(const ComplexDataVector& value,
                        size_t exponent) noexcept;

// A utility for copying a set of tags from one DataBox to another. Useful in
// the CCE tests, where often an expected input needs to be copied to the box
// which is to be tested.
template <typename... Tags>
struct CopyDataBoxTags {
  template <typename FromDataBox, typename ToDataBox>
  static void apply(const gsl::not_null<ToDataBox*> to_data_box,
                    const FromDataBox& from_data_box) noexcept {
    db::mutate<Tags...>(
        to_data_box,
        [](const gsl::not_null<db::item_type<Tags>*>... to_value,
           const typename Tags::type&... from_value) noexcept {
          const auto assign = [](auto to, const auto& from) noexcept {
            *to = from;
            return 0;
          };
          expand_pack(assign(to_value, from_value)...);
        },
        db::get<Tags>(from_data_box)...);
  }
};

// Given the angular and radial dependence separately, assembles the volume data
// by computing the tensor product.
void generate_volume_data_from_separated_values(
    gsl::not_null<ComplexDataVector*> volume_data,
    gsl::not_null<ComplexDataVector*> one_divided_by_r,
    const ComplexDataVector& angular_collocation,
    const ComplexModalVector& radial_coefficients, size_t l_max,
    size_t number_of_radial_grid_points) noexcept;

// A utility for separately verifying the values in several computation
// routines in the CCE quantity derivations. These are an independent
// computation of similar quantities needed in the CCE systems under the
// assumption that the values are separable, i.e. can be written as F(theta,
// phi, r) = f(theta, phi) * g(r), and performing semi-analytic manipulations
// using simplifications from separability and an explicit decomposition of g in
// inverse powers of r.
template <typename Tag>
struct CalculateSeparatedTag;

template <typename Tag>
struct CalculateSeparatedTag<Tags::Dy<Tag>> {
  // d x/dy = d x /dr * (2 R / (1-y)^2) = dx/dr * r^2 / (2 R)
  // So, for x = r^(-n),
  // d (r^(-n))/dy = -n * r^(-n - 1) * r^2 / (2 R) = -n / (2 R) r^(-n + 1)
  // except when n=0, then the result is zero. So, the entire process actually
  // moves the polynomials down in order
  template <typename AngularTagList, typename RadialCoefficientTagList>
  void operator()(
      const gsl::not_null<Variables<AngularTagList>*> angular_collocation,
      const gsl::not_null<Variables<RadialCoefficientTagList>*>
          radial_coefficients,
      const ComplexDataVector& /*one_divided_by_r*/,
      const ComplexDataVector& boundary_r, const size_t /*l_max*/) noexcept {
    get(get<AngularCollocationsFor<Tags::Dy<Tag>>>(*angular_collocation))
        .data() =
        get(get<AngularCollocationsFor<Tag>>(*angular_collocation)).data() /
        (2.0 * boundary_r);

    ComplexModalVector& dy_radial_values = get(
        get<RadialPolyCoefficientsFor<Tags::Dy<Tag>>>(*radial_coefficients));
    const ComplexModalVector& radial_values =
        get(get<RadialPolyCoefficientsFor<Tag>>(*radial_coefficients));

    for (size_t radial_power = 1;
         radial_power < radial_coefficients->number_of_grid_points();
         ++radial_power) {
      dy_radial_values[radial_power - 1] =
          -static_cast<double>(radial_power) * radial_values[radial_power];
    }
    dy_radial_values[radial_coefficients->number_of_grid_points() - 1] = 0.0;
  }
};

template <typename Tag, typename DerivKind>
struct CalculateSeparatedTag<Spectral::Swsh::Tags::Derivative<Tag, DerivKind>> {
  template <typename AngularTagList, typename RadialCoefficientTagList>
  void operator()(
      const gsl::not_null<Variables<AngularTagList>*> angular_collocation,
      const gsl::not_null<Variables<RadialCoefficientTagList>*>
          radial_coefficients,
      const ComplexDataVector& /*one_divided_by_r*/,
      const ComplexDataVector& /*boundary_r*/, const size_t l_max) noexcept {
    get(get<AngularCollocationsFor<
            Spectral::Swsh::Tags::Derivative<Tag, DerivKind>>>(
        *angular_collocation)) =
        Spectral::Swsh::angular_derivative<DerivKind>(
            l_max, 1,
            get(get<AngularCollocationsFor<Tag>>(*angular_collocation)));
    // The spin-weighted derivatives are evaluated at constant r, so the radial
    // coefficients are unaltered.
    get(get<RadialPolyCoefficientsFor<
            Spectral::Swsh::Tags::Derivative<Tag, DerivKind>>>(
        *radial_coefficients)) =
        get(get<RadialPolyCoefficientsFor<Tag>>(*radial_coefficients));
  }
};

template <typename LhsTag, typename RhsTag>
struct CalculateSeparatedTag<::Tags::Multiplies<LhsTag, RhsTag>> {
  template <typename AngularTagList, typename RadialCoefficientTagList>
  void operator()(
      const gsl::not_null<Variables<AngularTagList>*> angular_collocation,
      const gsl::not_null<Variables<RadialCoefficientTagList>*>
          radial_coefficients,
      const ComplexDataVector& /*one_divided_by_r*/,
      const ComplexDataVector& /*boundary_r*/,
      const size_t /*l_max*/) noexcept {
    get(get<AngularCollocationsFor<::Tags::Multiplies<LhsTag, RhsTag>>>(
        *angular_collocation)) =
        get(get<AngularCollocationsFor<LhsTag>>(*angular_collocation)) *
        get(get<AngularCollocationsFor<RhsTag>>(*angular_collocation));

    ComplexModalVector& multiplied_radial_values =
        get(get<RadialPolyCoefficientsFor<::Tags::Multiplies<LhsTag, RhsTag>>>(
            *radial_coefficients));
    const ComplexModalVector& lhs_radial_values =
        get(get<RadialPolyCoefficientsFor<LhsTag>>(*radial_coefficients));
    const ComplexModalVector& rhs_radial_values =
        get(get<RadialPolyCoefficientsFor<RhsTag>>(*radial_coefficients));
    multiplied_radial_values = 0.0;
    for (size_t lhs_radial_power = 0;
         lhs_radial_power < radial_coefficients->number_of_grid_points();
         ++lhs_radial_power) {
      for (size_t rhs_radial_power = 0;
           rhs_radial_power <
           radial_coefficients->number_of_grid_points() - lhs_radial_power;
           ++rhs_radial_power) {
        multiplied_radial_values[lhs_radial_power + rhs_radial_power] +=
            lhs_radial_values[lhs_radial_power] *
            rhs_radial_values[rhs_radial_power];
      }
    }
  }
};

template <>
struct CalculateSeparatedTag<Tags::BondiJbar> {
  template <typename AngularTagList, typename RadialCoefficientTagList>
  void operator()(
      const gsl::not_null<Variables<AngularTagList>*> angular_collocation,
      const gsl::not_null<Variables<RadialCoefficientTagList>*>
          radial_coefficients,
      const ComplexDataVector& /*one_divided_by_r*/,
      const ComplexDataVector& /*boundary_r*/,
      const size_t /*l_max*/) noexcept {
    get(get<AngularCollocationsFor<Tags::BondiJbar>>(*angular_collocation)) =
        conj(get(
            get<AngularCollocationsFor<Tags::BondiJ>>(*angular_collocation)));
    get(get<RadialPolyCoefficientsFor<Tags::BondiJbar>>(*radial_coefficients)) =
        conj(get(get<RadialPolyCoefficientsFor<Tags::BondiJ>>(
            *radial_coefficients)));
  }
};

template <>
struct CalculateSeparatedTag<Tags::BondiUbar> {
  template <typename AngularTagList, typename RadialCoefficientTagList>
  void operator()(
      const gsl::not_null<Variables<AngularTagList>*> angular_collocation,
      const gsl::not_null<Variables<RadialCoefficientTagList>*>
          radial_coefficients,
      const ComplexDataVector& /*one_divided_by_r*/,
      const ComplexDataVector& /*boundary_r*/,
      const size_t /*l_max*/) noexcept {
    get(get<AngularCollocationsFor<Tags::BondiUbar>>(*angular_collocation)) =
        conj(get(
            get<AngularCollocationsFor<Tags::BondiU>>(*angular_collocation)));
    get(get<RadialPolyCoefficientsFor<Tags::BondiUbar>>(*radial_coefficients)) =
        conj(get(get<RadialPolyCoefficientsFor<Tags::BondiU>>(
            *radial_coefficients)));
  }
};

template <>
struct CalculateSeparatedTag<Tags::BondiQbar> {
  template <typename AngularTagList, typename RadialCoefficientTagList>
  void operator()(
      const gsl::not_null<Variables<AngularTagList>*> angular_collocation,
      const gsl::not_null<Variables<RadialCoefficientTagList>*>
          radial_coefficients,
      const ComplexDataVector& /*one_divided_by_r*/,
      const ComplexDataVector& /*boundary_r*/,
      const size_t /*l_max*/) noexcept {
    get(get<AngularCollocationsFor<Tags::BondiQbar>>(*angular_collocation)) =
        conj(get(
            get<AngularCollocationsFor<Tags::BondiQ>>(*angular_collocation)));
    get(get<RadialPolyCoefficientsFor<Tags::BondiQbar>>(*radial_coefficients)) =
        conj(get(get<RadialPolyCoefficientsFor<Tags::BondiQ>>(
            *radial_coefficients)));
  }
};

// A generation function used in the tests of the CCE computations for the
// inputs to the integrand computation (`Test_PreSwshDerivatives.cpp`,
// and `Test_ComputeSwshDerivatives.cpp`). This function first generates the set
// of inputs specified by `InputTagList`, then emulates the cascaded
// computations of those utilities, using the tag lists as though the integrands
// that we wanted to compute were of the tags in `TargetTagList`. Instead of
// computing the values using the utilities in the main code base, though, the
// generation creates the expected values using the separable computations
// above. While this is also a fairly complicated computation (though simpler
// due to the assumption of separability), it is a completely independent method
// so can be regarded as a robust way of verifying the utilities in the main
// code base.
template <typename InputTagList, typename TargetTagList,
          typename PreSwshDerivativesTagList, typename SwshDerivativesTagList,
          typename AngularCollocationTagList,
          typename PreSwshDerivativesRadialModeTagList, typename Generator>
void generate_separable_expected(
    const gsl::not_null<Variables<PreSwshDerivativesTagList>*>
        pre_swsh_derivatives,
    const gsl::not_null<Variables<SwshDerivativesTagList>*> swsh_derivatives,
    const gsl::not_null<Variables<AngularCollocationTagList>*>
        angular_collocations,
    const gsl::not_null<Variables<PreSwshDerivativesRadialModeTagList>*>
        radial_modes,
    const gsl::not_null<Generator*> generator,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r, const size_t l_max,
    const size_t number_of_radial_points) noexcept {
  const ComplexDataVector y = outer_product(
      ComplexDataVector{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 1.0},
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
          number_of_radial_points));

  // generate the separable variables
  UniformCustomDistribution<double> dist(0.1, 1.0);

  ComplexDataVector one_divided_by_r =
      (1.0 - y) / (2.0 * create_vector_of_n_copies(boundary_r.data(),
                                                   number_of_radial_points));

  // the generation step for the 'input' tags
  tmpl::for_each<InputTagList>([
    &angular_collocations, &radial_modes, &pre_swsh_derivatives, &dist,
    &generator, &l_max, &number_of_radial_points, &one_divided_by_r
  ](auto tag_v) noexcept {
    using tag = typename decltype(tag_v)::type;
    using radial_polynomial_tag = RadialPolyCoefficientsFor<tag>;
    using angular_collocation_tag = AngularCollocationsFor<tag>;
    // generate the angular part randomly
    get(get<angular_collocation_tag>(*angular_collocations)).data() =
        make_with_random_values<ComplexDataVector>(
            generator, make_not_null(&dist),
            Spectral::Swsh::number_of_swsh_collocation_points(l_max));
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(
            &get(get<angular_collocation_tag>(*angular_collocations))),
        l_max, 2);
    // generate the radial part
    auto& radial_polynomial = get(get<radial_polynomial_tag>(*radial_modes));
    fill_with_random_values(make_not_null(&radial_polynomial), generator,
                            make_not_null(&dist));
    // filtering to make sure the results are reasonable,
    // filter function: exp(-10.0 * (i / N)**2)
    for (size_t i = 0; i < radial_polynomial.size(); ++i) {
      radial_polynomial[i] *= exp(
          -10.0 * square(static_cast<double>(i) /
                         static_cast<double>(radial_polynomial.size() - 1)));
      // aggressive Heaviside needed to get acceptable test precision at the low
      // resolutions needed to keep tests fast.
      if (i > 3) {
        radial_polynomial[i] = 0.0;
      }
    }

    generate_volume_data_from_separated_values(
        make_not_null(&get(get<tag>(*pre_swsh_derivatives)).data()),
        make_not_null(&one_divided_by_r),
        get(get<angular_collocation_tag>(*angular_collocations)).data(),
        radial_polynomial, l_max, number_of_radial_points);
  });

  // Compute the expected versions using the separable mathematics from above
  // utilities
  const auto calculate_separable_for_pre_swsh_derivative = [
    &angular_collocations, &radial_modes, &pre_swsh_derivatives, &l_max,
    &number_of_radial_points, &one_divided_by_r, &boundary_r
  ](auto pre_swsh_derivative_tag_v) noexcept {
    using pre_swsh_derivative_tag =
        typename decltype(pre_swsh_derivative_tag_v)::type;
    CalculateSeparatedTag<pre_swsh_derivative_tag>{}(
        angular_collocations, radial_modes, one_divided_by_r, boundary_r.data(),
        l_max);
    generate_volume_data_from_separated_values(
        make_not_null(
            &get(get<pre_swsh_derivative_tag>(*pre_swsh_derivatives)).data()),
        make_not_null(&one_divided_by_r),
        get(get<AngularCollocationsFor<pre_swsh_derivative_tag>>(
                *angular_collocations))
            .data(),
        get(get<RadialPolyCoefficientsFor<pre_swsh_derivative_tag>>(
            *radial_modes)),
        l_max, number_of_radial_points);
  };

  const auto calculate_separable_for_swsh_derivative = [
    &angular_collocations, &radial_modes, &swsh_derivatives, &l_max,
    &number_of_radial_points, &one_divided_by_r, &boundary_r
  ](auto swsh_derivative_tag_v) noexcept {
    using swsh_derivative_tag = typename decltype(swsh_derivative_tag_v)::type;
    CalculateSeparatedTag<swsh_derivative_tag>{}(angular_collocations,
                                                 radial_modes, one_divided_by_r,
                                                 boundary_r.data(), l_max);
    generate_volume_data_from_separated_values(
        make_not_null(&get(get<swsh_derivative_tag>(*swsh_derivatives)).data()),
        make_not_null(&one_divided_by_r),
        get(get<AngularCollocationsFor<swsh_derivative_tag>>(
                *angular_collocations))
            .data(),
        get(get<RadialPolyCoefficientsFor<swsh_derivative_tag>>(*radial_modes)),
        l_max, number_of_radial_points);
  };

  tmpl::for_each<TargetTagList>([
    &calculate_separable_for_pre_swsh_derivative, &
    calculate_separable_for_swsh_derivative
  ](auto target_tag_v) noexcept {
    using target_tag = typename decltype(target_tag_v)::type;
    tmpl::for_each<pre_swsh_derivative_tags_to_compute_for_t<target_tag>>(
        calculate_separable_for_pre_swsh_derivative);
    tmpl::for_each<single_swsh_derivative_tags_to_compute_for_t<target_tag>>(
        calculate_separable_for_swsh_derivative);
    tmpl::for_each<second_swsh_derivative_tags_to_compute_for_t<target_tag>>(
        calculate_separable_for_swsh_derivative);
  });
}
}  // namespace TestHelpers
}  // namespace Cce

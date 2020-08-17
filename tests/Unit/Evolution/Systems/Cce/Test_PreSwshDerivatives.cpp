// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <limits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/CceComputationTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {

namespace {
template <int Spin>
struct TestSpinWeightedScalar : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Spin>>;
};

using test_pre_swsh_derivative_dependencies =
    tmpl::list<Tags::BondiBeta, Tags::BondiJ, Tags::BondiQ, Tags::BondiU>;
}  // namespace

namespace detail {
// these are provided as a slimmed-down representation of a computational
// procedure that suffices to demonstrate that all of the template
// specializations of `PreSwshDerivatives` work correctly, rather than executing
// the full sequence of CCE steps, which would be too heavy for this test.
template <>
struct TagsToComputeForImpl<TestSpinWeightedScalar<0>> {
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiJ>, Tags::Dy<Tags::Dy<Tags::BondiJ>>,
                 Tags::Dy<Tags::BondiBeta>, Tags::Dy<Tags::Dy<Tags::BondiBeta>>,
                 Tags::Dy<Tags::BondiU>, Tags::Dy<Tags::Dy<Tags::BondiU>>,
                 Tags::Dy<Tags::BondiQ>, Tags::Dy<Tags::Dy<Tags::BondiQ>>>;
  using second_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
};

template <>
struct TagsToComputeForImpl<TestSpinWeightedScalar<1>> {
  using pre_swsh_derivative_tags = tmpl::list<
      Tags::BondiJbar, Tags::BondiUbar, Tags::BondiQbar,
      ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
      Tags::Dy<::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>>,
      Tags::Dy<Tags::Dy<::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>>>,
      ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>,
      ::Tags::Multiplies<Tags::BondiUbar, Tags::Dy<Tags::BondiJ>>,
      Tags::Dy<::Tags::Multiplies<Tags::BondiUbar, Tags::Dy<Tags::BondiJ>>>,
      Tags::Dy<::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>>>;
  using second_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
};
}  // namespace detail

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.PreSwshDerivatives",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(generator);
  const size_t l_max = 8;
  const size_t number_of_radial_grid_points = 8;

  using pre_swsh_derivative_tag_list = tmpl::append<
      pre_swsh_derivative_tags_to_compute_for_t<TestSpinWeightedScalar<0>>,
      pre_swsh_derivative_tags_to_compute_for_t<TestSpinWeightedScalar<1>>,
      test_pre_swsh_derivative_dependencies>;

  using pre_swsh_derivatives_variables_tag =
      ::Tags::Variables<pre_swsh_derivative_tag_list>;
  using swsh_derivatives_variables_tag =
      ::Tags::Variables<tmpl::list<Spectral::Swsh::Tags::Derivative<
          Tags::BondiBeta, Spectral::Swsh::Tags::Eth>>>;
  using separated_pre_swsh_derivatives_angular_data =
      ::Tags::Variables<db::wrap_tags_in<TestHelpers::AngularCollocationsFor,
                                         pre_swsh_derivative_tag_list>>;
  using separated_pre_swsh_derivatives_radial_modes =
      ::Tags::Variables<db::wrap_tags_in<TestHelpers::RadialPolyCoefficientsFor,
                                         pre_swsh_derivative_tag_list>>;

  const size_t number_of_grid_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
      number_of_radial_grid_points;

  auto expected_box = db::create<db::AddSimpleTags<
      pre_swsh_derivatives_variables_tag, swsh_derivatives_variables_tag,
      separated_pre_swsh_derivatives_angular_data,
      separated_pre_swsh_derivatives_radial_modes>>(
      typename pre_swsh_derivatives_variables_tag::type{number_of_grid_points},
      typename swsh_derivatives_variables_tag::type{number_of_grid_points, 0.0},
      typename separated_pre_swsh_derivatives_angular_data::type{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)},
      typename separated_pre_swsh_derivatives_radial_modes::type{
          number_of_radial_grid_points});

  db::mutate<pre_swsh_derivatives_variables_tag, swsh_derivatives_variables_tag,
             separated_pre_swsh_derivatives_angular_data,
             separated_pre_swsh_derivatives_radial_modes>(
      make_not_null(&expected_box),
      [&generator](
          const gsl::not_null<
              typename pre_swsh_derivatives_variables_tag::type*>
              pre_swsh_derivatives,
          const gsl::not_null<typename swsh_derivatives_variables_tag::type*>
              swsh_derivatives,
          const gsl::not_null<
              typename separated_pre_swsh_derivatives_angular_data::type*>
              pre_swsh_separated_angular_data,
          const gsl::not_null<
              typename separated_pre_swsh_derivatives_radial_modes::type*>
              pre_swsh_separated_radial_modes) {
        UniformCustomDistribution<double> dist(0.1, 1.0);
        SpinWeighted<ComplexDataVector, 0> boundary_r;
        boundary_r.data() =
            (10.0 +
             std::complex<double>(1.0, 0.0) *
                 make_with_random_values<DataVector>(
                     make_not_null(&generator), make_not_null(&dist),
                     Spectral::Swsh::number_of_swsh_collocation_points(l_max)));
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&boundary_r), l_max, l_max - 3);
        TestHelpers::generate_separable_expected<
            test_pre_swsh_derivative_dependencies,
            tmpl::list<TestSpinWeightedScalar<0>, TestSpinWeightedScalar<1>>>(
            pre_swsh_derivatives, swsh_derivatives,
            pre_swsh_separated_angular_data, pre_swsh_separated_radial_modes,
            make_not_null(&generator), boundary_r, l_max,
            number_of_radial_grid_points);
      });

  auto computation_box =
      db::create<db::AddSimpleTags<Tags::LMax, Tags::Integrand<Tags::BondiBeta>,
                                   Tags::Integrand<Tags::BondiU>,
                                   pre_swsh_derivatives_variables_tag,
                                   swsh_derivatives_variables_tag>>(
          l_max, db::get<Tags::Dy<Tags::BondiBeta>>(expected_box),
          db::get<Tags::Dy<Tags::BondiU>>(expected_box),
          typename pre_swsh_derivatives_variables_tag::type{
              number_of_grid_points, 0.0},
          typename swsh_derivatives_variables_tag::type{number_of_grid_points,
                                                        0.0});

  // duplicate the 'input' values to the computation box
  TestHelpers::CopyDataBoxTags<
      Tags::BondiBeta, Tags::BondiJ, Tags::BondiQ,
      Tags::BondiU>::apply(make_not_null(&computation_box), expected_box);

  mutate_all_pre_swsh_derivatives_for_tag<TestSpinWeightedScalar<0>>(
      make_not_null(&computation_box));
  mutate_all_pre_swsh_derivatives_for_tag<TestSpinWeightedScalar<1>>(
      make_not_null(&computation_box));
  // approximation needs a little bit of loosening to accommodate the comparison
  // between the 'separable' math and the more standard numerical procedures.
  Approx cce_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);

  CHECK_VARIABLES_CUSTOM_APPROX(
      db::get<pre_swsh_derivatives_variables_tag>(computation_box),
      db::get<pre_swsh_derivatives_variables_tag>(expected_box), cce_approx);

  // separately test the nonseparable Tags::JbarQMinus2EthBeta
  using pre_swsh_spare_variables_tag = ::Tags::Variables<
      tmpl::list<Tags::BondiJ, Tags::BondiQ, Tags::JbarQMinus2EthBeta,
                 Tags::DuRDividedByR, Tags::BondiH, Tags::OneMinusY,
                 Tags::Dy<Tags::BondiJ>, Tags::Du<Tags::BondiJ>>>;
  using swsh_derivatives_spare_variables_tag =
      ::Tags::Variables<tmpl::list<Spectral::Swsh::Tags::Derivative<
          Tags::BondiBeta, Spectral::Swsh::Tags::Eth>>>;
  auto spare_computation_box =
      db::create<db::AddSimpleTags<Tags::LMax, Tags::NumberOfRadialPoints,
                                   pre_swsh_spare_variables_tag,
                                   swsh_derivatives_spare_variables_tag>>(
          l_max, number_of_radial_grid_points,
          typename pre_swsh_spare_variables_tag::type{number_of_grid_points,
                                                      0.0},
          typename swsh_derivatives_spare_variables_tag::type{
              number_of_grid_points, 0.0});
  UniformCustomDistribution<double> dist(0.1, 1.0);
  tmpl::for_each<tmpl::list<Tags::BondiJ, Tags::BondiQ,
                            Spectral::Swsh::Tags::Derivative<
                                Tags::BondiBeta, Spectral::Swsh::Tags::Eth>,
                            Tags::DuRDividedByR, Tags::BondiH>>(
      [&spare_computation_box, &generator, &dist ](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        db::mutate<tag>(
            make_not_null(&spare_computation_box), [
              &generator, &dist
            ](const gsl::not_null<typename tag::type*> to_generate) noexcept {
              fill_with_random_values(to_generate, make_not_null(&generator),
                                      make_not_null(&dist));
            });
      });
  db::mutate_apply<
      PrecomputeCceDependencies<Tags::BoundaryValue, Tags::OneMinusY>>(
      make_not_null(&spare_computation_box));

  const auto& generated_j = get(db::get<Tags::BondiJ>(spare_computation_box));
  const auto& generated_q = get(db::get<Tags::BondiQ>(spare_computation_box));
  const auto& generated_eth_beta =
      get(db::get<Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                   Spectral::Swsh::Tags::Eth>>(
          spare_computation_box));
  db::mutate_apply<PreSwshDerivatives<Tags::JbarQMinus2EthBeta>>(
      make_not_null(&spare_computation_box));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::JbarQMinus2EthBeta>(spare_computation_box)).data(),
      conj(generated_j.data()) *
          (generated_q.data() - 2.0 * generated_eth_beta.data()));

  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
      make_not_null(&spare_computation_box));
  db::mutate_apply<PreSwshDerivatives<Tags::Du<Tags::BondiJ>>>(
      make_not_null(&spare_computation_box));

  const auto& generated_h = get(db::get<Tags::BondiH>(spare_computation_box));
  const auto& generated_dy_j =
      get(db::get<Tags::Dy<Tags::BondiJ>>(spare_computation_box));
  const auto& generated_du_r_divided_by_r =
      get(db::get<Tags::DuRDividedByR>(spare_computation_box));
  const auto& one_minus_y =
      get(db::get<Tags::OneMinusY>(spare_computation_box));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::Du<Tags::BondiJ>>(spare_computation_box)).data(),
      generated_h.data() - one_minus_y.data() *
                               generated_du_r_divided_by_r.data() *
                               generated_dy_j.data());
}
}  // namespace Cce

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <limits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/Evolution/Systems/Cce/CceComputationTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

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

  auto computation_box = db::create<db::AddSimpleTags<
      Spectral::Swsh::Tags::LMax, Tags::Integrand<Tags::BondiBeta>,
      Tags::Integrand<Tags::BondiU>, pre_swsh_derivatives_variables_tag,
      swsh_derivatives_variables_tag>>(
      l_max, db::get<Tags::Dy<Tags::BondiBeta>>(expected_box),
      db::get<Tags::Dy<Tags::BondiU>>(expected_box),
      typename pre_swsh_derivatives_variables_tag::type{number_of_grid_points,
                                                        0.0},
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
      tmpl::list<Tags::BondiJ, Tags::BondiQ, Tags::JbarQMinus2EthBeta>>;
  using swsh_derivatives_spare_variables_tag =
      ::Tags::Variables<tmpl::list<Spectral::Swsh::Tags::Derivative<
          Tags::BondiBeta, Spectral::Swsh::Tags::Eth>>>;
  auto spare_computation_box =
      db::create<db::AddSimpleTags<Spectral::Swsh::Tags::LMax,
                                   pre_swsh_spare_variables_tag,
                                   swsh_derivatives_spare_variables_tag>>(
          l_max,
          typename pre_swsh_spare_variables_tag::type{number_of_grid_points,
                                                      0.0},
          typename swsh_derivatives_spare_variables_tag::type{
              number_of_grid_points, 0.0});
  UniformCustomDistribution<double> dist(0.1, 1.0);
  ComplexDataVector generated_j = make_with_random_values<ComplexDataVector>(
      make_not_null(&generator), make_not_null(&dist), number_of_grid_points);
  ComplexDataVector generated_q = make_with_random_values<ComplexDataVector>(
      make_not_null(&generator), make_not_null(&dist), number_of_grid_points);
  ComplexDataVector generated_eth_beta =
      make_with_random_values<ComplexDataVector>(make_not_null(&generator),
                                                 make_not_null(&dist),
                                                 number_of_grid_points);
  db::mutate<Tags::BondiJ, Tags::BondiQ,
             Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                              Spectral::Swsh::Tags::Eth>>(
      make_not_null(&spare_computation_box),
      [&generated_j, &generated_q, &generated_eth_beta](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> q,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
              eth_beta) {
        get(*j) = generated_j;
        get(*q) = generated_q;
        get(*eth_beta) = generated_eth_beta;
      });
  db::mutate_apply<PreSwshDerivatives<Tags::JbarQMinus2EthBeta>>(
      make_not_null(&spare_computation_box));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::JbarQMinus2EthBeta>(spare_computation_box)).data(),
      conj(generated_j) * (generated_q - 2.0 * generated_eth_beta));
}
}  // namespace Cce

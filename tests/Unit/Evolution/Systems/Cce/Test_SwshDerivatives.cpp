// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <limits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Evolution/Systems/Cce/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
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
  static std::string name() noexcept { return "TestSpinWeightedScalar"; }
};

// note: J must be omitted from this due to the need for the Precomputation step
// in this test that was able to be omitted from the PreSwshDerivatives
// test.
using test_swsh_derivative_dependencies =
    tmpl::list<Tags::BondiBeta, Tags::BondiQ, Tags::BondiU>;
}  // namespace

namespace detail {
template <>
struct TagsToComputeForImpl<TestSpinWeightedScalar<0>> {
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiJ>, Tags::Dy<Tags::Dy<Tags::BondiJ>>,
                 Tags::Dy<Tags::BondiBeta>, Tags::Dy<Tags::Dy<Tags::BondiBeta>>,
                 Tags::Dy<Tags::BondiU>, Tags::Dy<Tags::Dy<Tags::BondiU>>,
                 Tags::Dy<Tags::BondiQ>, Tags::Dy<Tags::Dy<Tags::BondiQ>>>;
  using second_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::EthEth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::EthEthbar>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiJ>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::EthbarEthbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiQ, Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                       Spectral::Swsh::Tags::Eth>>;
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
      Tags::Dy<Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                Spectral::Swsh::Tags::Ethbar>>,
      Tags::Dy<::Tags::Multiplies<Tags::BondiUbar, Tags::Dy<Tags::BondiJ>>>,
      Tags::Dy<::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>>>;
  using second_swsh_derivative_tags =
      tmpl::list<Spectral::Swsh::Tags::Derivative<
          Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                           Spectral::Swsh::Tags::Ethbar>,
          Spectral::Swsh::Tags::Eth>>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          Tags::Dy<::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>>,
          Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<
          Tags::Dy<::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::EthEthbar>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiUbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Ethbar>>;
};
}  // namespace detail

namespace {
struct GenerateStartingData {
  template <typename Generator, typename Distribution>
  void operator()(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          boundary_r,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          boundary_du_r_divided_by_r,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
          angular_collocations_for_j,
      const gsl::not_null<Scalar<ComplexModalVector>*> radial_polynomials_for_j,
      const gsl::not_null<Generator*> generator,
      const gsl::not_null<Distribution*> dist, const size_t l_max,
      const size_t number_of_radial_grid_points,
      const ComplexDataVector& y) noexcept {
    get(*boundary_r).data() =
        (10.0 +
         std::complex<double>(1.0, 0.0) *
             make_with_random_values<DataVector>(
                 generator, dist,
                 Spectral::Swsh::number_of_swsh_collocation_points(l_max)));
    get(*boundary_du_r_divided_by_r).data() =
        std::complex<double>(1.0, 0.0) *
        make_with_random_values<DataVector>(
            generator, dist,
            Spectral::Swsh::number_of_swsh_collocation_points(l_max));
    // prevent aliasing; some terms are nonlinear in R
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*boundary_r)), l_max, 2);
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*boundary_du_r_divided_by_r)), l_max, 2);
    // generate the separable j needed for precomputation step
    get(*angular_collocations_for_j).data() =
        make_with_random_values<ComplexDataVector>(
            generator, dist,
            Spectral::Swsh::number_of_swsh_collocation_points(l_max));
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&get(*angular_collocations_for_j)), l_max, 2);
    get(*radial_polynomials_for_j) =
        make_with_random_values<ComplexModalVector>(
            generator, dist, number_of_radial_grid_points);
    for (size_t i = 0; i < number_of_radial_grid_points; ++i) {
      get(*radial_polynomials_for_j)[i] *=
          exp(-10.0 *
              pow<2>(static_cast<double>(i) /
                     static_cast<double>(number_of_radial_grid_points - 1)));
      if (i > 3){
        get(*radial_polynomials_for_j)[i] = 0.0;
      }
    }
    ComplexDataVector one_divided_by_r =
        (1.0 - y) /
        (2.0 * create_vector_of_n_copies(get(*boundary_r).data(),
                                         number_of_radial_grid_points));
    TestHelpers::generate_volume_data_from_separated_values(
        make_not_null(&get(*j).data()), make_not_null(&one_divided_by_r),
        get(*angular_collocations_for_j).data(), get(*radial_polynomials_for_j),
        l_max, number_of_radial_grid_points);
  }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.SwshDerivatives",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(generator);
  const size_t l_max = 10;
  const size_t number_of_radial_grid_points = 7;
  const size_t number_of_grid_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
      number_of_radial_grid_points;

  using pre_swsh_derivative_tag_list = tmpl::append<
      pre_swsh_derivative_tags_to_compute_for_t<TestSpinWeightedScalar<0>>,
      pre_swsh_derivative_tags_to_compute_for_t<TestSpinWeightedScalar<1>>,
      test_swsh_derivative_dependencies, tmpl::list<Tags::BondiJ>>;

  using swsh_derivative_tag_list = tmpl::remove_duplicates<tmpl::append<
      single_swsh_derivative_tags_to_compute_for_t<TestSpinWeightedScalar<0>>,
      second_swsh_derivative_tags_to_compute_for_t<TestSpinWeightedScalar<0>>,
      single_swsh_derivative_tags_to_compute_for_t<TestSpinWeightedScalar<1>>,
      second_swsh_derivative_tags_to_compute_for_t<TestSpinWeightedScalar<1>>>>;

  using boundary_value_variables_tag =
      ::Tags::Variables<pre_computation_boundary_tags<Tags::BoundaryValue>>;
  using integration_independent_variables_tag = ::Tags::Variables<
      tmpl::push_back<pre_computation_tags, Tags::DuRDividedByR>>;
  using pre_swsh_derivatives_variables_tag =
      ::Tags::Variables<pre_swsh_derivative_tag_list>;
  using swsh_derivatives_variables_tag =
      ::Tags::Variables<swsh_derivative_tag_list>;
  using swsh_derivatives_coefficient_buffer_variables_tag =
      ::Tags::Variables<tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
          swsh_derivative_tag_list,
          tmpl::bind<Spectral::Swsh::coefficient_buffer_tags_for_derivative_tag,
                     tmpl::_1>>>>>;
  using separated_angular_data_variables_tag = ::Tags::Variables<
      db::wrap_tags_in<TestHelpers::AngularCollocationsFor,
                       tmpl::append<pre_swsh_derivative_tag_list,
                                    swsh_derivative_tag_list>>>;
  using separated_radial_modes_variables_tag = ::Tags::Variables<
      db::wrap_tags_in<TestHelpers::RadialPolyCoefficientsFor,
                       tmpl::append<pre_swsh_derivative_tag_list,
                                    swsh_derivative_tag_list>>>;

  auto expected_box = db::create<db::AddSimpleTags<
    Spectral::Swsh::Tags::LMax, Spectral::Swsh::Tags::NumberOfRadialPoints,
      boundary_value_variables_tag, integration_independent_variables_tag,
      pre_swsh_derivatives_variables_tag, swsh_derivatives_variables_tag,
      separated_angular_data_variables_tag,
      separated_radial_modes_variables_tag>>(
      l_max, number_of_radial_grid_points,
      typename boundary_value_variables_tag::type{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0},
      typename integration_independent_variables_tag::type{
          number_of_grid_points, 0.0},
      typename pre_swsh_derivatives_variables_tag::type{number_of_grid_points},
      typename swsh_derivatives_variables_tag::type{number_of_grid_points, 0.0},
      typename separated_angular_data_variables_tag::type{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)},
      typename separated_radial_modes_variables_tag::type{
          number_of_radial_grid_points});

  // generate necessary boundary and integration-independent data for the rest
  // of the computation
  UniformCustomDistribution<double> dist(0.1, 1.0);
  const ComplexDataVector y = outer_product(
      ComplexDataVector{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 1.0},
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
          number_of_radial_grid_points));

  db::mutate<Tags::BoundaryValue<Tags::BondiR>,
             Tags::BoundaryValue<Tags::DuRDividedByR>, Tags::BondiJ,
             TestHelpers::AngularCollocationsFor<Tags::BondiJ>,
             TestHelpers::RadialPolyCoefficientsFor<Tags::BondiJ>>(
      make_not_null(&expected_box), GenerateStartingData{},
      make_not_null(&generator), make_not_null(&dist), l_max,
      number_of_radial_grid_points, y);

  // apply the separable computation for all of the pre_swsh_derivatives and
  // swsh_derivative quantities
  db::mutate<pre_swsh_derivatives_variables_tag, swsh_derivatives_variables_tag,
             separated_angular_data_variables_tag,
             separated_radial_modes_variables_tag>(
      make_not_null(&expected_box),
      [&generator](
          const gsl::not_null<
              typename pre_swsh_derivatives_variables_tag::type*>
              pre_swsh_derivatives,
          const gsl::not_null<typename swsh_derivatives_variables_tag::type*>
              swsh_derivatives,
          const gsl::not_null<
              typename separated_angular_data_variables_tag::type*>
              separated_angular_data,
          const gsl::not_null<
              typename separated_radial_modes_variables_tag::type*>
              separated_radial_modes,
          const SpinWeighted<ComplexDataVector, 0>& boundary_r) {
        TestHelpers::generate_separable_expected<
            test_swsh_derivative_dependencies,
            tmpl::list<TestSpinWeightedScalar<0>, TestSpinWeightedScalar<1>>>(
            pre_swsh_derivatives, swsh_derivatives, separated_angular_data,
            separated_radial_modes, make_not_null(&generator), boundary_r,
            l_max, number_of_radial_grid_points);
      },
      get(db::get<Tags::BoundaryValue<Tags::BondiR>>(expected_box)));

  auto computation_box = db::create<db::AddSimpleTags<
      Spectral::Swsh::Tags::LMax, Spectral::Swsh::Tags::NumberOfRadialPoints,
      Tags::Integrand<Tags::BondiBeta>, Tags::Integrand<Tags::BondiU>,
      boundary_value_variables_tag, integration_independent_variables_tag,
      pre_swsh_derivatives_variables_tag, swsh_derivatives_variables_tag,
      swsh_derivatives_coefficient_buffer_variables_tag>>(
      l_max, number_of_radial_grid_points,
      db::get<Tags::Dy<Tags::BondiBeta>>(expected_box),
      db::get<Tags::Dy<Tags::BondiU>>(expected_box),
      typename boundary_value_variables_tag::type{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0},
      typename integration_independent_variables_tag::type{
          number_of_grid_points, 0.0},
      typename pre_swsh_derivatives_variables_tag::type{number_of_grid_points,
                                                        0.0},
      typename swsh_derivatives_variables_tag::type{number_of_grid_points, 0.0},
      typename swsh_derivatives_coefficient_buffer_variables_tag::type{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max) *
              number_of_radial_grid_points,
          0.0});

  TestHelpers::CopyDataBoxTags<
      Tags::BoundaryValue<Tags::BondiR>,
      Tags::BoundaryValue<Tags::DuRDividedByR>,
      Tags::BondiJ>::apply(make_not_null(&computation_box), expected_box);

  mutate_all_precompute_cce_dependencies<Tags::BoundaryValue>(
      make_not_null(&computation_box));
  mutate_all_precompute_cce_dependencies<Tags::BoundaryValue>(
      make_not_null(&expected_box));

  // duplicate the 'input' values to the computation box
  TestHelpers::CopyDataBoxTags<
      Tags::BondiBeta, Tags::BondiJ, Tags::BondiQ,
      Tags::BondiU>::apply(make_not_null(&computation_box), expected_box);

  mutate_all_pre_swsh_derivatives_for_tag<TestSpinWeightedScalar<0>>(
      make_not_null(&computation_box));

  mutate_all_swsh_derivatives_for_tag<TestSpinWeightedScalar<0>>(
      make_not_null(&computation_box));

  mutate_all_pre_swsh_derivatives_for_tag<TestSpinWeightedScalar<1>>(
      make_not_null(&computation_box));

  mutate_all_swsh_derivatives_for_tag<TestSpinWeightedScalar<1>>(
      make_not_null(&computation_box));

  // this can be tightened at the cost of needing a higher resolution due to the
  // inherent aliasing in this system. A loose approx allows the test to be
  // (relatively) fast
  Approx loose_cce_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e8)
          .scale(1.0);

  CHECK_VARIABLES_CUSTOM_APPROX(
      db::get<swsh_derivatives_variables_tag>(computation_box),
      db::get<swsh_derivatives_variables_tag>(expected_box), loose_cce_approx);
}
}  // namespace Cce

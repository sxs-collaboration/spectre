// Distributed under the MIT License.
// See LICENSE.txt for details

#include "Framework/TestingFramework.hpp"

#include <algorithm>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/CceComputationTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Utilities/VectorAlgebra.hpp"

namespace Cce {

namespace {

// the first several helper functions in this file are used to create and
// process the polynomial data that will be used to create the expected values
// in each of the integration routines

ComplexDataVector radial_vector_from_power_series(
    const ComplexModalVector& powers,
    const ComplexDataVector& one_minus_y) noexcept {
  ComplexDataVector result{one_minus_y.size(), powers[0]};
  for (size_t i = 1; i < powers.size(); ++i) {
    // use of TestHelpers::power due to an internal bug in blaze powers of
    // Complex vectors
    result += powers[i] * TestHelpers::power(one_minus_y, i);
  }
  return result;
}

template <typename BondiValueTag, typename DataBoxTagList>
void make_boundary_data(const gsl::not_null<db::DataBox<DataBoxTagList>*> box,
                        const gsl::not_null<ComplexDataVector*> expected,
                        const size_t l_max) noexcept {
  db::mutate<Tags::BoundaryValue<BondiValueTag>>(
      box,
      [
        &expected, &l_max
      ](const gsl::not_null<db::item_type<Tags::BoundaryValue<BondiValueTag>>*>
            boundary) noexcept {
        get(*boundary).data() = ComplexDataVector{
            expected->data(),
            Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
      });
}

template <typename DataBoxTagList, typename Generator, typename Distribution,
          typename... Tags>
void generate_powers_for_tags(
    const gsl::not_null<db::DataBox<DataBoxTagList>*> box,
    const gsl::not_null<Generator*> generator,
    const gsl::not_null<Distribution*> distribution,
    const size_t number_of_modes, tmpl::list<Tags...> /*meta*/) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT(
      db::mutate<TestHelpers::RadialPolyCoefficientsFor<Tags>>(
          box,
          [](const gsl::not_null<Scalar<ComplexModalVector>*> modes,
             const gsl::not_null<Generator*> gen,
             const gsl::not_null<Distribution*> dist,
             const size_t lambda_number_of_modes) noexcept {
            get(*modes) = make_with_random_values<ComplexModalVector>(
                gen, dist, lambda_number_of_modes);
          },
          generator, distribution, number_of_modes));
}

template <typename Tag, typename DataBoxTagList>
void zero_top_modes(const gsl::not_null<db::DataBox<DataBoxTagList>*> box,
                    const size_t number_of_modes_to_zero,
                    const size_t total_number_of_modes) noexcept {
  db::mutate<TestHelpers::RadialPolyCoefficientsFor<Tag>>(
      box, [
        &number_of_modes_to_zero, &total_number_of_modes
      ](const gsl::not_null<Scalar<ComplexModalVector>*> modes) noexcept {
        for (size_t i = total_number_of_modes - number_of_modes_to_zero;
             i < total_number_of_modes; ++i) {
          get(*modes)[i] = 0.0;
        }
      });
}

template <typename DataBoxTagList, typename... Tags>
void generate_volume_data_from_separable(
    const gsl::not_null<db::DataBox<DataBoxTagList>*> box,
    const ComplexDataVector& angular_data, const ComplexDataVector& one_minus_y,
    tmpl::list<Tags...> /*meta*/) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT(db::mutate<Tags>(
      box,
      [](auto to_fill, const ComplexDataVector& lambda_one_minus_y,
         const ComplexDataVector& lambda_angular_data,
         const Scalar<ComplexModalVector>& modes) noexcept {
        get(*to_fill).data() = outer_product(
            lambda_angular_data,
            radial_vector_from_power_series(get(modes), lambda_one_minus_y));
      },
      one_minus_y, angular_data,
      db::get<TestHelpers::RadialPolyCoefficientsFor<Tags>>(*box)));
}

template <typename BondiValueTag>
auto create_box_for_bondi_integration(
    const size_t l_max, const size_t number_of_radial_grid_points,
    const size_t number_of_radial_polynomials) noexcept {
  const size_t number_of_grid_points =
      number_of_radial_grid_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  using integration_tags = tmpl::flatten<tmpl::list<
      BondiValueTag, typename RadialIntegrateBondi<
                         Tags::BoundaryValue, BondiValueTag>::integrand_tags>>;
  using integration_variables_tag = ::Tags::Variables<integration_tags>;
  using integration_modes_variables_tag =
      ::Tags::Variables<db::wrap_tags_in<TestHelpers::RadialPolyCoefficientsFor,
                                         integration_tags>>;

  return db::create<db::AddSimpleTags<
      integration_variables_tag, Tags::BoundaryValue<BondiValueTag>,
      integration_modes_variables_tag, Tags::LMax, Tags::NumberOfRadialPoints,
      Tags::OneMinusY>>(
      db::item_type<integration_variables_tag>{number_of_grid_points},
      db::item_type<Tags::BoundaryValue<BondiValueTag>>{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)},
      db::item_type<integration_modes_variables_tag>{
          number_of_radial_polynomials},
      l_max, number_of_radial_grid_points,
      Scalar<SpinWeighted<ComplexDataVector, 0>>{number_of_grid_points});
}

// This utility tests the standard integration (which really just applies the
// standard indefinite integral to each radial slice)
template <typename BondiValueTag, typename Generator>
void test_regular_integration(const gsl::not_null<Generator*> gen,
                              const size_t number_of_radial_grid_points,
                              const size_t l_max) noexcept {
  UniformCustomDistribution<double> dist(0.1, 5.0);
  const size_t number_of_radial_polynomials = 5;
  auto box = create_box_for_bondi_integration<BondiValueTag>(
      l_max, number_of_radial_grid_points, number_of_radial_polynomials);

  generate_powers_for_tags(make_not_null(&box), gen, make_not_null(&dist),
                           number_of_radial_polynomials,
                           tmpl::list<BondiValueTag>{});

  // use the above powers to infer the powers for the derivative
  db::mutate<
      TestHelpers::RadialPolyCoefficientsFor<Tags::Integrand<BondiValueTag>>>(
      make_not_null(&box),
      [](const gsl::not_null<Scalar<ComplexModalVector>*> integrand_modes,
         const Scalar<ComplexModalVector>& bondi_value_modes) noexcept {
        for (size_t i = 0; i < get(bondi_value_modes).size() - 1; ++i) {
          // sign change because these are modes of 1 - y.
          get(*integrand_modes)[i] =
              -static_cast<double>(i + 1) * get(bondi_value_modes)[i + 1];
        }
        get(*integrand_modes)[get(bondi_value_modes).size() - 1] = 0.0;
      },
      db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(box));

  const ComplexDataVector one_minus_y =
      std::complex<double>(1.0, 0.0) *
      (1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto>(
                 number_of_radial_grid_points));

  const auto random_angular_data = make_with_random_values<ComplexDataVector>(
      gen, make_not_null(&dist),
      Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  ComplexDataVector expected = outer_product(
      random_angular_data,
      radial_vector_from_power_series(
          get(db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(
              box)),
          one_minus_y));

  generate_volume_data_from_separable(
      make_not_null(&box), random_angular_data, one_minus_y,
      tmpl::list<Tags::Integrand<BondiValueTag>>{});

  make_boundary_data<BondiValueTag>(make_not_null(&box),
                                    make_not_null(&expected), l_max);

  db::mutate_apply<RadialIntegrateBondi<Tags::BoundaryValue, BondiValueTag>>(
      make_not_null(&box));

  Approx numerical_differentiation_approximation =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(expected,
                               get(db::get<BondiValueTag>(box)).data(),
                               numerical_differentiation_approximation);
}

// This algorithm tests the Q and W integration algorithms
template <typename BondiValueTag, typename Generator>
void test_pole_integration(const gsl::not_null<Generator*> gen,
                           const size_t number_of_radial_grid_points,
                           const size_t l_max) noexcept {
  UniformCustomDistribution<double> dist(0.1, 5.0);
  const size_t number_of_radial_polynomials = 5;

  auto box = create_box_for_bondi_integration<BondiValueTag>(
      l_max, number_of_radial_grid_points, number_of_radial_polynomials);

  generate_powers_for_tags(
      make_not_null(&box), gen, make_not_null(&dist),
      number_of_radial_polynomials,
      tmpl::list<BondiValueTag, Tags::PoleOfIntegrand<BondiValueTag>>{});

  // use the above powers to infer the powers for regular part of the integrand
  // The coefficients of the polynomials used in the below mutations can be
  // determined by expanding both sides of the differential equation
  // (1 - y) \partial_y f + 2 f = A + (1 - y) B,
  // and matching order-by-order in (1 - y).
  db::mutate<TestHelpers::RadialPolyCoefficientsFor<
      Tags::PoleOfIntegrand<BondiValueTag>>>(
      make_not_null(&box),
      [](const gsl::not_null<Scalar<ComplexModalVector>*>
             pole_of_integrand_modes,
         const Scalar<ComplexModalVector>& bondi_value_modes) noexcept {
        get(*pole_of_integrand_modes)[0] = 2.0 * get(bondi_value_modes)[0];
      },
      db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(box));

  db::mutate<TestHelpers::RadialPolyCoefficientsFor<
      Tags::RegularIntegrand<BondiValueTag>>>(
      make_not_null(&box),
      [](const gsl::not_null<Scalar<ComplexModalVector>*>
             regular_integrand_modes,
         const Scalar<ComplexModalVector>& bondi_value_modes,
         const Scalar<ComplexModalVector>& pole_integrand_modes) noexcept {
        for (size_t i = 0; i < get(bondi_value_modes).size() - 1; ++i) {
          // sign change because these are modes of 1 - y.
          get(*regular_integrand_modes)[i] =
              -get(pole_integrand_modes)[i + 1] +
              (1.0 - static_cast<double>(i)) * get(bondi_value_modes)[i + 1];
        }
        get(*regular_integrand_modes)[get(bondi_value_modes).size() - 1] = 0.0;
      },
      db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(box),
      db::get<TestHelpers::RadialPolyCoefficientsFor<
          Tags::PoleOfIntegrand<BondiValueTag>>>(box));

  const ComplexDataVector one_minus_y =
      std::complex<double>(1.0, 0.0) *
      (1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto>(
                 number_of_radial_grid_points));

  const auto random_angular_data = make_with_random_values<ComplexDataVector>(
      gen, make_not_null(&dist),
      Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  ComplexDataVector expected = outer_product(
      random_angular_data,
      radial_vector_from_power_series(
          get(db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(
              box)),
          one_minus_y));

  generate_volume_data_from_separable(
      make_not_null(&box), random_angular_data, one_minus_y,
      tmpl::list<Tags::PoleOfIntegrand<BondiValueTag>,
                 Tags::RegularIntegrand<BondiValueTag>>{});

  make_boundary_data<BondiValueTag>(make_not_null(&box),
                                    make_not_null(&expected), l_max);

  db::mutate<Tags::OneMinusY>(make_not_null(&box),
                              TestHelpers::volume_one_minus_y, l_max);

  db::mutate_apply<RadialIntegrateBondi<Tags::BoundaryValue, BondiValueTag>>(
      make_not_null(&box));

  Approx numerical_differentiation_approximation =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(expected,
                               get(db::get<BondiValueTag>(box)).data(),
                               numerical_differentiation_approximation);
}

// this utility tests the H integration algorithm
template <typename BondiValueTag, typename Generator>
void test_pole_integration_with_linear_operator(
    const gsl::not_null<Generator*> gen, size_t number_of_radial_grid_points,
    size_t l_max) noexcept {
  // The typical linear solve performed during realistic CCE evolution involves
  // fairly small wave amplitudes
  UniformCustomDistribution<double> dist(0.01, 0.1);
  const size_t number_of_radial_polynomials = 6;

  auto box = create_box_for_bondi_integration<BondiValueTag>(
      l_max, number_of_radial_grid_points, number_of_radial_polynomials);

  generate_powers_for_tags(
      make_not_null(&box), gen, make_not_null(&dist),
      number_of_radial_polynomials,
      tmpl::list<BondiValueTag, Tags::PoleOfIntegrand<BondiValueTag>,
                 Tags::LinearFactor<BondiValueTag>,
                 Tags::LinearFactorForConjugate<BondiValueTag>>{});

  // In the full treatment of the CCE equations the linear factor is 1.0
  // asymptotically (near y = 1.0) and the linear factor of the conjugate is 0.0
  // asymptotically, with often small perturbation at linear and higher orders.
  // Here we set that leading behavior explicitly.
  db::mutate<
      TestHelpers::RadialPolyCoefficientsFor<Tags::LinearFactor<BondiValueTag>>,
      TestHelpers::RadialPolyCoefficientsFor<
          Tags::LinearFactorForConjugate<BondiValueTag>>>(
      make_not_null(&box),
      [](const gsl::not_null<
             db::item_type<TestHelpers::RadialPolyCoefficientsFor<
                 Tags::LinearFactor<BondiValueTag>>>*>
             linear_factor_modes,
         const gsl::not_null<
             db::item_type<TestHelpers::RadialPolyCoefficientsFor<
                 Tags::LinearFactorForConjugate<BondiValueTag>>>*>
             linear_factor_of_conjugate_modes) noexcept {
        get(*linear_factor_modes)[0] = 1.0;
        get(*linear_factor_of_conjugate_modes)[0] = 0.0;
      });

  zero_top_modes<BondiValueTag>(make_not_null(&box), 2,
                                number_of_radial_polynomials);
  zero_top_modes<Tags::LinearFactor<BondiValueTag>>(
      make_not_null(&box), 3, number_of_radial_polynomials);
  zero_top_modes<Tags::LinearFactorForConjugate<BondiValueTag>>(
      make_not_null(&box), 3, number_of_radial_polynomials);

  const ComplexDataVector one_minus_y =
      std::complex<double>(1.0, 0.0) *
      (1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto>(
                 number_of_radial_grid_points));

  // generate random modes rather than random collocation values to ensure
  // representability, and to filter the last couple of l modes so that aliasing
  // doesn't hurt precision in the nonlinear formulas below.
  auto random_angular_data = make_with_random_values<ComplexDataVector>(
      gen, make_not_null(&dist),
      Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  SpinWeighted<ComplexModalVector, 0> random_angular_modes{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  fill_with_random_values(make_not_null(&random_angular_modes), gen,
                          make_not_null(&dist));
  const auto& coefficients_metadata =
      Spectral::Swsh::cached_coefficients_metadata(l_max);
  for (const auto& mode : coefficients_metadata) {
    if (mode.l > l_max - 2) {
      random_angular_modes.data()[mode.transform_of_real_part_offset] = 0.0;
      random_angular_modes.data()[mode.transform_of_imag_part_offset] = 0.0;
    }
  }
  SpinWeighted<ComplexDataVector, 0> filter_buffer;
  filter_buffer.set_data_ref(random_angular_data.data(),
                             random_angular_data.size());
  Spectral::Swsh::inverse_swsh_transform(
      l_max, 1, make_not_null(&filter_buffer), random_angular_modes);

  generate_volume_data_from_separable(
      make_not_null(&box), random_angular_data, one_minus_y,
      tmpl::list<Tags::PoleOfIntegrand<BondiValueTag>,
                 Tags::LinearFactor<BondiValueTag>,
                 Tags::LinearFactorForConjugate<BondiValueTag>>{});
  // unlike the above tests, the nonlinear operators in the H equation ensures
  // that we have to actually manually build up the last operator to ensure
  // consistency (from the above data, the pole of integrand is not separable).
  db::mutate<Tags::PoleOfIntegrand<BondiValueTag>,
             Tags::RegularIntegrand<BondiValueTag>>(
      make_not_null(&box),
      [
        &random_angular_data, &l_max, &one_minus_y, &
        number_of_radial_grid_points
      ](const gsl::not_null<
            db::item_type<Tags::PoleOfIntegrand<BondiValueTag>>*>
            pole_integrand,
        const gsl::not_null<
            db::item_type<Tags::RegularIntegrand<BondiValueTag>>*>
            regular_integrand,
        const Scalar<ComplexModalVector>& bondi_value_modes,
        const Scalar<ComplexModalVector>& pole_integrand_modes,
        const Scalar<ComplexModalVector>& linear_factor_modes,
        const Scalar<ComplexModalVector>&
            linear_factor_of_conjugate_modes) noexcept {
        for (size_t i = 0; i < number_of_radial_grid_points; ++i) {
          ComplexDataVector angular_view_for_pole_integrand{
              get(*pole_integrand).data().data() +
                  i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
              Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
          ComplexDataVector angular_view_for_regular_integrand{
              get(*regular_integrand).data().data() +
                  i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
              Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

          angular_view_for_pole_integrand +=
              square(random_angular_data) * get(bondi_value_modes)[0] *
                  get(linear_factor_modes)[0] +
              random_angular_data * conj(random_angular_data) *
                  conj(get(bondi_value_modes)[0]) *
                  get(linear_factor_of_conjugate_modes)[0] -
              random_angular_data * get(pole_integrand_modes)[0];
          angular_view_for_regular_integrand = 0.0;

          for (size_t j = 0; j < 5; ++j) {
            angular_view_for_regular_integrand +=
                (-static_cast<double>(j + 1) * get(bondi_value_modes)[j + 1] -
                 get(pole_integrand_modes)[j + 1]) *
                random_angular_data *
                (j == 0
                     ? 1.0
                     : (one_minus_y[i] == 0.0 ? 0.0 : pow(one_minus_y[i], j)));
            for (size_t k =
                     static_cast<size_t>(std::max(0, static_cast<int>(j) - 2));
                 k < std::min(j + 2, size_t{3}); ++k) {
              angular_view_for_regular_integrand +=
                  (square(random_angular_data) *
                       (get(linear_factor_modes)[k] *
                        get(bondi_value_modes)[(j + 1) - k]) +
                   random_angular_data * conj(random_angular_data) *
                       (get(linear_factor_of_conjugate_modes)[k] *
                        conj(get(bondi_value_modes)[(j + 1) - k]))) *
                  (j == 0 ? 1.0
                          : (real(one_minus_y[i]) == 0.0
                                 ? 0.0
                                 : pow(one_minus_y[i], j)));
            }
          }
        }
      },
      db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(box),
      db::get<TestHelpers::RadialPolyCoefficientsFor<
          Tags::PoleOfIntegrand<BondiValueTag>>>(box),
      db::get<TestHelpers::RadialPolyCoefficientsFor<
          Tags::LinearFactor<BondiValueTag>>>(box),
      db::get<TestHelpers::RadialPolyCoefficientsFor<
          Tags::LinearFactorForConjugate<BondiValueTag>>>(box));
  ComplexDataVector expected = outer_product(
      random_angular_data,
      radial_vector_from_power_series(
          get(db::get<TestHelpers::RadialPolyCoefficientsFor<BondiValueTag>>(
              box)),
          one_minus_y));

  make_boundary_data<BondiValueTag>(make_not_null(&box),
                                    make_not_null(&expected), l_max);

  db::mutate<Tags::OneMinusY>(make_not_null(&box),
                              TestHelpers::volume_one_minus_y, l_max);

  db::mutate_apply<RadialIntegrateBondi<Tags::BoundaryValue, BondiValueTag>>(
      make_not_null(&box));

  Approx numerical_differentiation_approximation =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);
  INFO("number of radial grid points: " << number_of_radial_grid_points);
  CHECK_ITERABLE_CUSTOM_APPROX(expected,
                               get(db::get<BondiValueTag>(box)).data(),
                               numerical_differentiation_approximation);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.LinearSolve", "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{3, 6};
  const size_t l_max = sdist(gen);
  const size_t number_of_radial_grid_points = sdist(gen) + 4;

  test_regular_integration<Tags::BondiBeta>(
      make_not_null(&gen), number_of_radial_grid_points, l_max);
  test_regular_integration<Tags::BondiU>(make_not_null(&gen),
                                         number_of_radial_grid_points, l_max);
  test_pole_integration<Tags::BondiQ>(make_not_null(&gen),
                                      number_of_radial_grid_points, l_max);
  test_pole_integration<Tags::BondiW>(make_not_null(&gen),
                                      number_of_radial_grid_points, l_max);
  test_pole_integration_with_linear_operator<Tags::BondiH>(
      make_not_null(&gen), number_of_radial_grid_points, l_max);
}
}  // namespace
}  // namespace Cce

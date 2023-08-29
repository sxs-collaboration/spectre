// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/VectorAlgebra.hpp"

namespace Cce {
namespace {

struct InverseCubicEvolutionGauge {
  using boundary_tags =
      tmpl::list<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::BondiJ>>,
                 Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>>;

  using mutate_tags = tmpl::list<Tags::BondiJ, Tags::CauchyCartesianCoords,
                                 Tags::CauchyAngularCoords>;
  using argument_tags =
      tmpl::append<boundary_tags,
                   tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>>;

  InverseCubicEvolutionGauge() = default;

  void operator()(
      const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      const gsl::not_null<tnsr::i<DataVector, 3>*> /*cartesian_x_of_x_tilde*/,
      const gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
      /*angular_cauchy_coordinates*/,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const size_t /*l_max*/, const size_t number_of_radial_points) const {
    const DataVector one_minus_y_collocation =
        1.0 - Spectral::collocation_points<
                  SpatialDiscretization::Basis::Legendre,
                  SpatialDiscretization::Quadrature::GaussLobatto>(
                  number_of_radial_points);
    for (size_t i = 0; i < number_of_radial_points; i++) {
      ComplexDataVector angular_view_j{
          get(*j).data().data() + get(boundary_j).size() * i,
          get(boundary_j).size()};
      // auto is acceptable here as these two values are only used once in the
      // below computation. `auto` causes an expression template to be
      // generated, rather than allocating.
      const auto one_minus_y_coefficient =
          0.25 * (3.0 * get(boundary_j).data() +
                  get(r).data() * get(boundary_dr_j).data());
      const auto one_minus_y_cubed_coefficient =
          -0.0625 *
          (get(boundary_j).data() + get(r).data() * get(boundary_dr_j).data());
      angular_view_j =
          one_minus_y_collocation[i] * one_minus_y_coefficient +
          pow<3>(one_minus_y_collocation[i]) * one_minus_y_cubed_coefficient;
    }
  }
};

// These gauge transforms are extremely hard to validate outside of a true
// evolution system. Here we settle for verifying that for an
// analytically-generated set of worldtube data, the transform for a
// well-behaved set of coordinates is the inverse of the transform for the
// inverse coordinate functions. This only verifies basic properties of the
// transform; full validation can only come from tests of evolving systems.
template <typename Generator>
void test_gauge_transforms_via_inverse_coordinate_map(
    const gsl::not_null<Generator*> gen) {
  const size_t l_max = 12;
  const size_t number_of_radial_grid_points = 10;
  const size_t number_of_angular_grid_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  using real_boundary_tags =
      tmpl::list<Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords,
                 Tags::PartiallyFlatAngularCoords,
                 Tags::PartiallyFlatCartesianCoords,
                 ::Tags::dt<Tags::CauchyCartesianCoords>,
                 ::Tags::dt<Tags::PartiallyFlatCartesianCoords>>;
  using spin_weighted_boundary_tags = tmpl::flatten<tmpl::list<
      tmpl::list<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                 Tags::PartiallyFlatGaugeOmega, Tags::CauchyGaugeC,
                 Tags::CauchyGaugeD, Tags::CauchyGaugeOmega,
                 Tags::Du<Tags::PartiallyFlatGaugeOmega>,
                 Spectral::Swsh::Tags::Derivative<Tags::PartiallyFlatGaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::CauchyGaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::BondiUAtScri>,
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>,
      Tags::characteristic_worldtube_boundary_tags<
          Tags::EvolutionGaugeBoundaryValue>>>;
  using coordinate_variables_tag = ::Tags::Variables<real_boundary_tags>;
  using spin_weighted_variables_tag =
      ::Tags::Variables<spin_weighted_boundary_tags>;
  using volume_spin_weighted_variables_tag = ::Tags::Variables<
      tmpl::list<Tags::BondiJ, Tags::Dy<Tags::BondiJ>, Tags::BondiU>>;

  auto forward_transform_box = db::create<db::AddSimpleTags<
      coordinate_variables_tag, spin_weighted_variables_tag,
      volume_spin_weighted_variables_tag, Tags::LMax,
      Tags::NumberOfRadialPoints,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::SwshInterpolator<
          Tags::PartiallyFlatAngularCoords>>>(
      typename coordinate_variables_tag::type{number_of_angular_grid_points},
      typename spin_weighted_variables_tag::type{number_of_angular_grid_points},
      typename volume_spin_weighted_variables_tag::type{
          number_of_angular_grid_points * number_of_radial_grid_points},
      l_max, number_of_radial_grid_points, Spectral::Swsh::SwshInterpolator{},
      Spectral::Swsh::SwshInterpolator{});

  // create analytic data for the forward transform
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0;

  // acceptable parameters for the fake sinusoid variation in the input
  // parameters
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);

  const size_t libsharp_size =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
  tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{libsharp_size};
  Scalar<ComplexModalVector> lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dt_lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dr_lapse_coefficients{libsharp_size};
  TestHelpers::create_fake_time_varying_modal_data(
      make_not_null(&spatial_metric_coefficients),
      make_not_null(&dt_spatial_metric_coefficients),
      make_not_null(&dr_spatial_metric_coefficients),
      make_not_null(&shift_coefficients), make_not_null(&dt_shift_coefficients),
      make_not_null(&dr_shift_coefficients), make_not_null(&lapse_coefficients),
      make_not_null(&dt_lapse_coefficients),
      make_not_null(&dr_lapse_coefficients), solution, extraction_radius,
      amplitude, frequency, target_time, l_max, false);

  db::mutate<spin_weighted_variables_tag>(
      [&spatial_metric_coefficients, &dt_spatial_metric_coefficients,
       &dr_spatial_metric_coefficients, &shift_coefficients,
       &dt_shift_coefficients, &dr_shift_coefficients, &lapse_coefficients,
       &dt_lapse_coefficients, &dr_lapse_coefficients, &extraction_radius](
          const gsl::not_null<Variables<spin_weighted_boundary_tags>*>
              spin_weighted_boundary_variables) {
        create_bondi_boundary_data(
            spin_weighted_boundary_variables, spatial_metric_coefficients,
            dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
            shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
            lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
            extraction_radius, l_max);
      },
      make_not_null(&forward_transform_box));

  // construct the coordinate transform quantities
  const double variation_amplitude = value_dist(*gen);
  const double variation_amplitude_inertial = value_dist(*gen);
  db::mutate<Tags::CauchyCartesianCoords>(
      [&l_max,
       &variation_amplitude](const gsl::not_null<tnsr::i<DataVector, 3>*>
                                 cauchy_cartesian_coordinates) {
        tnsr::i<DataVector, 2> cauchy_angular_coordinates{
            get<0>(*cauchy_cartesian_coordinates).size()};
        // There is a bug in Clang 10.0.0 that gives a nonsensical
        // error message for the following call to
        // cached_collocation_metadata unless l_max is captured in
        // this lambda. The capture should not be necessary because l_max is a
        // const integer type that is initialized by a constant expression. Note
        // that l_max need not be declared constexpr for its value to be
        // retrieved inside a lambda without capturing it. This
        // line silences the warning that says capturing l_max is not necessary.
        (void)l_max;
        const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
            Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
        for (const auto collocation_point : collocation) {
          get<1>(cauchy_angular_coordinates)[collocation_point.offset] =
              collocation_point.phi + 1.0e-2 * variation_amplitude *
                                          cos(collocation_point.phi) *
                                          sin(collocation_point.theta);
          get<0>(cauchy_angular_coordinates)[collocation_point.offset] =
              collocation_point.theta;
        }
        get<0>(*cauchy_cartesian_coordinates) =
            sin(get<0>(cauchy_angular_coordinates)) *
            cos(get<1>(cauchy_angular_coordinates));
        get<1>(*cauchy_cartesian_coordinates) =
            sin(get<0>(cauchy_angular_coordinates)) *
            sin(get<1>(cauchy_angular_coordinates));
        get<2>(*cauchy_cartesian_coordinates) =
            cos(get<0>(cauchy_angular_coordinates));
      },
      make_not_null(&forward_transform_box));

  db::mutate<Tags::PartiallyFlatCartesianCoords>(
      [&l_max, &variation_amplitude_inertial](
          const gsl::not_null<tnsr::i<DataVector, 3>*>
              inertial_cartesian_coordinates) {
        tnsr::i<DataVector, 2> inertial_angular_coordinates{
            get<0>(*inertial_cartesian_coordinates).size()};
        // There is a bug in Clang 10.0.0 that gives a nonsensical
        // error message for the following call to
        // cached_collocation_metadata unless l_max is captured in
        // this lambda. The capture should not be necessary because l_max is a
        // const integer type that is initialized by a constant expression. Note
        // that l_max need not be declared constexpr for its value to be
        // retrieved inside a lambda without capturing it. This
        // line silences the warning that says capturing l_max is not necessary.
        (void)l_max;
        const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
            Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
        for (const auto& collocation_point : collocation) {
          get<1>(inertial_angular_coordinates)[collocation_point.offset] =
              collocation_point.phi + 1.0e-2 * variation_amplitude_inertial *
                                          cos(collocation_point.phi) *
                                          sin(collocation_point.theta);
          get<0>(inertial_angular_coordinates)[collocation_point.offset] =
              collocation_point.theta;
        }
        get<0>(*inertial_cartesian_coordinates) =
            sin(get<0>(inertial_angular_coordinates)) *
            cos(get<1>(inertial_angular_coordinates));
        get<1>(*inertial_cartesian_coordinates) =
            sin(get<0>(inertial_angular_coordinates)) *
            sin(get<1>(inertial_angular_coordinates));
        get<2>(*inertial_cartesian_coordinates) =
            cos(get<0>(inertial_angular_coordinates));
      },
      make_not_null(&forward_transform_box));

  auto inverse_transform_box = db::create<db::AddSimpleTags<
      coordinate_variables_tag, spin_weighted_variables_tag,
      volume_spin_weighted_variables_tag, Tags::LMax,
      Tags::NumberOfRadialPoints,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::SwshInterpolator<
          Tags::PartiallyFlatAngularCoords>>>(
      typename coordinate_variables_tag::type{number_of_angular_grid_points},
      typename spin_weighted_variables_tag::type{number_of_angular_grid_points},
      typename volume_spin_weighted_variables_tag::type{
          number_of_angular_grid_points * number_of_radial_grid_points},
      l_max, number_of_radial_grid_points, Spectral::Swsh::SwshInterpolator{},
      Spectral::Swsh::SwshInterpolator{});

  db::mutate<Tags::CauchyCartesianCoords>(
      [&l_max,
       &variation_amplitude](const gsl::not_null<tnsr::i<DataVector, 3>*>
                                 inverse_cauchy_cartesian_coordinates) {
        tnsr::i<DataVector, 2> inverse_cauchy_angular_coordinates{
            get<0>(*inverse_cauchy_cartesian_coordinates).size()};
        // There is a bug in Clang 10.0.0 that gives a nonsensical
        // error message for the following call to
        // cached_collocation_metadata unless l_max is captured in
        // this lambda. The capture should not be necessary because l_max is a
        // const integer type that is initialized by a constant expression. Note
        // that l_max need not be declared constexpr for its value to be
        // retrieved inside a lambda without capturing it. This
        // line silences the warning that says capturing l_max is not necessary.
        (void)l_max;
        const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
            Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
        for (const auto collocation_point : collocation) {
          auto rootfind = RootFinder::toms748(
              [&collocation_point, &variation_amplitude](const double x) {
                return collocation_point.phi -
                       (x + 1.0e-2 * variation_amplitude * cos(x) *
                                sin(collocation_point.theta));
              },
              collocation_point.phi - 2.0e-2, collocation_point.phi + 2.0e-2,
              1.0e-15, 1.0e-15);
          get<1>(inverse_cauchy_angular_coordinates)[collocation_point.offset] =
              rootfind;
          get<0>(inverse_cauchy_angular_coordinates)[collocation_point.offset] =
              collocation_point.theta;
        }
        get<0>(*inverse_cauchy_cartesian_coordinates) =
            sin(get<0>(inverse_cauchy_angular_coordinates)) *
            cos(get<1>(inverse_cauchy_angular_coordinates));
        get<1>(*inverse_cauchy_cartesian_coordinates) =
            sin(get<0>(inverse_cauchy_angular_coordinates)) *
            sin(get<1>(inverse_cauchy_angular_coordinates));
        get<2>(*inverse_cauchy_cartesian_coordinates) =
            cos(get<0>(inverse_cauchy_angular_coordinates));
      },
      make_not_null(&inverse_transform_box));

  db::mutate<Tags::PartiallyFlatCartesianCoords>(
      [&l_max, &variation_amplitude_inertial](
          const gsl::not_null<tnsr::i<DataVector, 3>*>
              inverse_inertial_cartesian_coordinates) {
        tnsr::i<DataVector, 2> inverse_inertial_angular_coordinates{
            get<0>(*inverse_inertial_cartesian_coordinates).size()};
        // There is a bug in Clang 10.0.0 that gives a nonsensical
        // error message for the following call to
        // cached_collocation_metadata unless l_max is captured in
        // this lambda. The capture should not be necessary because l_max is a
        // const integer type that is initialized by a constant expression. Note
        // that l_max need not be declared constexpr for its value to be
        // retrieved inside a lambda without capturing it. This
        // line silences the warning that says capturing l_max is not necessary.
        (void)l_max;
        const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
            Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
        for (const auto& collocation_point : collocation) {
          auto rootfind = RootFinder::toms748(
              [&collocation_point,
               &variation_amplitude_inertial](const double x) {
                return collocation_point.phi -
                       (x + 1.0e-2 * variation_amplitude_inertial * cos(x) *
                                sin(collocation_point.theta));
              },
              collocation_point.phi - 2.0e-2, collocation_point.phi + 2.0e-2,
              1.0e-15, 1.0e-15);
          get<1>(
              inverse_inertial_angular_coordinates)[collocation_point.offset] =
              rootfind;
          get<0>(
              inverse_inertial_angular_coordinates)[collocation_point.offset] =
              collocation_point.theta;
        }
        get<0>(*inverse_inertial_cartesian_coordinates) =
            sin(get<0>(inverse_inertial_angular_coordinates)) *
            cos(get<1>(inverse_inertial_angular_coordinates));
        get<1>(*inverse_inertial_cartesian_coordinates) =
            sin(get<0>(inverse_inertial_angular_coordinates)) *
            sin(get<1>(inverse_inertial_angular_coordinates));
        get<2>(*inverse_inertial_cartesian_coordinates) =
            cos(get<0>(inverse_inertial_angular_coordinates));
      },
      make_not_null(&inverse_transform_box));

  {
    INFO("Checking GaugeUpdateAngularFromCartesian");
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&forward_transform_box));
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::PartiallyFlatAngularCoords, Tags::PartiallyFlatCartesianCoords>>(
        make_not_null(&forward_transform_box));
    double angular_phi = 0.0;
    double angular_phi_inertial = 0.0;
    const auto& computed_angular_cauchy_coordinates =
        db::get<Tags::CauchyAngularCoords>(forward_transform_box);
    const auto& computed_angular_inertial_coordinates =
        db::get<Tags::PartiallyFlatAngularCoords>(forward_transform_box);
    const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto collocation_point : collocation) {
      angular_phi = collocation_point.phi + 1.0e-2 * variation_amplitude *
                                                cos(collocation_point.phi) *
                                                sin(collocation_point.theta);
      angular_phi_inertial =
          collocation_point.phi + 1.0e-2 * variation_amplitude_inertial *
                                      cos(collocation_point.phi) *
                                      sin(collocation_point.theta);
      CHECK(
          get<1>(
              computed_angular_cauchy_coordinates)[collocation_point.offset] ==
          approx((angular_phi > M_PI) ? angular_phi - 2.0 * M_PI
                                      : angular_phi));
      CHECK(
          get<0>(
              computed_angular_cauchy_coordinates)[collocation_point.offset] ==
          approx(collocation_point.theta));
      CHECK(get<1>(computed_angular_inertial_coordinates)[collocation_point
                                                              .offset] ==
            approx((angular_phi_inertial > M_PI)
                       ? angular_phi_inertial - 2.0 * M_PI
                       : angular_phi_inertial));
      CHECK(get<0>(computed_angular_inertial_coordinates)[collocation_point
                                                              .offset] ==
            approx(collocation_point.theta));
    }
  }

  {
    INFO("Checking GaugeUpdateJacobianFromCoordinates and GaugeUpdateOmega");
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&inverse_transform_box));
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::PartiallyFlatAngularCoords, Tags::PartiallyFlatCartesianCoords>>(
        make_not_null(&inverse_transform_box));

    db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
        make_not_null(&forward_transform_box));
    db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
        make_not_null(&inverse_transform_box));

    db::mutate_apply<GaugeUpdateInterpolator<Tags::PartiallyFlatAngularCoords>>(
        make_not_null(&forward_transform_box));
    db::mutate_apply<GaugeUpdateInterpolator<Tags::PartiallyFlatAngularCoords>>(
        make_not_null(&inverse_transform_box));

    db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
        Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&forward_transform_box));
    db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
        Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&inverse_transform_box));

    db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
        Tags::CauchyGaugeC, Tags::CauchyGaugeD,
        Tags::PartiallyFlatAngularCoords, Tags::PartiallyFlatCartesianCoords>>(
        make_not_null(&forward_transform_box));
    db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
        Tags::CauchyGaugeC, Tags::CauchyGaugeD,
        Tags::PartiallyFlatAngularCoords, Tags::PartiallyFlatCartesianCoords>>(
        make_not_null(&inverse_transform_box));

    db::mutate_apply<
        GaugeUpdateOmega<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                         Tags::PartiallyFlatGaugeOmega>>(
        make_not_null(&forward_transform_box));
    db::mutate_apply<
        GaugeUpdateOmega<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                         Tags::PartiallyFlatGaugeOmega>>(
        make_not_null(&inverse_transform_box));

    db::mutate_apply<GaugeUpdateOmega<Tags::CauchyGaugeC, Tags::CauchyGaugeD,
                                      Tags::CauchyGaugeOmega>>(
        make_not_null(&forward_transform_box));
    db::mutate_apply<GaugeUpdateOmega<Tags::CauchyGaugeC, Tags::CauchyGaugeD,
                                      Tags::CauchyGaugeOmega>>(
        make_not_null(&inverse_transform_box));

    const auto& forward_cauchy_angular_coordinates =
        db::get<Tags::CauchyAngularCoords>(forward_transform_box);
    const auto& inverse_cauchy_angular_coordinates =
        db::get<Tags::CauchyAngularCoords>(inverse_transform_box);

    const auto& forward_inertial_angular_coordinates =
        db::get<Tags::PartiallyFlatAngularCoords>(forward_transform_box);
    const auto& inverse_inertial_angular_coordinates =
        db::get<Tags::PartiallyFlatAngularCoords>(inverse_transform_box);

    const Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(forward_cauchy_angular_coordinates),
        get<1>(forward_cauchy_angular_coordinates), l_max};

    const Spectral::Swsh::SwshInterpolator inverse_interpolator{
        get<0>(inverse_cauchy_angular_coordinates),
        get<1>(inverse_cauchy_angular_coordinates), l_max};

    const Spectral::Swsh::SwshInterpolator interpolator_inertial{
        get<0>(forward_inertial_angular_coordinates),
        get<1>(forward_inertial_angular_coordinates), l_max};

    const Spectral::Swsh::SwshInterpolator inverse_interpolator_inertial{
        get<0>(inverse_inertial_angular_coordinates),
        get<1>(inverse_inertial_angular_coordinates), l_max};

    Scalar<SpinWeighted<ComplexDataVector, 0>> interpolated_forward_omega_cd{
        number_of_angular_grid_points};
    Scalar<SpinWeighted<ComplexDataVector, 0>>
        interpolated_forward_omega_cauchy_cd{number_of_angular_grid_points};

    // check that the coordinates are actually inverses of one another.
    interpolator.interpolate(
        make_not_null(&get(interpolated_forward_omega_cd)),
        get(db::get<Tags::PartiallyFlatGaugeOmega>(inverse_transform_box)));
    interpolator_inertial.interpolate(
        make_not_null(&get(interpolated_forward_omega_cauchy_cd)),
        get(db::get<Tags::CauchyGaugeOmega>(inverse_transform_box)));

    SpinWeighted<ComplexDataVector, 0>
        forward_and_inverse_interpolated_omega_cd{
            number_of_angular_grid_points};
    inverse_interpolator.interpolate(
        make_not_null(&forward_and_inverse_interpolated_omega_cd),
        get(interpolated_forward_omega_cd));

    SpinWeighted<ComplexDataVector, 0>
        forward_and_inverse_interpolated_omega_cauchy_cd{
            number_of_angular_grid_points};
    inverse_interpolator_inertial.interpolate(
        make_not_null(&forward_and_inverse_interpolated_omega_cauchy_cd),
        get(interpolated_forward_omega_cauchy_cd));

    const auto& check_rhs =
        get(db::get<Tags::PartiallyFlatGaugeOmega>(inverse_transform_box))
            .data();
    CHECK_ITERABLE_APPROX(forward_and_inverse_interpolated_omega_cd.data(),
                          check_rhs);
    const auto& check_rhs_cauchy =
        get(db::get<Tags::CauchyGaugeOmega>(inverse_transform_box)).data();
    CHECK_ITERABLE_APPROX(
        forward_and_inverse_interpolated_omega_cauchy_cd.data(),
        check_rhs_cauchy);

    interpolator.interpolate(
        make_not_null(&get(interpolated_forward_omega_cd)),
        get(db::get<Tags::PartiallyFlatGaugeOmega>(inverse_transform_box)));
    const auto& inverse_omega_cd =
        db::get<Tags::PartiallyFlatGaugeOmega>(forward_transform_box);
    const auto another_check_rhs = 1.0 / (get(inverse_omega_cd).data());
    CHECK_ITERABLE_APPROX(get(interpolated_forward_omega_cd).data(),
                          another_check_rhs);

    interpolator_inertial.interpolate(
        make_not_null(&get(interpolated_forward_omega_cauchy_cd)),
        get(db::get<Tags::CauchyGaugeOmega>(inverse_transform_box)));
    const auto& inverse_omega_cauchy_cd =
        db::get<Tags::CauchyGaugeOmega>(forward_transform_box);
    const auto another_check_rhs_cauchy =
        1.0 / (get(inverse_omega_cauchy_cd).data());
    CHECK_ITERABLE_APPROX(get(interpolated_forward_omega_cauchy_cd).data(),
                          another_check_rhs_cauchy);
  }

  Approx interpolation_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  const auto check_gauge_adjustment_against_inverse = [&forward_transform_box,
                                                       &inverse_transform_box,
                                                       &interpolation_approx](
                                                          auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    INFO("computing tag : " << db::tag_name<tag>());
    db::mutate_apply<GaugeAdjustedBoundaryValue<tag>>(
        make_not_null(&forward_transform_box));
    db::mutate<Tags::BoundaryValue<tag>>(
        [](const gsl::not_null<typename Tags::BoundaryValue<tag>::type*>
               inverse_transform_boundary_value,
           const typename Tags::EvolutionGaugeBoundaryValue<tag>::type&
               forward_transform_evolution_gauge_value) {
          *inverse_transform_boundary_value =
              forward_transform_evolution_gauge_value;
        },
        make_not_null(&inverse_transform_box),
        db::get<Tags::EvolutionGaugeBoundaryValue<tag>>(forward_transform_box));
    if (std::is_same_v<tag, Tags::BondiQ>) {
      // populate dr_u in the inverse box using the equation of motion.
      db::mutate<Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>>(
          [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
                 dr_u,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
             const Scalar<SpinWeighted<ComplexDataVector, 1>>& q,
             const Scalar<SpinWeighted<ComplexDataVector, 2>>& j) {
            SpinWeighted<ComplexDataVector, 0> k;
            k.data() = sqrt(1.0 + get(j).data() * conj(get(j).data()));
            get(*dr_u).data() = exp(2.0 * get(beta).data()) /
                                square(get(r).data()) *
                                (k.data() * get(q).data() -
                                 get(j).data() * conj(get(q).data()));
          },
          make_not_null(&inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::BondiBeta>>(inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::BondiR>>(inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::BondiQ>>(inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::BondiJ>>(inverse_transform_box));
    }
    if (std::is_same_v<tag, Tags::BondiH>) {
      db::mutate<Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>>(
          [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                 du_j,
             const Scalar<SpinWeighted<ComplexDataVector, 2>>& h,
             const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>&
                 du_r_divided_by_r) {
            get(*du_j) = get(h) - get(r) * get(du_r_divided_by_r) * get(dr_j);
          },
          make_not_null(&inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::BondiH>>(inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
              inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::BondiR>>(inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::DuRDividedByR>>(
              inverse_transform_box));
    }
    db::mutate_apply<GaugeAdjustedBoundaryValue<tag>>(
        make_not_null(&inverse_transform_box));

    // Once we've done the same transform with the the forward and inverse
    // coordinate maps, check that the final quantity is approximately the
    // input.
    const auto& check_lhs =
        db::get<Tags::BoundaryValue<tag>>(forward_transform_box);
    const auto& check_rhs =
        db::get<Tags::EvolutionGaugeBoundaryValue<tag>>(inverse_transform_box);
    CHECK_ITERABLE_CUSTOM_APPROX(check_lhs, check_rhs, interpolation_approx);
  };

  using first_phase_gauge_adjustments =
      tmpl::list<Tags::BondiR, Tags::BondiJ, Tags::Dr<Tags::BondiJ>>;
  {
    INFO(
        "Checking first part of GaugeAdjustedBoundaryValue computations "
        "(before the initial data computation)");
    tmpl::for_each<first_phase_gauge_adjustments>(
        check_gauge_adjustment_against_inverse);
  }

  db::mutate_apply<InverseCubicEvolutionGauge::mutate_tags,
                   InverseCubicEvolutionGauge::argument_tags>(
      InverseCubicEvolutionGauge{}, make_not_null(&forward_transform_box));
  db::mutate_apply<InverseCubicEvolutionGauge::mutate_tags,
                   InverseCubicEvolutionGauge::argument_tags>(
      InverseCubicEvolutionGauge{}, make_not_null(&inverse_transform_box));

  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
      make_not_null(&forward_transform_box));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
      make_not_null(&inverse_transform_box));

  using second_phase_gauge_adjustments =
      tmpl::list<Tags::BondiBeta, Tags::BondiU, Tags::BondiQ>;
  {
    INFO(
        "Checking second part of GaugeAdjustedBoundaryValue "
        "computations (before the BondiUAtScri computation)");
    tmpl::for_each<second_phase_gauge_adjustments>(
        check_gauge_adjustment_against_inverse);
  }

  // Set up a reasonable volume U and compute the corresponding value in the
  // inverse gauge so that we can appropriately check the transform for the
  // second phase. This is used only to determine the rate of change of the
  // angular coordinates, and does not need to be consistent with the boundary
  // conditions.
  db::mutate<Tags::BondiU>(
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
             bondi_u,
         const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary_u) {
        const ComplexDataVector time_transform = 10.0 * get(boundary_u).data();
        fill_with_n_copies(make_not_null(&(get(*bondi_u).data())),
                           time_transform, number_of_radial_grid_points);
      },
      make_not_null(&forward_transform_box),
      db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiU>>(
          forward_transform_box));

  // choose a U value for the inverse transform that results in the appropriate
  // inverse time dependence for the inverse coordinate transformation.
  db::mutate<Tags::BondiU, Tags::BoundaryValue<Tags::BondiU>>(
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
             bondi_u,
         const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
             boundary_u,
         const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
         const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
         const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
         const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
             x_of_x_tilde) {
        Spectral::Swsh::SwshInterpolator interpolator{
            get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max};
        SpinWeighted<ComplexDataVector, 1> evolution_coords_u{
            get(*boundary_u).size()};
        interpolator.interpolate(make_not_null(&evolution_coords_u),
                                 get(*boundary_u));

        const ComplexDataVector minus_u =
            10.0 * (-0.5 / square(get(omega_cd).data()) *
                    (-get(c).data() * conj(evolution_coords_u.data()) +
                     conj(get(d).data()) * evolution_coords_u.data()));
        fill_with_n_copies(make_not_null(&(get(*bondi_u).data())), minus_u,
                           number_of_radial_grid_points);
      },
      make_not_null(&inverse_transform_box),
      db::get<Tags::PartiallyFlatGaugeC>(inverse_transform_box),
      db::get<Tags::PartiallyFlatGaugeD>(inverse_transform_box),
      db::get<Tags::PartiallyFlatGaugeOmega>(inverse_transform_box),
      db::get<Tags::CauchyAngularCoords>(inverse_transform_box));

  // subtract off the gauge adjustments from the gauge for the inverse
  // transformation box so that the remaining quantities are all in a consistent
  // gauge. This additional manipulation is not performed by the standard gauge
  // transform as the extra operations are not required for typical evaluation.
  db::mutate<Tags::EvolutionGaugeBoundaryValue<Tags::BondiU>>(
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
             evolution_gauge_boundary_u,
         const Scalar<SpinWeighted<ComplexDataVector, 1>>&
             evolution_gauge_full_u) {
        const ComplexDataVector evolution_gauge_u_scri_slice;
        make_const_view(
            make_not_null(&evolution_gauge_u_scri_slice),
            get(evolution_gauge_full_u).data(),
            (number_of_radial_grid_points - 1) *
                Spectral::Swsh::number_of_swsh_collocation_points(l_max),
            Spectral::Swsh::number_of_swsh_collocation_points(l_max));

        // we add instead of subtracting to save the inconvenience of extracting
        // the value from the other box and again performing the jacobian
        // manipulations used in determining the gauge alteration in the first
        // place.
        get(*evolution_gauge_boundary_u).data() += evolution_gauge_u_scri_slice;
      },
      make_not_null(&inverse_transform_box),
      db::get<Tags::BondiU>(inverse_transform_box));

  db::mutate_apply<GaugeUpdateTimeDerivatives>(
      make_not_null(&forward_transform_box));
  db::mutate_apply<GaugeUpdateTimeDerivatives>(
      make_not_null(&inverse_transform_box));

  // check that the du_omega_cd are as expected.
  const auto& x_of_x_tilde =
      db::get<Tags::CauchyAngularCoords>(inverse_transform_box);
  const auto& forward_du_omega_cd =
      db::get<Tags::Du<Tags::PartiallyFlatGaugeOmega>>(forward_transform_box);
  const auto& forward_u_0 = db::get<Tags::BondiUAtScri>(forward_transform_box);
  const auto& forward_eth_omega_cd =
      db::get<Spectral::Swsh::Tags::Derivative<Tags::PartiallyFlatGaugeOmega,
                                               Spectral::Swsh::Tags::Eth>>(
          forward_transform_box);
  const auto& forward_omega_cd =
      get(db::get<Tags::PartiallyFlatGaugeOmega>(forward_transform_box));
  const SpinWeighted<ComplexDataVector, 0>
      forward_adjusted_du_omega_cd_over_omega =
          (get(forward_du_omega_cd) -
           0.5 * (get(forward_u_0) * conj(get(forward_eth_omega_cd)) +
                  conj(get(forward_u_0)) * get(forward_eth_omega_cd))) /
          forward_omega_cd;
  const Spectral::Swsh::SwshInterpolator omega_interpolator{
      get<0>(x_of_x_tilde), get<1>(x_of_x_tilde), l_max};

  SpinWeighted<ComplexDataVector, 0> omega_comparison_lhs{
      number_of_angular_grid_points};
  omega_interpolator.interpolate(make_not_null(&omega_comparison_lhs),
                                 forward_adjusted_du_omega_cd_over_omega);
  const SpinWeighted<ComplexDataVector, 0> omega_comparison_rhs =
      -get(db::get<Tags::Du<Tags::PartiallyFlatGaugeOmega>>(
          inverse_transform_box)) /
      get(db::get<Tags::PartiallyFlatGaugeOmega>(inverse_transform_box));

  // check that the eth_omega_cd are as expected.
  SpinWeighted<ComplexDataVector, 0> interpolated_forward_omega{
      number_of_angular_grid_points};
  omega_interpolator.interpolate(make_not_null(&interpolated_forward_omega),
                                 forward_omega_cd);

  SpinWeighted<ComplexDataVector, 1> eth_interpolated_forward_omega{
      number_of_angular_grid_points};
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, make_not_null(&eth_interpolated_forward_omega),
      interpolated_forward_omega);

  const auto& inverse_c =
      db::get<Tags::PartiallyFlatGaugeC>(inverse_transform_box);
  const auto& inverse_d =
      db::get<Tags::PartiallyFlatGaugeD>(inverse_transform_box);

  SpinWeighted<ComplexDataVector, 1> eth_omega{number_of_angular_grid_points};
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, make_not_null(&eth_omega), forward_omega_cd);

  SpinWeighted<ComplexDataVector, 1> interpolated_forward_eth_omega_cd{
      number_of_angular_grid_points};
  omega_interpolator.interpolate(
      make_not_null(&interpolated_forward_eth_omega_cd),
      get(forward_eth_omega_cd));

  SpinWeighted<ComplexDataVector, 1> eth_omega_comparison_lhs;
  eth_omega_comparison_lhs.data() =
      (0.5 * ((get(inverse_c).data() *
               conj(interpolated_forward_eth_omega_cd.data())) +
              conj(get(inverse_d).data()) *
                  interpolated_forward_eth_omega_cd.data()));

  CHECK_ITERABLE_CUSTOM_APPROX(eth_omega_comparison_lhs,
                               eth_interpolated_forward_omega,
                               interpolation_approx);

  using third_phase_gauge_adjustments =
      tmpl::list<Tags::DuRDividedByR, Tags::BondiH, Tags::BondiW>;
  {
    INFO(
        "Checking third part of GaugeAdjustedBoundaryValue "
        "computations (after the BondiUAtScri computation)");
    tmpl::for_each<third_phase_gauge_adjustments>(
        check_gauge_adjustment_against_inverse);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.GaugeTransformBoundaryData",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_gauge_transforms_via_inverse_coordinate_map(make_not_null(&gen));
}
}  // namespace Cce

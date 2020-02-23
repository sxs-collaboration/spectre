// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/InitializeCce.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/VectorAlgebra.hpp"
#include "tests/Unit/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {
namespace {

// These gauge transforms are extremely hard to validate outside of a true
// evolution system. Here we settle for verifying that for an
// analytically-generated set of worldtube data, the transform for a
// well-behaved set of coordinates is the inverse of the transform for the
// inverse coordinate functions. This only verifies basic properties of the
// transform; full validation can only come from tests of evolving systems.
template <typename Generator>
void test_gauge_transforms_via_inverse_coordinate_map(
    const gsl::not_null<Generator*> gen) noexcept {
  const size_t l_max = 12;
  const size_t number_of_radial_grid_points = 10;
  const size_t number_of_angular_grid_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  using real_boundary_tags =
      tmpl::list<Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords,
                 ::Tags::dt<Tags::CauchyCartesianCoords>>;
  using spin_weighted_boundary_tags = tmpl::flatten<tmpl::list<
      tmpl::list<Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmega,
                 Tags::Du<Tags::GaugeOmega>,
                 Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
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
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>>(
      db::item_type<coordinate_variables_tag>{number_of_angular_grid_points},
      db::item_type<spin_weighted_variables_tag>{number_of_angular_grid_points},
      db::item_type<volume_spin_weighted_variables_tag>{
          number_of_angular_grid_points * number_of_radial_grid_points},
      l_max, number_of_radial_grid_points, Spectral::Swsh::SwshInterpolator{});

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

  create_bondi_boundary_data(
      make_not_null(&forward_transform_box), spatial_metric_coefficients,
      dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
      shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
      lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
      extraction_radius, l_max);

  // construct the coordinate transform quantities
  const double variation_amplitude = value_dist(*gen);
  db::mutate<Tags::CauchyCartesianCoords>(
      make_not_null(&forward_transform_box),
      [&variation_amplitude](const gsl::not_null<tnsr::i<DataVector, 3>*>
                                 cauchy_cartesian_coordinates) noexcept {
        tnsr::i<DataVector, 2> cauchy_angular_coordinates{
            get<0>(*cauchy_cartesian_coordinates).size()};
        const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
            Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
        for (const auto& collocation_point : collocation) {
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
      });

  auto inverse_transform_box = db::create<db::AddSimpleTags<
      coordinate_variables_tag, spin_weighted_variables_tag,
      volume_spin_weighted_variables_tag, Tags::LMax,
      Tags::NumberOfRadialPoints,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>>(
      db::item_type<coordinate_variables_tag>{number_of_angular_grid_points},
      db::item_type<spin_weighted_variables_tag>{number_of_angular_grid_points},
      db::item_type<volume_spin_weighted_variables_tag>{
          number_of_angular_grid_points * number_of_radial_grid_points},
      l_max, number_of_radial_grid_points, Spectral::Swsh::SwshInterpolator{});

  db::mutate<Tags::CauchyCartesianCoords>(
      make_not_null(&inverse_transform_box),
      [&variation_amplitude](
          const gsl::not_null<tnsr::i<DataVector, 3>*>
              inverse_cauchy_cartesian_coordinates) noexcept {
        tnsr::i<DataVector, 2> inverse_cauchy_angular_coordinates{
            get<0>(*inverse_cauchy_cartesian_coordinates).size()};
        const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
            Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
        for (const auto& collocation_point : collocation) {
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
      });

  {
    INFO("Checking GaugeUpdateAngularFromCartesian");
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&forward_transform_box));
    double angular_phi;
    const auto& computed_angular_coordinates =
        db::get<Tags::CauchyAngularCoords>(forward_transform_box);
    const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      angular_phi = collocation_point.phi + 1.0e-2 * variation_amplitude *
                                                cos(collocation_point.phi) *
                                                sin(collocation_point.theta);
      CHECK(get<1>(computed_angular_coordinates)[collocation_point.offset] ==
            approx((angular_phi > M_PI) ? angular_phi - 2.0 * M_PI
                                        : angular_phi));
      CHECK(get<0>(computed_angular_coordinates)[collocation_point.offset] ==
            approx(collocation_point.theta));
    }
  }

  {
    INFO("Checking GaugeUpdateJacobianFromCoordinates and GaugeUpdateOmega");
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&inverse_transform_box));

    db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
        make_not_null(&forward_transform_box));
    db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
        make_not_null(&inverse_transform_box));

    db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
        Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>>(make_not_null(&forward_transform_box));
    db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
        Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
        Tags::CauchyCartesianCoords>>(make_not_null(&inverse_transform_box));

    db::mutate_apply<GaugeUpdateOmega>(make_not_null(&forward_transform_box));
    db::mutate_apply<GaugeUpdateOmega>(make_not_null(&inverse_transform_box));

    const auto& forward_cauchy_angular_coordinates =
        db::get<Tags::CauchyAngularCoords>(forward_transform_box);
    const auto& inverse_cauchy_angular_coordinates =
        db::get<Tags::CauchyAngularCoords>(inverse_transform_box);

    const Spectral::Swsh::SwshInterpolator interpolator{
        get<0>(forward_cauchy_angular_coordinates),
        get<1>(forward_cauchy_angular_coordinates), l_max};

    const Spectral::Swsh::SwshInterpolator inverse_interpolator{
        get<0>(inverse_cauchy_angular_coordinates),
        get<1>(inverse_cauchy_angular_coordinates), l_max};

    Scalar<SpinWeighted<ComplexDataVector, 0>> interpolated_forward_omega_cd{
        number_of_angular_grid_points};
    // check that the coordinates are actually inverses of one another.
    interpolator.interpolate(
        make_not_null(&get(interpolated_forward_omega_cd)),
        get(db::get<Tags::GaugeOmega>(inverse_transform_box)));

    SpinWeighted<ComplexDataVector, 0>
        forward_and_inverse_interpolated_omega_cd{
            number_of_angular_grid_points};
    inverse_interpolator.interpolate(
        make_not_null(&forward_and_inverse_interpolated_omega_cd),
        get(interpolated_forward_omega_cd));

    const auto& check_rhs =
        get(db::get<Tags::GaugeOmega>(inverse_transform_box)).data();
    CHECK_ITERABLE_APPROX(forward_and_inverse_interpolated_omega_cd.data(),
                          check_rhs);

    interpolator.interpolate(
        make_not_null(&get(interpolated_forward_omega_cd)),
        get(db::get<Tags::GaugeOmega>(inverse_transform_box)));
    const auto& inverse_omega_cd =
        db::get<Tags::GaugeOmega>(forward_transform_box);
    const auto another_check_rhs = 1.0 / (get(inverse_omega_cd).data());
    CHECK_ITERABLE_APPROX(get(interpolated_forward_omega_cd).data(),
                          another_check_rhs);
  }

  Approx interpolation_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  const auto check_gauge_adjustment_against_inverse =
      [&forward_transform_box, &inverse_transform_box, &
       interpolation_approx ](auto tag_v) noexcept {
    using tag = typename decltype(tag_v)::type;
    INFO("computing tag : " << db::tag_name<tag>());
    db::mutate_apply<GaugeAdjustedBoundaryValue<tag>>(
        make_not_null(&forward_transform_box));
    db::mutate<Tags::BoundaryValue<tag>>(
        make_not_null(&inverse_transform_box),
        [](const gsl::not_null<db::item_type<Tags::BoundaryValue<tag>>*>
               inverse_transform_boundary_value,
           const db::item_type<Tags::EvolutionGaugeBoundaryValue<tag>>&
               forward_transform_evolution_gauge_value) noexcept {
          *inverse_transform_boundary_value =
              forward_transform_evolution_gauge_value;
        },
        db::get<Tags::EvolutionGaugeBoundaryValue<tag>>(forward_transform_box));
    if (cpp17::is_same_v<tag, Tags::BondiQ>) {
      // populate dr_u in the inverse box using the equation of motion.
      db::mutate<Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>>(
          make_not_null(&inverse_transform_box),
          [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
                 dr_u,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
             const Scalar<SpinWeighted<ComplexDataVector, 1>>& q,
             const Scalar<SpinWeighted<ComplexDataVector, 2>>& j) noexcept {
            SpinWeighted<ComplexDataVector, 0> k;
            k.data() = sqrt(1.0 + get(j).data() * conj(get(j).data()));
            get(*dr_u).data() = exp(2.0 * get(beta).data()) /
                                square(get(r).data()) *
                                (k.data() * get(q).data() -
                                 get(j).data() * conj(get(q).data()));
          },
          db::get<Tags::BoundaryValue<Tags::BondiBeta>>(inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::BondiR>>(inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::BondiQ>>(inverse_transform_box),
          db::get<Tags::BoundaryValue<Tags::BondiJ>>(inverse_transform_box));
    }
    if (cpp17::is_same_v<tag, Tags::BondiH>) {
      db::mutate<Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>>(
          make_not_null(&inverse_transform_box),
          [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                 du_j,
             const Scalar<SpinWeighted<ComplexDataVector, 2>>& h,
             const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
             const Scalar<SpinWeighted<ComplexDataVector, 0>>&
                 du_r_divided_by_r) noexcept {
            get(*du_j) = get(h) - get(r) * get(du_r_divided_by_r) * get(dr_j);
          },
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

  db::mutate_apply<InitializeJ<Tags::EvolutionGaugeBoundaryValue>>(
      make_not_null(&forward_transform_box));
  db::mutate_apply<InitializeJ<Tags::EvolutionGaugeBoundaryValue>>(
      make_not_null(&inverse_transform_box));

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
      make_not_null(&forward_transform_box),
      [
      ](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
            bondi_u,
        const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary_u) noexcept {
        const ComplexDataVector time_transform = 10.0 * get(boundary_u).data();
        fill_with_n_copies(make_not_null(&(get(*bondi_u).data())),
                           time_transform, number_of_radial_grid_points);
      },
      db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiU>>(
          forward_transform_box));

  // choose a U value for the inverse transform that results in the appropriate
  // inverse time dependence for the inverse coordinate transformation.
  db::mutate<Tags::BondiU, Tags::BoundaryValue<Tags::BondiU>>(
      make_not_null(&inverse_transform_box),
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
             bondi_u,
         const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
             boundary_u,
         const Scalar<SpinWeighted<ComplexDataVector, 2>>& c,
         const Scalar<SpinWeighted<ComplexDataVector, 0>>& d,
         const Scalar<SpinWeighted<ComplexDataVector, 0>>& omega_cd,
         const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
             x_of_x_tilde) noexcept {
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
      db::get<Tags::GaugeC>(inverse_transform_box),
      db::get<Tags::GaugeD>(inverse_transform_box),
      db::get<Tags::GaugeOmega>(inverse_transform_box),
      db::get<Tags::CauchyAngularCoords>(inverse_transform_box));

  // subtract off the gauge adjustments from the gauge for the inverse
  // transformation box so that the remaining quantities are all in a consistent
  // gauge. This additional manipulation is not performed by the standard gauge
  // transform as the extra operations are not required for typical evaluation.
  db::mutate<Tags::EvolutionGaugeBoundaryValue<Tags::BondiU>>(
      make_not_null(&inverse_transform_box),
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
             evolution_gauge_boundary_u,
         const Scalar<SpinWeighted<ComplexDataVector, 1>>&
             evolution_gauge_full_u) noexcept {
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
      db::get<Tags::BondiU>(inverse_transform_box));

  db::mutate_apply<GaugeUpdateTimeDerivatives>(
      make_not_null(&forward_transform_box));
  db::mutate_apply<GaugeUpdateTimeDerivatives>(
      make_not_null(&inverse_transform_box));

  // check that the du_omega_cd are as expected.
  const auto& x_of_x_tilde =
      db::get<Tags::CauchyAngularCoords>(inverse_transform_box);
  const auto& forward_du_omega_cd =
      db::get<Tags::Du<Tags::GaugeOmega>>(forward_transform_box);
  const auto& forward_u_0 = db::get<Tags::BondiUAtScri>(forward_transform_box);
  const auto& forward_eth_omega_cd = db::get<Spectral::Swsh::Tags::Derivative<
      Tags::GaugeOmega, Spectral::Swsh::Tags::Eth>>(forward_transform_box);
  const auto& forward_omega_cd =
      get(db::get<Tags::GaugeOmega>(forward_transform_box));
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
      -get(db::get<Tags::Du<Tags::GaugeOmega>>(inverse_transform_box)) /
      get(db::get<Tags::GaugeOmega>(inverse_transform_box));

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

  const auto& inverse_c = db::get<Tags::GaugeC>(inverse_transform_box);
  const auto& inverse_d = db::get<Tags::GaugeD>(inverse_transform_box);

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

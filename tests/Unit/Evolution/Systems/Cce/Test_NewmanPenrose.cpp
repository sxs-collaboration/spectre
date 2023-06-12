// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/NewmanPenrose.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace {
void pypp_test_volume_weyl() {
  pypp::SetupLocalPythonEnvironment local_python_env{"Evolution/Systems/Cce/"};

  const size_t num_pts = 5;

  pypp::check_with_random_values<1>(&(VolumeWeyl<Tags::Psi0>::apply),
                                    "NewmanPenrose", {"psi0"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
}
}  // namespace

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
        1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
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

// This unit test is to validate the calculation of the Weyl scalar psi0 on the
// worldtube. The structure is in parallel with Test_GaugeTransformBoundaryData
// (most codes are copied from there). The test constructs a stationary Kerr
// spacetime in nontrivial time-dependent oscillating coordinates on both Cauchy
// and CCE grids. Then we compute psi0 on the worldtube. In principle, it should
// be consistent with 0.
template <typename Generator>
void compute_psi0_of_bh_on_wt(const gsl::not_null<Generator*> gen) {
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
                 Spectral::Swsh::Tags::Derivative<Tags::PartiallyFlatGaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::CauchyGaugeOmega,
                                                  Spectral::Swsh::Tags::Eth>>,
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>,
      Tags::characteristic_worldtube_boundary_tags<
          Tags::EvolutionGaugeBoundaryValue>>>;
  using coordinate_variables_tag = ::Tags::Variables<real_boundary_tags>;
  using spin_weighted_variables_tag =
      ::Tags::Variables<spin_weighted_boundary_tags>;
  using matching_variables_tag = ::Tags::Variables<
      tmpl::list<Tags::BoundaryValue<Tags::Psi0Match>,
                 Tags::BoundaryValue<Tags::Dlambda<Tags::Psi0Match>>>>;
  using volume_spin_weighted_variables_tag = ::Tags::Variables<
      tmpl::list<Tags::BondiJ, Tags::Dy<Tags::BondiJ>,
                 Tags::BondiJCauchyView, Tags::Dy<Tags::BondiJCauchyView>,
                 Tags::Dy<Tags::Dy<Tags::BondiJCauchyView>>, Tags::Psi0Match,
                 Tags::OneMinusY, Tags::Dy<Tags::Psi0Match>>>;

  auto box = db::create<db::AddSimpleTags<
      coordinate_variables_tag, spin_weighted_variables_tag,
      volume_spin_weighted_variables_tag, Tags::LMax, matching_variables_tag,
      Tags::NumberOfRadialPoints,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::SwshInterpolator<
          Tags::PartiallyFlatAngularCoords>>>(
      typename coordinate_variables_tag::type{number_of_angular_grid_points},
      typename spin_weighted_variables_tag::type{number_of_angular_grid_points},
      typename volume_spin_weighted_variables_tag::type{
          number_of_angular_grid_points * number_of_radial_grid_points},
      l_max,
      typename matching_variables_tag::type{number_of_angular_grid_points},
      number_of_radial_grid_points, Spectral::Swsh::SwshInterpolator{},
      Spectral::Swsh::SwshInterpolator{});

  // create analytic Cauchy data (a stationary Kerr BH)
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
      make_not_null(&box));

  // construct the coordinate transform quantities
  const double variation_amplitude_inertial = value_dist(*gen);
  db::mutate<Tags::CauchyCartesianCoords>(
      [&l_max](const gsl::not_null<tnsr::i<DataVector, 3>*>
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
              collocation_point.phi;
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
      make_not_null(&box));

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
      make_not_null(&box));

  // Update various auxiliary boundary variables in order to prepare the
  // boundary data for BondiJ. Operartions are done for both Cauchy coordiantes
  // and partially flat coordinates.
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      make_not_null(&box));
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::PartiallyFlatAngularCoords, Tags::PartiallyFlatCartesianCoords>>(
      make_not_null(&box));

  db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
      make_not_null(&box));
  db::mutate_apply<GaugeUpdateInterpolator<Tags::PartiallyFlatAngularCoords>>(
      make_not_null(&box));

  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      make_not_null(&box));
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::CauchyGaugeC, Tags::CauchyGaugeD, Tags::PartiallyFlatAngularCoords,
      Tags::PartiallyFlatCartesianCoords>>(make_not_null(&box));

  db::mutate_apply<
      GaugeUpdateOmega<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                       Tags::PartiallyFlatGaugeOmega>>(make_not_null(&box));
  db::mutate_apply<GaugeUpdateOmega<Tags::CauchyGaugeC, Tags::CauchyGaugeD,
                                    Tags::CauchyGaugeOmega>>(
      make_not_null(&box));

  // Now we need to transform the boundary data of BondiJ and BondiR from the
  // Cauchy grid to the partially flat grid.
  const auto perform_gauge_adjustment = [&box](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    INFO("computing tag : " << db::tag_name<tag>());
    db::mutate_apply<GaugeAdjustedBoundaryValue<tag>>(
        make_not_null(&box));
  };

  using gauge_adjustments =
      tmpl::list<Tags::BondiR, Tags::BondiJ, Tags::Dr<Tags::BondiJ>>;
  tmpl::for_each<gauge_adjustments>(perform_gauge_adjustment);

  // Now we construct the volume data of BondiJ (on a null slice) based on its
  // boundary data.
  db::mutate_apply<InverseCubicEvolutionGauge::mutate_tags,
                   InverseCubicEvolutionGauge::argument_tags>(
      InverseCubicEvolutionGauge{}, make_not_null(&box));

  // Then we compute psi0 on the worldtube
  db::mutate_apply<TransformBondiJToCauchyCoords>(make_not_null(&box));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJCauchyView>>>(
      make_not_null(&box));
  db::mutate_apply<
      PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJCauchyView>>>>(
      make_not_null(&box));
  db::mutate_apply<
      PrecomputeCceDependencies<Tags::BoundaryValue, Tags::OneMinusY>>(
      make_not_null(&box));
  db::mutate_apply<VolumeWeyl<Tags::Psi0Match>>(
      make_not_null(&box));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Psi0Match>>>(
      make_not_null(&box));
  db::mutate_apply<InnerBoundaryWeyl>(make_not_null(&box));

  // Finally, we expect the results should be consistent with 0.
  const auto& psi0_wt =
      db::get<Tags::BoundaryValue<Tags::Psi0Match>>(box);
  SpinWeighted<ComplexDataVector, 2> psi0_desired{number_of_angular_grid_points,
                                                  0.0};
  Approx interpolation_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(get(psi0_wt).data(), psi0_desired.data(),
                               interpolation_approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.NewmanPenrose", "[Unit][Cce]") {
  pypp_test_volume_weyl();
  MAKE_GENERATOR(gen);
  compute_psi0_of_bh_on_wt(make_not_null(&gen));
}
}  // namespace Cce

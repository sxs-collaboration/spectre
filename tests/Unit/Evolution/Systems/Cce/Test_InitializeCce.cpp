// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "Evolution/Systems/Cce/NewmanPenrose.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {

template <typename DbTags>
void test_initialize_j_inverse_cubic(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t l_max, const size_t number_of_radial_points) {
  db::mutate_apply<InitializeJ::InitializeJ<true>::mutate_tags,
                   InitializeJ::InitializeJ<true>::argument_tags>(
      InitializeJ::InverseCubic<true>{}, box_to_initialize);

  SpinWeighted<ComplexDataVector, 2> dy_j{
      number_of_radial_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  SpinWeighted<ComplexDataVector, 2> dy_dy_j{
      number_of_radial_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  logical_partial_directional_derivative_of_complex(
      make_not_null(&dy_j.data()),
      get(db::get<Tags::BondiJ>(*box_to_initialize)).data(),
      Mesh<3>{{{Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max),
                Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max),
                number_of_radial_points}},
              Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto},
      2);
  logical_partial_directional_derivative_of_complex(
      make_not_null(&dy_dy_j.data()), dy_j.data(),
      Mesh<3>{{{Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max),
                Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max),
                number_of_radial_points}},
              Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto},
      2);

  // The goal for this initial data is that it should:
  // - match the value of J and its first derivative on the boundary
  // - have vanishing value and second derivative at scri+
  ComplexDataVector mutable_j_copy =
      get(db::get<Tags::BondiJ>(*box_to_initialize)).data();
  const auto boundary_slice_j = ComplexDataVector{
      mutable_j_copy.data(),
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  const auto boundary_slice_dy_j = ComplexDataVector{
      dy_j.data().data(),
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  const auto scri_slice_j = ComplexDataVector{
      mutable_j_copy.data() +
          (number_of_radial_points - 1) *
              Spectral::Swsh::number_of_swsh_collocation_points(l_max),
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  const auto scri_slice_dy_dy_j = ComplexDataVector{
      dy_dy_j.data().data() +
          (number_of_radial_points - 1) *
              Spectral::Swsh::number_of_swsh_collocation_points(l_max),
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  Approx cce_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(
      get(db::get<Tags::BoundaryValue<Tags::BondiJ>>(*box_to_initialize))
          .data(),
      boundary_slice_j, cce_approx);
  const auto boundary_slice_dr_j =
      (2.0 / get(db::get<Tags::BoundaryValue<Tags::BondiR>>(*box_to_initialize))
                 .data()) *
      boundary_slice_dy_j;
  CHECK_ITERABLE_CUSTOM_APPROX(
      boundary_slice_dr_j,
      get(db::get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
              *box_to_initialize))
          .data(),
      cce_approx);
  const ComplexDataVector scri_plus_zeroes{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0};
  CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_j, scri_plus_zeroes, cce_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_dy_dy_j, scri_plus_zeroes,
                               cce_approx);
}
template <typename DbTags>
void test_initialize_j_zero_nonsmooth(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t /*l_max*/, const size_t /*number_of_radial_points*/) {
  // The iterative procedure can reach error levels better than 1.0e-8, but it
  // is difficult to do so reliably and quickly for randomly generated data.
  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      InitializeJ::ZeroNonSmooth{1.0e-8, 400}, box_to_initialize);

  // note we want to copy here to compare against the next version of the
  // computation
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto initialized_j = db::get<Tags::BondiJ>(*box_to_initialize);

  const auto initializer = InitializeJ::ZeroNonSmooth{1.0e-8, 400};
  const auto serialized_and_deserialized_initializer =
      serialize_and_deserialize(initializer);

  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      serialized_and_deserialized_initializer, box_to_initialize);
  const auto& initialized_j_from_serialized_and_deserialized =
      db::get<Tags::BondiJ>(*box_to_initialize);

  CHECK_ITERABLE_APPROX(
      get(initialized_j).data(),
      get(initialized_j_from_serialized_and_deserialized).data());

  // generate the extra gauge quantities and verify that the boundary value for
  // J is indeed within the tolerance.
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
      box_to_initialize);
  db::mutate_apply<
      GaugeUpdateOmega<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                       Tags::PartiallyFlatGaugeOmega>>(box_to_initialize);

  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiJ>>(box_to_initialize);

  const auto& gauge_adjusted_boundary_j =
      db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>>(
          *box_to_initialize);
  for (auto val : get(gauge_adjusted_boundary_j).data()) {
    CHECK(real(val) < 1.0e-8);
    CHECK(imag(val) < 1.0e-8);
  }

  for (auto val : get(initialized_j).data()) {
    CHECK(real(val) < 1.0e-8);
    CHECK(imag(val) < 1.0e-8);
  }
}

// [[OutputRegex, Initial data iterative angular solve]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.InitializeJ.ZeroNonSmoothError",
    "[Unit][Cce]") {
  ERROR_TEST();
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<size_t> sdist{5, 8};
  const size_t l_max = sdist(generator);
  const size_t number_of_radial_points = sdist(generator);

  using boundary_variables_tag =
      ::Tags::Variables<InitializeJ::InverseCubic<false>::boundary_tags>;
  using pre_swsh_derivatives_variables_tag =
      ::Tags::Variables<tmpl::list<Tags::BondiJ>>;
  using tensor_variables_tag = ::Tags::Variables<
      tmpl::list<Tags::CauchyCartesianCoords, Tags::CauchyAngularCoords>>;

  const size_t number_of_boundary_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_volume_points =
      number_of_boundary_points * number_of_radial_points;
  auto box_to_initialize = db::create<db::AddSimpleTags<
      boundary_variables_tag, pre_swsh_derivatives_variables_tag,
      tensor_variables_tag, Tags::LMax, Tags::NumberOfRadialPoints,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>>(
      typename boundary_variables_tag::type{number_of_boundary_points},
      typename pre_swsh_derivatives_variables_tag::type{
          number_of_volume_points},
      typename tensor_variables_tag::type{number_of_boundary_points}, l_max,
      number_of_radial_points, Spectral::Swsh::SwshInterpolator{});

  // generate some random values for the boundary data. Mode magnitudes are
  // roughly representative of typical strains seen in simulations, and are of a
  // scale that can be fully solved by the iterative procedure used in the more
  // elaborate initial data generators.
  UniformCustomDistribution<double> dist(1.0e-5, 1.0e-4);
  db::mutate<Tags::BoundaryValue<Tags::BondiR>,
             Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
             Tags::BoundaryValue<Tags::BondiJ>>(
      make_not_null(&box_to_initialize),
      [&generator, &dist, &l_max](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_r,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              boundary_dr_j,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              boundary_j) {
        SpinWeighted<ComplexModalVector, 2> generated_modes{
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
        Spectral::Swsh::TestHelpers::generate_swsh_modes<2>(
            make_not_null(&generated_modes.data()), make_not_null(&generator),
            make_not_null(&dist), 1, l_max);

        get(*boundary_j) =
            Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_modes);
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*boundary_j)), l_max, l_max / 2);

        Spectral::Swsh::TestHelpers::generate_swsh_modes<2>(
            make_not_null(&generated_modes.data()), make_not_null(&generator),
            make_not_null(&dist), 1, l_max);
        get(*boundary_dr_j) =
            Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_modes) /
            10.0;

        SpinWeighted<ComplexModalVector, 0> generated_r_modes{
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
        Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
            make_not_null(&generated_modes.data()), make_not_null(&generator),
            make_not_null(&dist), 1, l_max);

        get(*boundary_r) = Spectral::Swsh::inverse_swsh_transform(
                               l_max, 1, generated_r_modes) +
                           10.0;
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*boundary_r)), l_max, l_max / 2);
      });
  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      InitializeJ::ZeroNonSmooth{1.0e-12, 1, true},
      make_not_null(&box_to_initialize));
  ERROR("Failed to trigger ERROR in an error test");
}

template <typename DbTags>
void test_initialize_j_no_radiation(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t l_max, const size_t /*number_of_radial_points*/) {
  // The iterative procedure can reach error levels better than 1.0e-8, but it
  // is difficult to do so reliably and quickly for randomly generated data.
  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      InitializeJ::NoIncomingRadiation{1.0e-8, 400}, box_to_initialize);

  // note we want to copy here to compare against the next version of the
  // computation
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto initialized_j = db::get<Tags::BondiJ>(*box_to_initialize);

  const auto initializer = InitializeJ::NoIncomingRadiation{1.0e-8, 400};
  const auto serialized_and_deserialized_initializer =
      serialize_and_deserialize(initializer);

  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      serialized_and_deserialized_initializer, box_to_initialize);
  const auto& initialized_j_from_serialized_and_deserialized =
      db::get<Tags::BondiJ>(*box_to_initialize);

  CHECK_ITERABLE_APPROX(
      get(initialized_j).data(),
      get(initialized_j_from_serialized_and_deserialized).data());

  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      box_to_initialize);
  db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
      box_to_initialize);
  db::mutate_apply<
      GaugeUpdateOmega<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                       Tags::PartiallyFlatGaugeOmega>>(box_to_initialize);

  db::mutate_apply<PrecomputeCceDependencies<Tags::EvolutionGaugeBoundaryValue,
                                             Tags::OneMinusY>>(
      box_to_initialize);
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiJ>>(box_to_initialize);

  // check that the gauge-transformed boundary data matches up.
  const auto& boundary_gauge_j =
      db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>>(
          *box_to_initialize);
  for (size_t i = 0;
       i < Spectral::Swsh::number_of_swsh_collocation_points(l_max); ++i) {
    CHECK(approx(real(get(initialized_j).data()[i])) ==
          real(get(boundary_gauge_j).data()[i]));
    CHECK(approx(imag(get(initialized_j).data()[i])) ==
          imag(get(boundary_gauge_j).data()[i]));
  }

  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiR>>(box_to_initialize);

  db::mutate_apply<PrecomputeCceDependencies<Tags::EvolutionGaugeBoundaryValue,
                                             Tags::BondiR>>(box_to_initialize);
  db::mutate_apply<PrecomputeCceDependencies<Tags::EvolutionGaugeBoundaryValue,
                                             Tags::BondiK>>(box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
      box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJ>>>>(
      box_to_initialize);

  db::mutate_apply<VolumeWeyl<Tags::Psi0>>(box_to_initialize);

  Approx cce_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);
  // check that the psi_0 condition holds to acceptable precision -- note the
  // result of this involves multiple numerical derivatives, so needs to be
  // slightly loose.
  for (auto val : get(db::get<Tags::Psi0>(*box_to_initialize)).data()) {
    CHECK(cce_approx(real(val)) == 0.0);
    CHECK(cce_approx(imag(val)) == 0.0);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.InitializeJ", "[Unit][Cce]") {
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<size_t> sdist{5, 8};
  const size_t l_max = sdist(generator);
  const size_t number_of_radial_points = sdist(generator);

  using boundary_variables_tag = ::Tags::Variables<tmpl::push_back<
      InitializeJ::InverseCubic<true>::boundary_tags, Tags::PartiallyFlatGaugeC,
      Tags::PartiallyFlatGaugeD, Tags::PartiallyFlatGaugeOmega,
      Spectral::Swsh::Tags::Derivative<Tags::PartiallyFlatGaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>>>;
  using pre_swsh_derivatives_variables_tag = ::Tags::Variables<tmpl::list<
      Tags::BondiJ, Tags::Dy<Tags::BondiJ>, Tags::Dy<Tags::Dy<Tags::BondiJ>>,
      Tags::BondiK, Tags::BondiR, Tags::OneMinusY, Tags::Psi0>>;
  using tensor_variables_tag = ::Tags::Variables<tmpl::list<
      Tags::CauchyCartesianCoords, Tags::CauchyAngularCoords,
      Tags::PartiallyFlatCartesianCoords, Tags::PartiallyFlatAngularCoords>>;

  const size_t number_of_boundary_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_volume_points =
      number_of_boundary_points * number_of_radial_points;
  auto box_to_initialize = db::create<db::AddSimpleTags<
      boundary_variables_tag, pre_swsh_derivatives_variables_tag,
      tensor_variables_tag, Tags::LMax, Tags::NumberOfRadialPoints,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>>(
      typename boundary_variables_tag::type{number_of_boundary_points},
      typename pre_swsh_derivatives_variables_tag::type{
          number_of_volume_points},
      typename tensor_variables_tag::type{number_of_boundary_points}, l_max,
      number_of_radial_points, Spectral::Swsh::SwshInterpolator{});

  // generate some random values for the boundary data. Mode magnitudes are
  // roughly representative of typical strains seen in simulations, and are of a
  // scale that can be fully solved by the iterative procedure used in the more
  // elaborate initial data generators.
  UniformCustomDistribution<double> dist(1.0e-5, 1.0e-4);
  db::mutate<Tags::BoundaryValue<Tags::BondiR>,
             Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
             Tags::BoundaryValue<Tags::BondiJ>>(
      make_not_null(&box_to_initialize),
      [&generator, &dist, &l_max](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_r,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              boundary_dr_j,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              boundary_j) {
        SpinWeighted<ComplexModalVector, 2> generated_modes{
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
        Spectral::Swsh::TestHelpers::generate_swsh_modes<2>(
            make_not_null(&generated_modes.data()), make_not_null(&generator),
            make_not_null(&dist), 1, l_max);

        get(*boundary_j) =
            Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_modes);
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*boundary_j)), l_max, l_max / 2);

        SpinWeighted<ComplexModalVector, 0> generated_r_modes{
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
        Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
            make_not_null(&generated_modes.data()), make_not_null(&generator),
            make_not_null(&dist), 1, l_max);

        get(*boundary_r) = Spectral::Swsh::inverse_swsh_transform(
                               l_max, 1, generated_r_modes) +
                           10.0;
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*boundary_r)), l_max, l_max / 2);

        get(*boundary_dr_j) = -get(*boundary_j) / get(*boundary_r);
      });
  SECTION("Check inverse cubic initial data generator") {
    test_initialize_j_inverse_cubic(make_not_null(&box_to_initialize), l_max,
                                    number_of_radial_points);
  }
  SECTION("Check zero nonsmooth initial data generator") {
    test_initialize_j_zero_nonsmooth(make_not_null(&box_to_initialize), l_max,
                                     number_of_radial_points);
  }
  SECTION("Check no incoming radiation initial data generator") {
    test_initialize_j_no_radiation(make_not_null(&box_to_initialize), l_max,
                                   number_of_radial_points);
  }
}
}  // namespace Cce

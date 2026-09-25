// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/Initialize/CauchySecondOrder.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/NewmanPenrose.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
// Required when registering all SpanInterpolator subclasses with Charm++.
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshFiltering.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace Cce {

namespace {

template <template <typename> typename BoundaryTag, typename DbTags>
void check_boundary_and_asymptotic_j(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t number_of_radial_points, const size_t l_max) {
  // The goal for this initial data is that it should:
  // - match the value of J and its first derivative on the boundary
  // - have vanishing value and second derivative at scri+
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const SpinWeighted<ComplexDataVector, 2> boundary_slice_dy_j;
  make_const_view(make_not_null(&boundary_slice_dy_j),
                  get(db::get<Tags::Dy<Tags::BondiJ>>(*box_to_initialize)), 0,
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 2> boundary_slice_j;
  const SpinWeighted<ComplexDataVector, 2> scri_slice_j;
  make_const_view(make_not_null(&boundary_slice_j),
                  get(db::get<Tags::BondiJ>(*box_to_initialize)), 0,
                  number_of_angular_points);
  make_const_view(make_not_null(&scri_slice_j),
                  get(db::get<Tags::BondiJ>(*box_to_initialize)),
                  number_of_angular_points * (number_of_radial_points - 1),
                  number_of_angular_points);

  const SpinWeighted<ComplexDataVector, 2> scri_slice_dy_dy_j;
  make_const_view(
      make_not_null(&scri_slice_dy_dy_j),
      get(db::get<Tags::Dy<Tags::Dy<Tags::BondiJ>>>(*box_to_initialize)),
      number_of_angular_points * (number_of_radial_points - 1),
      number_of_angular_points);

  Approx cce_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(
      get(db::get<BoundaryTag<Tags::BondiJ>>(*box_to_initialize)).data(),
      boundary_slice_j, cce_approx);
  const auto boundary_slice_dr_j =
      (2.0 /
       get(db::get<BoundaryTag<Tags::BondiR>>(*box_to_initialize)).data()) *
      boundary_slice_dy_j.data();
  CHECK_ITERABLE_CUSTOM_APPROX(
      boundary_slice_dr_j,
      get(db::get<BoundaryTag<Tags::Dr<Tags::BondiJ>>>(*box_to_initialize))
          .data(),
      cce_approx);
  const ComplexDataVector scri_plus_zeroes{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0};
  CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_j, scri_plus_zeroes, cce_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_dy_dy_j.data(), scri_plus_zeroes,
                               cce_approx);
}

// Fill a worldtube boundary value with filtered random spin-weighted data of
// the requested spin weight. Used to populate the additional boundary
// quantities consumed by the `CauchySecondOrder` generator.
template <int Spin, typename Generator, typename Distribution>
void assign_random_swsh_boundary_value(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, Spin>>*> field,
    const gsl::not_null<Generator*> generator,
    const gsl::not_null<Distribution*> distribution, const size_t l_max) {
  SpinWeighted<ComplexModalVector, Spin> generated_modes{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  Spectral::Swsh::TestHelpers::generate_swsh_modes<Spin>(
      make_not_null(&generated_modes.data()), generator, distribution, 1,
      l_max);
  get(*field) =
      Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_modes);
  Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(*field)),
                                                l_max, l_max / 2);
}

template <typename DbTags>
void test_initialize_j_inverse_cubic(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t l_max, const size_t number_of_radial_points) {
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::InitializeJ<true>::mutate_tags,
                   InitializeJ::InitializeJ<true>::argument_tags>(
      InitializeJ::InverseCubic<true>{}, box_to_initialize,
      make_not_null(&node_lock));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
      box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJ>>>>(
      box_to_initialize);
  check_boundary_and_asymptotic_j<Tags::BoundaryValue>(
      box_to_initialize, number_of_radial_points, l_max);
}

template <typename DbTags>
void test_initialize_j_zero_nonsmooth(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t /*l_max*/, const size_t /*number_of_radial_points*/) {
  // The iterative procedure can reach error levels better than 1.0e-8, but it
  // is difficult to do so reliably and quickly for randomly generated data.
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      InitializeJ::ZeroNonSmooth{1.0e-8, 400}, box_to_initialize,
      make_not_null(&node_lock));

  // note we want to copy here to compare against the next version of the
  // computation
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto initialized_j = db::get<Tags::BondiJ>(*box_to_initialize);

  const auto initializer = InitializeJ::ZeroNonSmooth{1.0e-8, 400};
  const auto serialized_and_deserialized_initializer =
      serialize_and_deserialize(initializer);

  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      serialized_and_deserialized_initializer, box_to_initialize,
      make_not_null(&node_lock));
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

template <typename DbTags>
void test_zero_non_smooth_error(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t /*l_max*/, const size_t /*number_of_radial_points*/) {
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      InitializeJ::ZeroNonSmooth{1.0e-12, 1, true}, box_to_initialize,
      make_not_null(&node_lock));
}

template <typename DbTags>
void test_initialize_j_no_radiation(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t l_max, const size_t /*number_of_radial_points*/) {
  // The iterative procedure can reach error levels better than 1.0e-8, but it
  // is difficult to do so reliably and quickly for randomly generated data.
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      InitializeJ::NoIncomingRadiation{1.0e-8, 400}, box_to_initialize,
      make_not_null(&node_lock));

  // note we want to copy here to compare against the next version of the
  // computation
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto initialized_j = db::get<Tags::BondiJ>(*box_to_initialize);

  const auto initializer = InitializeJ::NoIncomingRadiation{1.0e-8, 400};
  const auto serialized_and_deserialized_initializer =
      serialize_and_deserialize(initializer);

  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      serialized_and_deserialized_initializer, box_to_initialize,
      make_not_null(&node_lock));
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

// The interpolator `CauchySecondOrder` hands to the worldtube data manager for
// the Du(Dr(J)) boundary value. These tests supply that boundary value directly
// rather than through a manager, so it is only carried along and serialized.
std::unique_ptr<intrp::SpanInterpolator> make_du_dr_j_interpolator() {
  return std::make_unique<intrp::BarycentricRationalSpanInterpolator>(2_st,
                                                                      2_st);
}

// The interpolator the generator hands to the worldtube data manager has to
// survive the trip into the GlobalCache and back out of a checkpoint.
void test_cauchy_second_order_interpolator_round_trip() {
  const InitializeJ::CauchySecondOrder with_interpolator{
      1.0e-10, 400, true,    1.0e-1,
      1.0e-14, 10,  1.0e-12, make_du_dr_j_interpolator()};
  REQUIRE(with_interpolator.du_dr_j_interpolator() != nullptr);
  CHECK(with_interpolator.du_dr_j_interpolator()
            ->required_number_of_points_before_and_after() == 2);

  const auto clone = with_interpolator.get_clone();
  REQUIRE(clone->du_dr_j_interpolator() != nullptr);
  CHECK(clone->du_dr_j_interpolator()
            ->required_number_of_points_before_and_after() == 2);

  const auto round_tripped = serialize_and_deserialize(with_interpolator);
  REQUIRE(round_tripped.du_dr_j_interpolator() != nullptr);
  CHECK(round_tripped.du_dr_j_interpolator()
            ->required_number_of_points_before_and_after() == 2);

  // A generator that asks for nothing leaves the manager on `H5Interpolator`.
  const InitializeJ::CauchySecondOrder without_interpolator{
      1.0e-10, 400, true, 1.0e-1, 1.0e-14, 10, 1.0e-12, nullptr};
  CHECK(without_interpolator.du_dr_j_interpolator() == nullptr);
  CHECK(without_interpolator.get_clone()->du_dr_j_interpolator() == nullptr);
  CHECK(InitializeJ::InverseCubic<false>{}.du_dr_j_interpolator() == nullptr);
}

// The partially flat gauge condition on the second asymptotic coefficient
// determines the second radial expansion coefficient J^(2) of the Cauchy-gauge
// ansatz as the root of a contraction mapping. Exercise that solve directly,
// away from the worldtube machinery that supplies its inputs in the generator.
void test_cauchy_second_order_j2_fixed_point() {
  namespace second_order = InitializeJ::CauchySecondOrder_detail;
  const size_t number_of_points = 8;
  // Strain-scale stand-ins for the x-independent parts of J^(0) and J^(1).
  ComplexDataVector j0_at_zero{number_of_points};
  ComplexDataVector j1_at_zero{number_of_points};
  for (size_t i = 0; i < number_of_points; ++i) {
    const auto index = static_cast<double>(i);
    j0_at_zero[i] = std::complex<double>(1.0e-5 * (1.0 + (0.1 * index)),
                                         -2.0e-5 * (1.0 - (0.05 * index)));
    j1_at_zero[i] = std::complex<double>(-3.0e-4 * (1.0 - (0.02 * index)),
                                         1.5e-4 * (1.0 + (0.03 * index)));
  }

  // A zero iteration budget performs no passes and leaves J^(2) = 0.
  ComplexDataVector j2{number_of_points, 1.0};
  double step = 0.0;
  CHECK(second_order::solve_asymptotic_j2(make_not_null(&j2),
                                          make_not_null(&step), j0_at_zero,
                                          j1_at_zero, 1.0e-16, 0) == 0);
  CHECK(max(abs(j2)) == 0.0);
  CHECK(step == std::numeric_limits<double>::infinity());

  // Given a budget the iteration converges in a couple of passes, ...
  const size_t iterations = second_order::solve_asymptotic_j2(
      make_not_null(&j2), make_not_null(&step), j0_at_zero, j1_at_zero, 1.0e-16,
      50);
  CHECK(iterations <= 3);
  CHECK(step < 1.0e-16);

  // ... to a root of the size the constraint implies, of order
  // |J^(0)| |J^(1)|^2, ...
  CHECK(max(abs(j2)) < 1.0e-11);
  CHECK(max(abs(j2)) > 1.0e-14);

  // ... which satisfies the constraint it was built from.
  const ComplexDataVector j0 = j0_at_zero + second_order::j0_x_coefficient * j2;
  const ComplexDataVector j1 = j1_at_zero + second_order::j1_x_coefficient * j2;
  const DataVector k1 = real(j0 * conj(j1)) / sqrt(1.0 + real(j0 * conj(j0)));
  const ComplexDataVector constraint_residual =
      j2 - 0.5 * j0 * (real(j1 * conj(j1)) - k1 * k1);
  CHECK(max(abs(constraint_residual)) < 1.0e-16);

  // A budget too small for the requested tolerance is reported back to the
  // caller through `final_step` rather than silently accepted.
  CHECK(second_order::solve_asymptotic_j2(make_not_null(&j2),
                                          make_not_null(&step), j0_at_zero,
                                          j1_at_zero, 1.0e-16, 1) == 1);
  CHECK(step >= 1.0e-16);

  // A non-finite input stops the iteration after the pass that produces it and
  // is reported through a NaN `final_step`, however large the budget. The NaN
  // is put in the last element because a `max` reduction can drop a NaN that
  // is not the first element. This path only matters where floating point
  // exceptions are not trapped (e.g. on aarch64), so disable trapping to
  // exercise it. The checks come after the scope so that trapping is restored
  // before anything else touches the NaN.
  size_t corrupted_iterations = 0;
  bool step_is_nan = false;
  {
    const ScopedFpeState disable_fpes(false);
    ComplexDataVector corrupted_j0_at_zero = j0_at_zero;
    corrupted_j0_at_zero[number_of_points - 1] =
        std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 0.0);
    corrupted_iterations = second_order::solve_asymptotic_j2(
        make_not_null(&j2), make_not_null(&step), corrupted_j0_at_zero,
        j1_at_zero, 1.0e-16, 50);
    step_is_nan = std::isnan(step);
    // `j2` now holds NaN; reset it so later uses cannot trap.
    j2 = ComplexDataVector{number_of_points, 0.0};
    step = 0.0;
  }
  CHECK(corrupted_iterations == 1);
  CHECK(step_is_nan);
}

// The x-dependence of the other coefficients of the Cauchy-gauge ansatz is
// what keeps the worldtube match intact for any x = J^(2). Check the match
// directly, with O(1) values so that any error in that dependence is visible.
void test_cauchy_second_order_radial_ansatz_coefficients() {
  namespace second_order = InitializeJ::CauchySecondOrder_detail;
  const size_t number_of_points = 8;
  ComplexDataVector boundary_j{number_of_points};
  ComplexDataVector boundary_dr_j{number_of_points};
  ComplexDataVector boundary_dy_dy_j{number_of_points};
  ComplexDataVector boundary_r{number_of_points};
  ComplexDataVector j2{number_of_points};
  for (size_t i = 0; i < number_of_points; ++i) {
    const auto index = static_cast<double>(i);
    boundary_j[i] =
        std::complex<double>(0.3 - (0.02 * index), 0.1 + (0.03 * index));
    boundary_dr_j[i] =
        std::complex<double>(-1.5e-3 * (1.0 + (0.1 * index)), 0.5e-3);
    boundary_dy_dy_j[i] = std::complex<double>((0.05 * index) - 0.2, -0.1);
    boundary_r[i] = std::complex<double>(200.0 + (5.0 * index), 0.0);
    j2[i] = std::complex<double>(0.07 - (0.01 * index), 0.04);
  }
  ComplexDataVector j0{number_of_points};
  ComplexDataVector j1{number_of_points};
  ComplexDataVector j3{number_of_points};
  second_order::radial_ansatz_coefficients(
      make_not_null(&j0), make_not_null(&j1), make_not_null(&j3), j2,
      boundary_j, boundary_dr_j, boundary_dy_dy_j, boundary_r);

  // At the worldtube 1 - y = 2, and d/dy = -d/d(1 - y).
  const ComplexDataVector worldtube_j = j0 + 2.0 * j1 + 4.0 * j2 + 8.0 * j3;
  const ComplexDataVector worldtube_dy_j = -(j1 + 4.0 * j2 + 12.0 * j3);
  const ComplexDataVector worldtube_dy_dy_j = 2.0 * j2 + 12.0 * j3;
  // At the worldtube radius R, dy/dr = 2 / R.
  const ComplexDataVector expected_dy_j = 0.5 * boundary_r * boundary_dr_j;
  CHECK_ITERABLE_APPROX(worldtube_j, boundary_j);
  CHECK_ITERABLE_APPROX(worldtube_dy_j, expected_dy_j);
  CHECK_ITERABLE_APPROX(worldtube_dy_dy_j, boundary_dy_dy_j);
}

// The largest violation over the angular grid of the partially flat condition
// on the second asymptotic coefficient, J2 = 0.5 * Dy^2 J at scri+, for the J
// currently in the box.
template <typename DbTags>
double max_scri_j2(const gsl::not_null<db::DataBox<DbTags>*> box,
                   const size_t l_max, const size_t number_of_radial_points) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(box);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJ>>>>(box);
  const SpinWeighted<ComplexDataVector, 2> scri_dy_dy_j;
  make_const_view(make_not_null(&scri_dy_dy_j),
                  get(db::get<Tags::Dy<Tags::Dy<Tags::BondiJ>>>(*box)),
                  number_of_angular_points * (number_of_radial_points - 1),
                  number_of_angular_points);
  return 0.5 * max(abs(scri_dy_dy_j.data()));
}

template <typename DbTags>
void test_cauchy_second_order_j2_convergence_error(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize) {
  // A single fixed-point pass moves J^(2) from zero to a value far larger than
  // 1e-16, so it cannot meet that tolerance. The solve must report the failure.
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      InitializeJ::CauchySecondOrder{1.0e-10, 1000, true, 1.0e-1, 1.0e-16, 1,
                                     1.0e-12, make_du_dr_j_interpolator()},
      box_to_initialize, make_not_null(&node_lock));
}

template <typename DbTags>
void test_cauchy_second_order_j2_threshold_error(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize) {
  // The J^(2) solve leaves a small discretization residual in J2 after the
  // gauge transformation, so an unachievably small `MaxPartiallyFlatJ2` trips
  // the check. Allow enough angular iterations (as in the successful case) for
  // the angular solve to converge first.
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      InitializeJ::CauchySecondOrder{1.0e-10, 1000, true, 1.0e-1, 1.0e-14, 10,
                                     1.0e-30, make_du_dr_j_interpolator()},
      box_to_initialize, make_not_null(&node_lock));
}

template <typename DbTags>
void test_cauchy_second_order_j0_convergence_error(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize) {
  // Ten angular iterations cannot reach this tolerance for these data.
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      InitializeJ::CauchySecondOrder{1.0e-14, 10, true, 1.0e-1, 1.0e-14, 10,
                                     1.0e-12, make_du_dr_j_interpolator()},
      box_to_initialize, make_not_null(&node_lock));
}

template <typename DbTags>
void test_initialize_j_cauchy_second_order(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const size_t l_max, const size_t number_of_radial_points) {
  auto node_lock = Parallel::NodeLock{};
  // Without the J^(2) solve the constructed J keeps J^(2) = 0 in the Cauchy
  // gauge, which violates the partially flat condition on the second
  // asymptotic coefficient after the gauge transformation. Record that
  // violation so the solve below can be checked to remove it.
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      InitializeJ::CauchySecondOrder{1.0e-10, 1000, true, 1.0e-1, 1.0e-14, 0,
                                     1.0e-12, make_du_dr_j_interpolator()},
      box_to_initialize, make_not_null(&node_lock));
  const double scri_j2_without_solve =
      max_scri_j2(box_to_initialize, l_max, number_of_radial_points);

  // The angular coordinates are adapted iteratively (as in NoIncomingRadiation
  // and ConformalFactor). For randomly generated data the linearized solve
  // occasionally needs more than a few hundred iterations to reach 1e-10, so
  // we allow up to 1000 iterations to reliably converge.
  const auto initializer = InitializeJ::CauchySecondOrder{
      1.0e-10, 1000, true,    1.0e-1,
      1.0e-14, 10,   1.0e-12, make_du_dr_j_interpolator()};
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      initializer, box_to_initialize, make_not_null(&node_lock));

  // note we want to copy here to compare against the next version of the
  // computation
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto initialized_j = db::get<Tags::BondiJ>(*box_to_initialize);
  const auto serialized_and_deserialized_initializer =
      serialize_and_deserialize(initializer);
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      serialized_and_deserialized_initializer, box_to_initialize,
      make_not_null(&node_lock));
  CHECK_ITERABLE_APPROX(get(initialized_j).data(),
                        get(db::get<Tags::BondiJ>(*box_to_initialize)).data());

  // generate the gauge quantities so the boundary data can be compared in the
  // evolution gauge.
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

  // The polynomial construction matches J and its first radial derivative at
  // the worldtube, so the volume J on the boundary should equal the
  // gauge-transformed boundary value of J.
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const auto& boundary_gauge_j =
      db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>>(
          *box_to_initialize);
  for (size_t i = 0; i < number_of_angular_points; ++i) {
    CHECK(approx(real(get(initialized_j).data()[i])) ==
          real(get(boundary_gauge_j).data()[i]));
    CHECK(approx(imag(get(initialized_j).data()[i])) ==
          imag(get(boundary_gauge_j).data()[i]));
  }

  // Check both partially flat constraints on the final initial data: the
  // angular solve drives J0 = J at scri+ below its tolerance, and the J^(2)
  // solve removes most of the J2 = 0.5 * Dy^2 J violation left without it.
  // The J^(2) constraint is exact in the Cauchy gauge, but it is nonlinear, so
  // interpolating J to the adapted angular coordinates at this small l_max
  // leaves a residual of a few to about fifteen percent of the violation
  // (checked over many random seeds). Hence the relative comparison rather
  // than an absolute tolerance.
  const SpinWeighted<ComplexDataVector, 2> scri_j;
  make_const_view(make_not_null(&scri_j), get(initialized_j),
                  number_of_angular_points * (number_of_radial_points - 1),
                  number_of_angular_points);
  CHECK(max(abs(scri_j.data())) < 1.0e-10);
  const double scri_j2_with_solve =
      max_scri_j2(box_to_initialize, l_max, number_of_radial_points);
  CAPTURE(scri_j2_without_solve);
  CAPTURE(scri_j2_with_solve);
  CHECK(scri_j2_with_solve < scri_j2_without_solve / 3.0);
}

template <typename DbTags>
void test_cauchy_second_order_cauchy_j0_threshold(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize) {
  // Before the angular solve the constructed J is generically nonzero at scri+
  // at the scale of the strain, so an unachievably small `MaxAngularSolveError`
  // trips the pre-solve guard even on uncorrupted worldtube data. This checks
  // that the threshold is honored from the option rather than hard-coded.
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      InitializeJ::CauchySecondOrder{1.0e-10, 400, true, 1.0e-14, 1.0e-14, 10,
                                     1.0e-12, make_du_dr_j_interpolator()},
      box_to_initialize, make_not_null(&node_lock));
}

template <typename DbTags>
void test_cauchy_second_order_asymptotic_j_error(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize) {
  // Inflate the worldtube J without the compensating radial derivative so the
  // asymptotic value of the constructed Cauchy-coordinate J is far too large
  // for the angular solve to eliminate, tripping the pre-solve guard. This
  // corrupts the boundary data, so it must run after the other checks.
  db::mutate<Tags::BoundaryValue<Tags::BondiJ>>(
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
             boundary_j) { get(*boundary_j).data() += 1.0; },
      box_to_initialize);
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::CauchySecondOrder::return_tags,
                   InitializeJ::CauchySecondOrder::argument_tags>(
      InitializeJ::CauchySecondOrder{1.0e-10, 400, true, 1.0e-1, 1.0e-14, 10,
                                     1.0e-12, make_du_dr_j_interpolator()},
      box_to_initialize, make_not_null(&node_lock));
}

template <typename DbTags>
void test_initialize_j_conformal_factor(
    const gsl::not_null<db::DataBox<DbTags>*> box_to_initialize,
    const bool optimize_l_0_mode, const bool use_beta_integral_estimate,
    const bool use_input_modes, const bool read_modes_from_file,
    const ::Cce::InitializeJ::ConformalFactorIterationHeuristic
        iteration_heuristic,
    const size_t l_max, const size_t number_of_radial_points) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  CAPTURE(optimize_l_0_mode);
  CAPTURE(use_beta_integral_estimate);
  CAPTURE(use_input_modes);
  CAPTURE(read_modes_from_file);
  auto node_lock = Parallel::NodeLock{};
  InitializeJ::ConformalFactor initialize_j_conformal_factor;
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<double> dist(1.0e-4, 1.0e-3);
  const std::string filename = "ConformalFactorInputModes.h5";

  std::vector<double> input_mode_data(2 * square(l_max + 1));
  for (size_t i = 0; i < 8; ++i) {
    input_mode_data[i] = 0.0;
  }
  for (size_t i = 8; i < input_mode_data.size(); ++i) {
    // exponentially decay higher l-modes
    input_mode_data[i] = dist(generator) * exp(-1.0 * sqrt(i / 2));
  }
  std::vector<std::complex<double>> input_modes(square(l_max + 1));
  for (size_t i = 0; i < input_modes.size(); ++i) {
    input_modes[i] = std::complex<double>(input_mode_data[i * 2],
                                          input_mode_data[i * 2 + 1]);
  }
  if (use_input_modes) {
    if (read_modes_from_file) {
      if (file_system::check_if_file_exists(filename)) {
        file_system::rm(filename, true);
      }
      h5::H5File<h5::AccessType::ReadWrite> input_h5_modes{filename};

      std::vector<std::string> file_legend{};
      for (int l = 0; l <= static_cast<int>(l_max); ++l) {
        for (int m = -l; m <= l; ++m) {
          file_legend.push_back("Real Y_" + std::to_string(l) + "," +
                                std::to_string(m));
          file_legend.push_back("Imag Y_" + std::to_string(l) + "," +
                                std::to_string(m));
        }
      }
      auto& dataset =
          input_h5_modes.try_insert<h5::Dat>("/InitialJ", file_legend, 0);
      dataset.append(input_mode_data);
      input_h5_modes.close_current_object();
    }
  }
  if (read_modes_from_file) {
    InitializeJ::ConformalFactor initialize_j_constructed{
        1.0e-8,
        400,
        true,
        optimize_l_0_mode,
        use_beta_integral_estimate,
        iteration_heuristic,
        use_input_modes,
        filename};
    initialize_j_conformal_factor =
        serialize_and_deserialize(initialize_j_constructed);
  } else {
    InitializeJ::ConformalFactor initialize_j_constructed{
        1.0e-8,
        400,
        true,
        optimize_l_0_mode,
        use_beta_integral_estimate,
        iteration_heuristic,
        use_input_modes,
        input_modes};
    const auto initialize_j_cloned = initialize_j_constructed.get_clone();
    initialize_j_conformal_factor =
        *dynamic_cast<::Cce::InitializeJ::ConformalFactor*>(
            initialize_j_cloned.get());
  }

  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      initialize_j_conformal_factor, box_to_initialize,
      make_not_null(&node_lock));
  Approx iterative_solve_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);

  // perform gauge transforms on the boundary
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
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiBeta>>(
      box_to_initialize);
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiR>>(box_to_initialize);
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::BondiJ>>(box_to_initialize);
  db::mutate_apply<GaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>>>(
      box_to_initialize);
  const ComplexDataVector surface_zeroes{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0};
  db::mutate_apply<
      PrecomputeCceDependencies<Tags::BoundaryValue, Tags::OneMinusY>>(
      box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::BondiJ>>>(
      box_to_initialize);
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Dy<Tags::BondiJ>>>>(
      box_to_initialize);

  if (use_beta_integral_estimate) {
    // check the conformal factor on scri
    db::mutate_apply<ComputeBondiIntegrand<Tags::Integrand<Tags::BondiBeta>>>(
        box_to_initialize);
    db::mutate_apply<RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue,
                                          Tags::BondiBeta>>(box_to_initialize);
    Approx beta_estimate_approx = Approx::custom().epsilon(5.0e-8).scale(1.0);
    auto mutable_beta_copy = get(db::get<Tags::BondiBeta>(*box_to_initialize));
    SpinWeighted<ComplexDataVector, 0> scri_slice_beta{ComplexDataVector{
        mutable_beta_copy.data().data() +
            (number_of_radial_points - 1) * number_of_angular_points,
        number_of_angular_points}};
    if (optimize_l_0_mode) {
      CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_beta, surface_zeroes,
                                   beta_estimate_approx);
    } else {
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&scri_slice_beta), l_max, 1_st, l_max);
      CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_beta, surface_zeroes,
                                   beta_estimate_approx);
    }
  } else {
    // When not using the beta integral estimate, the conformal factor target on
    // the boundary is chosen to minimize the value of beta
    if (optimize_l_0_mode) {
      CHECK_ITERABLE_CUSTOM_APPROX(
          get(db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>>(
                  *box_to_initialize))
              .data(),
          surface_zeroes, iterative_solve_approx);
    } else {
      auto filtered_beta =
          get(db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>>(
              *box_to_initialize));
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&filtered_beta), l_max, 1_st, l_max);
      CHECK_ITERABLE_CUSTOM_APPROX(filtered_beta, surface_zeroes,
                                   iterative_solve_approx);
    }
  }

  check_boundary_and_asymptotic_j<Tags::EvolutionGaugeBoundaryValue>(
      box_to_initialize, number_of_radial_points, l_max);
  if (use_input_modes) {
    // check the correctness of the initial data:
    // - if using input modes, the 1/r part of j should match those modes.
    const SpinWeighted<ComplexDataVector, 2> scri_slice_dy_j;
    make_const_view(make_not_null(&scri_slice_dy_j),
                    get(db::get<Tags::Dy<Tags::BondiJ>>(*box_to_initialize)),
                    (number_of_radial_points - 1) * number_of_angular_points,
                    number_of_angular_points);
    // asymptotically, we have J = J^{(1)}/r = J^{(1)} * (1 - y) / 2 R,
    SpinWeighted<ComplexDataVector, 2> inverse_r_part_of_asymptotic_j =
        -2.0 * scri_slice_dy_j *
        get(db::get<Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>>(
            *box_to_initialize));
    auto inverse_r_asymptotic_modes =
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1_st,
                                           inverse_r_part_of_asymptotic_j),
            l_max);
    for (size_t i = 0; i < square(l_max + 1); ++i) {
      CAPTURE(i);
      CHECK(approx(real(inverse_r_asymptotic_modes.data()[i])) ==
            real(input_modes[i]));
      CHECK(approx(imag(inverse_r_asymptotic_modes.data()[i])) ==
            imag(input_modes[i]));
    }
  }
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  const auto spin_weight_1_created = TestHelpers::test_creation<
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic>(
      "SpinWeight1CoordPerturbation");
  CHECK(spin_weight_1_created ==
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
            SpinWeight1CoordPerturbation);
  const auto only_vary_gauge_d_created = TestHelpers::test_creation<
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic>("OnlyVaryGaugeD");
  CHECK(only_vary_gauge_d_created ==
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::OnlyVaryGaugeD);
  const std::string spin_weight_1_streamed =
      MakeString{} << ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
          SpinWeight1CoordPerturbation;
  CHECK(spin_weight_1_streamed == "SpinWeight1CoordPerturbation");
  const std::string only_vary_gauge_d_streamed =
      MakeString{}
      << ::Cce::InitializeJ::ConformalFactorIterationHeuristic::OnlyVaryGaugeD;
  CHECK(only_vary_gauge_d_streamed == "OnlyVaryGaugeD");
}

}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.InitializeJ", "[Unit][Cce]") {
  // `CauchySecondOrder` holds an interpolator, so serializing it needs the
  // derived span interpolators registered.
  register_derived_classes_with_charm<intrp::SpanInterpolator>();
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<size_t> sdist{5, 6};
  const size_t l_max = sdist(generator);
  const size_t number_of_radial_points = sdist(generator);

  using boundary_variables_tag = ::Tags::Variables<tmpl::push_back<
      InitializeJ::InverseCubic<true>::boundary_tags, Tags::PartiallyFlatGaugeC,
      Tags::PartiallyFlatGaugeD, Tags::PartiallyFlatGaugeOmega,
      Spectral::Swsh::Tags::Derivative<Tags::PartiallyFlatGaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiJ>,
      Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::BondiJ>>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>,
      Tags::BoundaryValue<Tags::BondiU>, Tags::BoundaryValue<Tags::BondiW>,
      Tags::BoundaryValue<Tags::BondiQ>,
      Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>,
      Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>,
      Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>>;
  using pre_swsh_derivatives_variables_tag = ::Tags::Variables<tmpl::list<
      Tags::BondiJ, Tags::Dy<Tags::BondiJ>, Tags::Dy<Tags::Dy<Tags::BondiJ>>,
      Tags::BondiK, Tags::BondiR, Tags::Integrand<Tags::BondiBeta>,
      Tags::BondiBeta, Tags::OneMinusY, Tags::Psi0>>;
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
             Tags::BoundaryValue<Tags::BondiBeta>,
             Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>,
             Tags::BoundaryValue<Tags::BondiJ>>(
      [&generator, &dist, &l_max](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_r,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_beta,
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
                           100.0;
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*boundary_r)), l_max, l_max / 2);
        get(*boundary_beta) =
            Spectral::Swsh::inverse_swsh_transform(l_max, 1, generated_r_modes);
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*boundary_r)), l_max, l_max / 2);

        get(*boundary_dr_j) = -get(*boundary_j) / get(*boundary_r);
      },
      make_not_null(&box_to_initialize));
  // The CauchySecondOrder generator additionally consumes the worldtube values
  // of U, W, Q, Du(J), Du(Dr(J)), and Du(R) to evaluate the H hypersurface
  // equation for dy^2 J. These enter through factors of the (large) worldtube
  // radius R, so to keep the resulting dy^2 J at the same (strain) scale as J -
  // and thus small enough for the angular coordinate solve to converge - they
  // are drawn from a correspondingly smaller distribution.
  UniformCustomDistribution<double> second_order_dist(1.0e-10, 1.0e-9);
  db::mutate<Tags::BoundaryValue<Tags::BondiU>,
             Tags::BoundaryValue<Tags::BondiW>,
             Tags::BoundaryValue<Tags::BondiQ>,
             Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>,
             Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>,
             Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>(
      [&generator, &second_order_dist, &l_max](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
              boundary_u,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_w,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
              boundary_q,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              boundary_du_j,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
              boundary_du_dr_j,
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              boundary_du_r) {
        assign_random_swsh_boundary_value<1>(
            boundary_u, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
        assign_random_swsh_boundary_value<0>(
            boundary_w, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
        assign_random_swsh_boundary_value<1>(
            boundary_q, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
        assign_random_swsh_boundary_value<2>(
            boundary_du_j, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
        assign_random_swsh_boundary_value<2>(
            boundary_du_dr_j, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
        assign_random_swsh_boundary_value<0>(
            boundary_du_r, make_not_null(&generator),
            make_not_null(&second_order_dist), l_max);
      },
      make_not_null(&box_to_initialize));
  {
    INFO("Check inverse cubic initial data generator");
    test_initialize_j_inverse_cubic(make_not_null(&box_to_initialize), l_max,
                                    number_of_radial_points);
  }
  {
    INFO("Check zero nonsmooth initial data generator");
    test_initialize_j_zero_nonsmooth(make_not_null(&box_to_initialize), l_max,
                                     number_of_radial_points);
  }
  CHECK_THROWS_WITH(
      (test_zero_non_smooth_error(make_not_null(&box_to_initialize), l_max,
                                  number_of_radial_points)),
      Catch::Matchers::ContainsSubstring(
          "Initial data iterative angular solve"));
  {
    INFO("Check no incoming radiation initial data generator");
    test_initialize_j_no_radiation(make_not_null(&box_to_initialize), l_max,
                                   number_of_radial_points);
  }
  {
    INFO("Check conformal factor initial data generator");
    test_initialize_j_conformal_factor(
        make_not_null(&box_to_initialize), false, false, false, false,
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
            SpinWeight1CoordPerturbation,
        l_max, number_of_radial_points);
    test_initialize_j_conformal_factor(
        make_not_null(&box_to_initialize), false, true, false, false,
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::OnlyVaryGaugeD,
        l_max, number_of_radial_points);
    test_initialize_j_conformal_factor(
        make_not_null(&box_to_initialize), true, false, true, false,
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
            SpinWeight1CoordPerturbation,
        l_max, number_of_radial_points);
    test_initialize_j_conformal_factor(
        make_not_null(&box_to_initialize), true, true, true, true,
        ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
            SpinWeight1CoordPerturbation,
        l_max, number_of_radial_points);
  }
  {
    INFO("Check second-order initial data generator");
    test_initialize_j_cauchy_second_order(make_not_null(&box_to_initialize),
                                          l_max, number_of_radial_points);
  }

  CHECK_THROWS_WITH(test_cauchy_second_order_j2_threshold_error(
                        make_not_null(&box_to_initialize)),
                    Catch::Matchers::ContainsSubstring(
                        "set by the MaxPartiallyFlatJ2 option"));
  CHECK_THROWS_WITH(
      test_cauchy_second_order_j0_convergence_error(
          make_not_null(&box_to_initialize)),
      Catch::Matchers::ContainsSubstring("Initial data iterative angular solve "
                                         "did not reach target tolerance"));
  {
    INFO(
        "Check the second-order generator's Du(Dr(J)) interpolator survives "
        "cloning and serialization");
    test_cauchy_second_order_interpolator_round_trip();
  }
  {
    INFO("Check the fixed-point solve for the second-order J^(2) coefficient");
    test_cauchy_second_order_j2_fixed_point();
  }
  {
    INFO("Check the second-order ansatz matches the worldtube data for any J2");
    test_cauchy_second_order_radial_ansatz_coefficients();
  }
  CHECK_THROWS_WITH(
      test_cauchy_second_order_j2_convergence_error(
          make_not_null(&box_to_initialize)),
      Catch::Matchers::ContainsSubstring("The initial J^(2) fixed-point solve "
                                         "did not reach target tolerance"));
  CHECK_THROWS_WITH(test_cauchy_second_order_cauchy_j0_threshold(
                        make_not_null(&box_to_initialize)),
                    Catch::Matchers::ContainsSubstring(
                        "set by the MaxAngularSolveError option"));
  CHECK_THROWS_WITH(
      test_cauchy_second_order_asymptotic_j_error(
          make_not_null(&box_to_initialize)),
      Catch::Matchers::ContainsSubstring(
          "The asymptotic value of the initial J in Cauchy coordinates has "
          "magnitude"));
}
}  // namespace Cce

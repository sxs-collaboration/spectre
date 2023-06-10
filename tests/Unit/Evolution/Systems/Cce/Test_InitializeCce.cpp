// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
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
#include "Helpers/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {

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
      },
      make_not_null(&box_to_initialize));
  auto node_lock = Parallel::NodeLock{};
  db::mutate_apply<InitializeJ::InitializeJ<false>::mutate_tags,
                   InitializeJ::InitializeJ<false>::argument_tags>(
      InitializeJ::ZeroNonSmooth{1.0e-12, 1, true},
      make_not_null(&box_to_initialize), make_not_null(&node_lock));
  ERROR("Failed to trigger ERROR in an error test");
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
      Tags::EvolutionGaugeBoundaryValue<Tags::Dr<Tags::BondiJ>>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiR>,
      Tags::EvolutionGaugeBoundaryValue<Tags::BondiBeta>>>;
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
  {
    INFO("Check no incoming radiation initial data generator")
    test_initialize_j_no_radiation(make_not_null(&box_to_initialize), l_max,
                                   number_of_radial_points);
  }
  {
    INFO("Check conformal factor initial data generator")
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
}
}  // namespace Cce

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/Cce/InitializeCce.hpp"
#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.InitializeJ",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<size_t> sdist{5, 8};
  const size_t l_max = sdist(generator);
  const size_t number_of_radial_points = sdist(generator);

  using boundary_variables_tag =
      ::Tags::Variables<InitializeJInverseCubic::boundary_tags>;
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
      tensor_variables_tag, Tags::LMax, Tags::NumberOfRadialPoints>>(
      typename boundary_variables_tag::type{number_of_boundary_points},
      typename pre_swsh_derivatives_variables_tag::type{
          number_of_volume_points},
      typename tensor_variables_tag::type{number_of_boundary_points}, l_max,
      number_of_radial_points);

  // generate some random values for the boundary data
  UniformCustomDistribution<double> dist(0.1, 1.0);
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
        get(*boundary_j).data() = make_with_random_values<ComplexDataVector>(
            make_not_null(&generator), make_not_null(&dist),
            Spectral::Swsh::number_of_swsh_collocation_points(l_max));
        get(*boundary_r).data() = make_with_random_values<ComplexDataVector>(
            make_not_null(&generator), make_not_null(&dist),
            Spectral::Swsh::number_of_swsh_collocation_points(l_max));
        get(*boundary_dr_j).data() = make_with_random_values<ComplexDataVector>(
            make_not_null(&generator), make_not_null(&dist),
            Spectral::Swsh::number_of_swsh_collocation_points(l_max));
      });

  db::mutate_apply<InitializeJ::mutate_tags, InitializeJ::argument_tags>(
      InitializeJInverseCubic{}, make_not_null(&box_to_initialize));

  SpinWeighted<ComplexDataVector, 2> dy_j{
      number_of_radial_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  SpinWeighted<ComplexDataVector, 2> dy_dy_j{
      number_of_radial_points *
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  logical_partial_directional_derivative_of_complex(
      make_not_null(&dy_j.data()),
      get(db::get<Tags::BondiJ>(box_to_initialize)).data(),
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

  // The goal for the initial data is that it should:
  // - match the value of J and its first derivative on the boundary
  // - have vanishing value and second derivative at scri+
  ComplexDataVector mutable_j_copy =
      get(db::get<Tags::BondiJ>(box_to_initialize)).data();
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
      get(db::get<Tags::BoundaryValue<Tags::BondiJ>>(box_to_initialize)).data(),
      boundary_slice_j, cce_approx);
  const auto boundary_slice_dr_j =
      (2.0 / get(db::get<Tags::BoundaryValue<Tags::BondiR>>(box_to_initialize))
                 .data()) *
      boundary_slice_dy_j;
  CHECK_ITERABLE_CUSTOM_APPROX(
      boundary_slice_dr_j,
      get(db::get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
              box_to_initialize))
          .data(),
      cce_approx);
  const ComplexDataVector scri_plus_zeroes{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max), 0.0};
  CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_j, scri_plus_zeroes, cce_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(scri_slice_dy_dy_j, scri_plus_zeroes,
                               cce_approx);
}
}  // namespace Cce

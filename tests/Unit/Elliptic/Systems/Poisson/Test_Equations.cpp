// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include <pup.h>

// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tags::div
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
template <size_t Dim, typename DbTagsList>
void test_first_order_operator_action(db::DataBox<DbTagsList>&& domain_box) {
  const auto& inertial_coords =
      get<Tags::Coordinates<Dim, Frame::Inertial>>(domain_box);

  const Poisson::Solutions::ProductOfSinusoids<Dim> solution(
      make_array<Dim>(1.));
  const auto auxiliary_field =
      get<Poisson::AuxiliaryField<Dim>>(solution.variables(
          inertial_coords,
          tmpl::list<Poisson::Field, Poisson::AuxiliaryField<Dim>>{}));

  auto grad_field = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
      inertial_coords, 0.);
  for (size_t d = 0; d < Dim; d++) {
    grad_field.get(d) = auxiliary_field.get(d);
  }

  auto box = db::create_from<
      db::RemoveTags<>,
      db::AddSimpleTags<
          Tags::deriv<LinearSolver::Tags::Operand<Poisson::Field>,
                      tmpl::size_t<Dim>, Frame::Inertial>,
          LinearSolver::Tags::Operand<Poisson::AuxiliaryField<Dim>>,
          LinearSolver::Tags::OperatorAppliedTo<
              LinearSolver::Tags::Operand<Poisson::Field>>,
          LinearSolver::Tags::OperatorAppliedTo<
              LinearSolver::Tags::Operand<Poisson::AuxiliaryField<Dim>>>>>(
      std::move(domain_box), std::move(grad_field), auxiliary_field,
      make_with_value<db::item_type<LinearSolver::Tags::OperatorAppliedTo<
          LinearSolver::Tags::Operand<Poisson::Field>>>>(inertial_coords, 0.),
      make_with_value<db::item_type<LinearSolver::Tags::OperatorAppliedTo<
          LinearSolver::Tags::Operand<Poisson::AuxiliaryField<Dim>>>>>(
          inertial_coords, 0.));

  db::mutate_apply<
      tmpl::list<
          LinearSolver::Tags::OperatorAppliedTo<
              LinearSolver::Tags::Operand<Poisson::Field>>,
          LinearSolver::Tags::OperatorAppliedTo<
              LinearSolver::Tags::Operand<Poisson::AuxiliaryField<Dim>>>>,
      typename Poisson::ComputeFirstOrderOperatorAction<Dim>::argument_tags>(
      Poisson::ComputeFirstOrderOperatorAction<Dim>{}, make_not_null(&box));

  const auto auxiliary_source =
      get<Tags::Source<Poisson::AuxiliaryField<Dim>>>(solution.variables(
          inertial_coords,
          tmpl::list<Tags::Source<Poisson::AuxiliaryField<Dim>>>{}));

  // Field volume contribution should be -div(v), which is not exactly the
  // source f because of discretization error
  auto div_vars =
      make_with_value<Variables<tmpl::list<Poisson::AuxiliaryField<Dim>>>>(
          auxiliary_field, 0.);
  get<Poisson::AuxiliaryField<Dim>>(div_vars) = auxiliary_field;
  auto div_auxiliary_field =
      get<Tags::div<Poisson::AuxiliaryField<Dim>>>(divergence(
          div_vars, get<Tags::Mesh<Dim>>(box),
          get<Tags::InverseJacobian<Tags::ElementMap<Dim>,
                                    Tags::Coordinates<Dim, Frame::Logical>>>(
              box)));
  div_auxiliary_field.get() *= -1.;
  CHECK_ITERABLE_APPROX(get<LinearSolver::Tags::OperatorAppliedTo<
                            LinearSolver::Tags::Operand<Poisson::Field>>>(box),
                        div_auxiliary_field);
  // Auxiliary field volume contribution grad(u) - v should be exactly zero
  // since grad(u) was set to v, as opposed to being computed numerically
  CHECK_ITERABLE_APPROX(
      get<LinearSolver::Tags::OperatorAppliedTo<
          LinearSolver::Tags::Operand<Poisson::AuxiliaryField<Dim>>>>(box),
      auxiliary_source);
}

template <size_t Dim, typename DbTagsList>
void test_first_order_normal_dot_fluxes(
    const db::DataBox<DbTagsList>& domain_box) {
  const auto num_points =
      get<Tags::Mesh<Dim>>(domain_box).number_of_grid_points();

  const auto& inertial_coords =
      get<Tags::Coordinates<Dim, Frame::Inertial>>(domain_box);

  const Poisson::Solutions::ProductOfSinusoids<Dim> solution(
      make_array<Dim>(1.));
  const auto field_variables = solution.variables(
      inertial_coords,
      tmpl::list<Poisson::Field, Poisson::AuxiliaryField<Dim>>{});

  // Any numbers are fine, doesn't have anything to do with unit normal
  tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal(num_points, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    unit_normal.get(d) = inertial_coords.get(d);
  }

  Scalar<DataVector> normal_dot_flux_for_field(num_points, 0.0);
  tnsr::I<DataVector, Dim, Frame::Inertial> normal_dot_flux_for_aux_field(
      num_points, 0.0);

  Poisson::ComputeFirstOrderNormalDotFluxes<Dim>::apply(
      make_not_null(&normal_dot_flux_for_field),
      make_not_null(&normal_dot_flux_for_aux_field),
      get<Poisson::Field>(field_variables),
      get<Poisson::AuxiliaryField<Dim>>(field_variables), unit_normal);

  DataVector normal_dot_flux_for_field_expected(num_points, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    normal_dot_flux_for_field_expected +=
        unit_normal.get(d) *
        get<Poisson::AuxiliaryField<Dim>>(field_variables).get(d);
  }
  CHECK_ITERABLE_APPROX(get(normal_dot_flux_for_field),
                        normal_dot_flux_for_field_expected);

  tnsr::I<DataVector, Dim, Frame::Inertial>
      normal_dot_flux_for_aux_field_expected(num_points, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    normal_dot_flux_for_aux_field_expected.get(d) =
        -unit_normal.get(d) * get(get<Poisson::Field>(field_variables));
    CHECK_ITERABLE_APPROX(normal_dot_flux_for_aux_field.get(d),
                          normal_dot_flux_for_aux_field_expected.get(d));
  }
}

template <class... Tags, class FluxType, class... NormalDotNumericalFluxTypes>
void apply_numerical_flux(
    const FluxType& flux,
    const Variables<tmpl::list<Tags...>>& packaged_data_int,
    const Variables<tmpl::list<Tags...>>& packaged_data_ext,
    NormalDotNumericalFluxTypes&&... normal_dot_numerical_flux) {
  flux(std::forward<NormalDotNumericalFluxTypes>(normal_dot_numerical_flux)...,
       get<Tags>(packaged_data_int)..., get<Tags>(packaged_data_ext)...);
}

template <size_t Dim, typename DbTagsList>
void test_first_order_internal_penalty_flux(
    const db::DataBox<DbTagsList>& domain_box) {
  const auto num_points =
      get<Tags::Mesh<Dim>>(domain_box).number_of_grid_points();

  const auto& inertial_coords =
      get<Tags::Coordinates<Dim, Frame::Inertial>>(domain_box);

  const Poisson::Solutions::ProductOfSinusoids<Dim> solution(
      make_array<Dim>(1.));
  const auto field_variables = solution.variables(
      inertial_coords,
      tmpl::list<Poisson::Field, Poisson::AuxiliaryField<Dim>>{});

  // Any numbers are fine, doesn't have anything to do with unit normal
  tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal(num_points, 0.0);
  tnsr::i<DataVector, Dim, Frame::Inertial> opposite_unit_normal(num_points,
                                                                 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    unit_normal.get(d) = inertial_coords.get(d);
    opposite_unit_normal.get(d) = -inertial_coords.get(d);
  }

  Variables<typename Poisson::FirstOrderInternalPenaltyFlux<Dim>::package_tags>
      packaged_data_int(num_points, 0.0);
  Variables<typename Poisson::FirstOrderInternalPenaltyFlux<Dim>::package_tags>
      packaged_data_ext(num_points, 0.0);

  tnsr::i<DataVector, Dim, Frame::Inertial> grad_field(num_points, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    grad_field.get(d) =
        get<Poisson::AuxiliaryField<Dim>>(field_variables).get(d);
  }

  Poisson::FirstOrderInternalPenaltyFlux<Dim> flux_computer(10.);
  Scalar<DataVector> normal_dot_numerical_flux_for_field(num_points, 0.0);
  tnsr::I<DataVector, Dim, Frame::Inertial>
      normal_dot_numerical_flux_for_aux_field(num_points, 0.0);

  tnsr::i<DataVector, Dim, Frame::Inertial> normal_times_field(num_points, 0.0);
  Scalar<DataVector> normal_dot_aux_field(num_points, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    normal_times_field.get(d) =
        unit_normal.get(d) * get(get<Poisson::Field>(field_variables));
    get(normal_dot_aux_field) +=
        unit_normal.get(d) *
        get<Poisson::AuxiliaryField<Dim>>(field_variables).get(d);
  }

  // Consistency: u^*(u,u)=u
  {
    flux_computer.package_data(make_not_null(&packaged_data_int),
                               get<Poisson::Field>(field_variables), grad_field,
                               unit_normal);
    flux_computer.package_data(make_not_null(&packaged_data_ext),
                               get<Poisson::Field>(field_variables), grad_field,
                               opposite_unit_normal);
    apply_numerical_flux(
        flux_computer, packaged_data_int, packaged_data_ext,
        make_not_null(&normal_dot_numerical_flux_for_field),
        make_not_null(&normal_dot_numerical_flux_for_aux_field));

    CHECK_ITERABLE_APPROX(get(normal_dot_numerical_flux_for_field),
                          get(normal_dot_aux_field));
    for (size_t d = 0; d < Dim; ++d) {
      CHECK_ITERABLE_APPROX(normal_dot_numerical_flux_for_aux_field.get(d),
                            -normal_times_field.get(d));
    }
  }

  // Manufacture different data for the exterior
  Scalar<DataVector> ext_field(num_points, 0.0);
  get(ext_field) += get<0>(inertial_coords);
  tnsr::i<DataVector, Dim, Frame::Inertial> ext_grad_field(num_points, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    ext_grad_field.get(d) =
        get<Poisson::AuxiliaryField<Dim>>(field_variables).get(d) +
        inertial_coords.get(d);
  }

  // Conservation: u^* is single-valued (same on both sides of the interface)
  {
    flux_computer.package_data(make_not_null(&packaged_data_int),
                               get<Poisson::Field>(field_variables), grad_field,
                               unit_normal);
    flux_computer.package_data(make_not_null(&packaged_data_ext), ext_field,
                               ext_grad_field, opposite_unit_normal);
    apply_numerical_flux(
        flux_computer, packaged_data_int, packaged_data_ext,
        make_not_null(&normal_dot_numerical_flux_for_field),
        make_not_null(&normal_dot_numerical_flux_for_aux_field));

    Scalar<DataVector> reversed_normal_dot_numerical_flux_for_field(num_points,
                                                                    0.0);
    tnsr::I<DataVector, Dim, Frame::Inertial>
        reversed_normal_dot_numerical_flux_for_aux_field(num_points, 0.0);
    apply_numerical_flux(
        flux_computer, packaged_data_ext, packaged_data_int,
        make_not_null(&reversed_normal_dot_numerical_flux_for_field),
        make_not_null(&reversed_normal_dot_numerical_flux_for_aux_field));

    CHECK_ITERABLE_APPROX(get(normal_dot_numerical_flux_for_field),
                          -get(reversed_normal_dot_numerical_flux_for_field));
    for (size_t d = 0; d < Dim; ++d) {
      CHECK_ITERABLE_APPROX(
          normal_dot_numerical_flux_for_aux_field.get(d),
          -reversed_normal_dot_numerical_flux_for_aux_field.get(d));
    }
  }

  // Explicit data
  {
    flux_computer.package_data(make_not_null(&packaged_data_int),
                               get<Poisson::Field>(field_variables), grad_field,
                               unit_normal);
    flux_computer.package_data(make_not_null(&packaged_data_ext), ext_field,
                               ext_grad_field, opposite_unit_normal);
    apply_numerical_flux(
        flux_computer, packaged_data_int, packaged_data_ext,
        make_not_null(&normal_dot_numerical_flux_for_field),
        make_not_null(&normal_dot_numerical_flux_for_aux_field));

    Scalar<DataVector> expected_normal_dot_numerical_flux_for_field(num_points,
                                                                    0.0);
    tnsr::I<DataVector, Dim, Frame::Inertial>
        expected_normal_dot_numerical_flux_for_aux_field(num_points, 0.0);
    get(expected_normal_dot_numerical_flux_for_field) =
        -10. * (get(get<Poisson::Field>(field_variables)) - get(ext_field));
    for (size_t d = 0; d < Dim; d++) {
      get(expected_normal_dot_numerical_flux_for_field) +=
          0.5 * unit_normal.get(d) *
          (grad_field.get(d) + ext_grad_field.get(d));
      expected_normal_dot_numerical_flux_for_aux_field.get(d) =
          -0.5 * unit_normal.get(d) *
          (get(get<Poisson::Field>(field_variables)) + get(ext_field));
    }

    CHECK_ITERABLE_APPROX(get(normal_dot_numerical_flux_for_field),
                          get(expected_normal_dot_numerical_flux_for_field));
    for (size_t d = 0; d < Dim; ++d) {
      CHECK_ITERABLE_APPROX(
          normal_dot_numerical_flux_for_aux_field.get(d),
          expected_normal_dot_numerical_flux_for_aux_field.get(d));
    }
  }
}

template <size_t Dim>
using simple_tags = db::AddSimpleTags<Tags::Mesh<Dim>, Tags::ElementMap<Dim>>;
template <size_t Dim>
using compute_tags = db::AddComputeTags<
    Tags::LogicalCoordinates<Dim>,
    Tags::MappedCoordinates<Tags::ElementMap<Dim>,
                            Tags::Coordinates<Dim, Frame::Logical>>,
    Tags::InverseJacobian<Tags::ElementMap<Dim>,
                          Tags::Coordinates<Dim, Frame::Logical>>>;
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Poisson.FirstOrder",
                  "[Unit][Elliptic]") {
  using Affine = domain::CoordinateMaps::Affine;
  Mesh<1> mesh_1d{5, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto};
  ElementMap<1, Frame::Inertial> element_map_1d{
      ElementId<1>{0, make_array<1>(SegmentId{0, 0})},
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine{-1., 1., 0., M_PI})};
  auto domain_box_1d = db::create<simple_tags<1>, compute_tags<1>>(
      mesh_1d, std::move(element_map_1d));
  Mesh<2> mesh_2d{5, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto};
  ElementMap<2, Frame::Inertial> element_map_2d{
      ElementId<2>{0, make_array<2>(SegmentId{0, 0})},
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>(
              Affine{-1., 1., 0., M_PI}, Affine{-1., 1., 0., M_PI}))};
  auto domain_box_2d = db::create<simple_tags<2>, compute_tags<2>>(
      mesh_2d, std::move(element_map_2d));
  Mesh<3> mesh_3d{5, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto};
  ElementMap<3, Frame::Inertial> element_map_3d{
      ElementId<3>{0, make_array<3>(SegmentId{0, 0})},
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>(
              Affine{-1., 1., 0., M_PI}, Affine{-1., 1., 0., M_PI},
              Affine{-1., 1., 0., M_PI}))};
  auto domain_box_3d = db::create<simple_tags<3>, compute_tags<3>>(
      mesh_3d, std::move(element_map_3d));

  SECTION("NormalDotFluxes") {
    test_first_order_normal_dot_fluxes<1>(domain_box_1d);
    test_first_order_normal_dot_fluxes<2>(domain_box_2d);
    test_first_order_normal_dot_fluxes<3>(domain_box_3d);
  }
  SECTION("InternalPenaltyFlux") {
    test_first_order_internal_penalty_flux<1>(domain_box_1d);
    test_first_order_internal_penalty_flux<2>(domain_box_2d);
    test_first_order_internal_penalty_flux<3>(domain_box_3d);
  }
  SECTION("OperatorAction") {
    test_first_order_operator_action<1>(std::move(domain_box_1d));
    test_first_order_operator_action<2>(std::move(domain_box_2d));
    test_first_order_operator_action<3>(std::move(domain_box_3d));
  }
}

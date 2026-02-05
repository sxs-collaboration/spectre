// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/Ccz4/ATilde.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Sommerfeld.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/DummyReconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/SoTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Ccz4WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugePlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
namespace {

template <bool EnableSubcell>
struct DummyEvolutionMetaVars {
  struct SubcellOptions {
    static constexpr bool subcell_enabled_at_external_boundary = EnableSubcell;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BoundaryConditions::BoundaryCondition,
                             BoundaryConditions::standard_boundary_conditions>>;
  };
};

using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

// Test second order CCZ4 in Minkowski spacetime
void test_minkowski(const bool evolve_lapse_and_shift) {
  // set up subcell grid
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 5;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};

  const std::array<double, SpatialDim> lower_bound{-2., 0., -0.5};
  const std::array<double, SpatialDim> upper_bound{2., 2., -0.1};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  // set up displaced logical coords
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = coord_map(logical_coords);
  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};

  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element();

  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::Minkowski::
                  compute_prim_solution_for_Minkowski<false>,
              coords_range);

  // Get system evolved variables
  // Use the physical inertial coords
  const auto evolved_vars = TestHelpers::Ccz4::fd::detail::Minkowski::
      compute_prim_solution_for_Minkowski(x);

  const DataVector used_for_size =
      DataVector(subcell_mesh.number_of_grid_points(),
                 std::numeric_limits<double>::signaling_NaN());
  const auto k_0 = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  // put needed quantities into databox
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  auto box = db::create<db::AddSimpleTags<
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<SpatialDim>,
      fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<false>>,
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>, dt_variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>,
      evolution::dg::subcell::Tags::Coordinates<SpatialDim, Frame::Inertial>>>(
      kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<false>{}, evolved_vars, eta, k_0,
      upper_spatial_z4_constraint,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
      all_ghost_data,
      std::vector<DirectionMap<
          SpatialDim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>{},
      x);
  // Check that all time derivatives are 0
  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));
  const auto zero = DataVector(used_for_size.size(), 0.0);

  tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
    [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<::Tags::dt<Tag>>();
        CAPTURE(tag_name);
        for (auto& component : get<::Tags::dt<Tag>>(box)) {
            CHECK_ITERABLE_APPROX(component, zero);
        }
  });
}

// Test second-order CCZ4 in KerrSchild spacetime
//
// evolve_shift: whether or not to evolve the shift (always true for SO-CCZ4);
// slicing_condition_type: which slicing condition to use (always 1+log for
// SO-CCZ4)
void test_kerrschild(const bool evolve_lapse_and_shift) {
  const bool evolve_shift = evolve_lapse_and_shift;
  const Ccz4::SlicingConditionType slicing_condition_type =
      Ccz4::SlicingConditionType::Log;  // always use 1+log slicing

  // set up subcell grid
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 20;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};

  const std::array<double, SpatialDim> lower_bound{0.8, 1., 1.3};
  const std::array<double, SpatialDim> upper_bound{1.2, 1.2, 1.4};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  // set up displaced logical coords
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = coord_map(logical_coords);
  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element();

  // Setup solution
  const double mass = 2.0;
  const std::array<double, SpatialDim> spin{{0.2, 0.4, 0.8}};
  const std::array<double, SpatialDim> center{{0.2, 0.5, 0.1}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  const double f = Ccz4::fd::System::f;

  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::KerrSchild::
                  compute_prim_solution_for_KerrSchild,
              coords_range, t, f, evolve_shift, solution);

  const auto evolved_vars = TestHelpers::Ccz4::fd::detail::KerrSchild::
      compute_prim_solution_for_KerrSchild(x, t, f, evolve_shift, solution);

  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
  const auto d_lapse = partial_derivative(
      lapse, subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);
  const DataVector used_for_size =
      DataVector(subcell_mesh.number_of_grid_points(),
                 std::numeric_limits<double>::signaling_NaN());
  const auto eta = make_with_value<Scalar<DataVector>>(
      used_for_size, 0.1);                      // change eta to non-zero later
  const Scalar<DataVector> slicing_condition =  // g(\alpha)
      TestHelpers::Ccz4::fd::detail::KerrSchild::get_slicing_condition(
          slicing_condition_type, lapse);
  const auto k_0 = TestHelpers::Ccz4::fd::detail::KerrSchild::get_k_0_kerr(
      get<gr::Tags::Shift<DataVector, SpatialDim>>(evolved_vars), lapse,
      d_lapse, slicing_condition,
      get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars),
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars));
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  // put needed quantities into databox
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  auto box = db::create<db::AddSimpleTags<
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<SpatialDim>,
      fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<false>>,
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>, dt_variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>,
        evolution::dg::subcell::Tags::Coordinates<
            SpatialDim, Frame::Inertial>>>(
      kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<false>{}, evolved_vars, eta, k_0,
      upper_spatial_z4_constraint,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
      all_ghost_data,
      std::vector<DirectionMap<
          SpatialDim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>{},
      x);
  // Check that all time derivatives are 0
  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));
  const auto zero = DataVector(used_for_size.size(), 0.0);
  const Approx custom_approx =
      Approx::custom().epsilon(1.0e-9).scale(*std::max_element(
          evolved_vars.data(), evolved_vars.data() + evolved_vars.size() - 1));

  // Remove dt_b from the list, which will be checked separately below
  tmpl::for_each<tmpl::remove<Ccz4::fd::System::variables_tag_list,
    ::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<::Tags::dt<Tag>>();
        CAPTURE(tag_name);
        for (auto& component : get<::Tags::dt<Tag>>(box)) {
          CHECK_ITERABLE_CUSTOM_APPROX(component, zero, custom_approx);
        }
      });

  // eq 12i
  // \partial_t b will not be 0 for KerrSchild if evolve_shift == true
  // since we assume the shift is time-independent for testing
  // but KerrSchild is not stationary under 1+log slicing
  const auto d_gamma_hat = partial_derivative(
      get<::Ccz4::Tags::GammaHat<DataVector, SpatialDim>>(box), subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian);
  const auto d_b = partial_derivative(
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>(box),
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);
  const tnsr::I<DataVector, SpatialDim, FrameType> dt_b_expected =
      TestHelpers::Ccz4::fd::detail::KerrSchild::get_dt_b_kerr_expected(
          evolve_shift, get<::Ccz4::Tags::Eta<DataVector>>(box),
          get<gr::Tags::Shift<DataVector, SpatialDim>>(box), d_gamma_hat,
          get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>(box), d_b);
  const auto& dt_b_actual =
      get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>>(
          box);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_b_actual, dt_b_expected, custom_approx);
}

// Test second-order CCZ4 in a GaugePlaneWave spacetime
void test_gauge_plane_wave(
    const std::array<double, 3>& wave_vector,
    std::unique_ptr<MathFunction<1, Frame::Inertial>> profile, const double t,
    const bool evolve_lapse_and_shift) {
  // set up subcell grid
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 20;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};
  const std::array<double, SpatialDim> lower_bound{-0.5, -2., 1.};
  const std::array<double, SpatialDim> upper_bound{0.0, 2., 2.};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  // set up displaced logical coords
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = coord_map(logical_coords);
  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  double omega = 0.0;
  for (const auto& k : wave_vector) {
    omega += square(k);
  }
  omega = pow(omega, 1.0 / 2.0);

  const DataVector used_for_size =
      DataVector(subcell_mesh.number_of_grid_points(),
                 std::numeric_limits<double>::signaling_NaN());

  tnsr::i<DataVector, SpatialDim, FrameType> k_tnsr{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    k_tnsr.get(i) =
        make_with_value<DataVector>(used_for_size, gsl::at(wave_vector, i));
  }

  const gr::Solutions::GaugePlaneWave<SpatialDim>::IntermediateVars<DataVector>
      intermediate_sol(wave_vector, profile, omega, x, t);
  const Scalar<DataVector> h{intermediate_sol.h};
  const Scalar<DataVector> du_h{intermediate_sol.du_h};
  const Scalar<DataVector> du_du_h{intermediate_sol.du_du_h};

  // Setup solutions
  const gr::Solutions::GaugePlaneWave<SpatialDim> solution(wave_vector,
                                                           std::move(profile));
  const auto gauge_plane_wave_vars = solution.variables(
      x, t,
      typename gr::Solutions::GaugePlaneWave<SpatialDim>::tags<DataVector>{});

  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element();

  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data = TestHelpers::Ccz4::fd::detail::compute_ghost_data(
          subcell_mesh, x, element.neighbors(), ghost_zone_size,
          TestHelpers::Ccz4::fd::detail::GaugePlaneWave::
              compute_prim_solution_for_GaugePlaneWave,
          coords_range, t, solution, intermediate_sol);

  const auto evolved_vars = TestHelpers::Ccz4::fd::detail::GaugePlaneWave::
      compute_prim_solution_for_GaugePlaneWave(x, t, solution,
                                               intermediate_sol);

  const auto& k_0 =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars);

  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  // put needed quantities into databox
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  auto box = db::create<db::AddSimpleTags<
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<SpatialDim>,
      fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<false>>,
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>, dt_variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>,
        evolution::dg::subcell::Tags::Coordinates<
            SpatialDim, Frame::Inertial>>>(
      kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<false>{}, evolved_vars, eta, k_0,
      upper_spatial_z4_constraint,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
      all_ghost_data,
      std::vector<DirectionMap<
          SpatialDim,
          std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>{},
      x);
  // Check all time derivatives
  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));

  const Approx custom_approx =
      Approx::custom().epsilon(1.0e-9).scale(*std::max_element(
          evolved_vars.data(), evolved_vars.data() + evolved_vars.size() - 1));
  // eq 12a
  const auto& dt_conformal_spatial_metric_actual =
      get<::Tags::dt<::Ccz4::Tags::ConformalMetric<DataVector, SpatialDim>>>(
          box);

  const auto conformal_factor =
      get(get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars));
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = square(conformal_factor);
  const auto& dt_spatial_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataVector, SpatialDim>>>(
          gauge_plane_wave_vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim, FrameType>>(
          gauge_plane_wave_vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, FrameType>>(
          gauge_plane_wave_vars);
  const auto dt_conformal_spatial_metric_expected = TestHelpers::Ccz4::fd::
      detail::GaugePlaneWave::get_dt_conformal_spatial_metric_gauge_plane_wave(
          conformal_factor_squared, spatial_metric, inverse_spatial_metric,
          dt_spatial_metric);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_conformal_spatial_metric_actual,
                               dt_conformal_spatial_metric_expected,
                               custom_approx);

  // eq 12b
  const auto& dt_lapse_actual =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(box);
  const auto& d_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                        FrameType>>(gauge_plane_wave_vars);
  auto dt_lapse_expected = make_with_value<Scalar<DataVector>>(
          used_for_size, 0.0);
  if (evolve_lapse_and_shift) {
      dt_lapse_expected = TestHelpers::Ccz4::fd::detail::GaugePlaneWave::
      get_dt_lapse_gauge_plane_wave(
          d_lapse, get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(
                       gauge_plane_wave_vars));
  }
  CHECK_ITERABLE_CUSTOM_APPROX(dt_lapse_actual, dt_lapse_expected,
                               custom_approx);

  // eq 12c
  const auto& dt_shift_actual =
      get<::Tags::dt<gr::Tags::Shift<DataVector, SpatialDim>>>(box);
  const auto& d_shift =
      get<::Tags::deriv<gr::Tags::Shift<DataVector, SpatialDim, FrameType>,
                        tmpl::size_t<SpatialDim>, FrameType>>(
          gauge_plane_wave_vars);
  auto dt_shift_expected = make_with_value<tnsr::I<DataVector, 3>>(
        used_for_size, 0.0);
  if (evolve_lapse_and_shift) {
    dt_shift_expected = TestHelpers::Ccz4::fd::detail::GaugePlaneWave::
        get_dt_shift_gauge_plane_wave(
            get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(
                gauge_plane_wave_vars),
            d_shift);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(dt_shift_actual, dt_shift_expected,
                               custom_approx);

  // eq 12d
  const auto& dt_conformal_factor_actual =
      get<::Tags::dt<::Ccz4::Tags::ConformalFactor<DataVector>>>(box);
  const auto dt_conformal_factor_expected = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_conformal_factor_gauge_plane_wave(
          inverse_spatial_metric, dt_spatial_metric,
          Scalar<DataVector>(conformal_factor));
  CHECK_ITERABLE_CUSTOM_APPROX(dt_conformal_factor_actual,
                               dt_conformal_factor_expected, custom_approx);

  // eq 12e
  const auto& dt_a_tilde_actual =
      get<::Tags::dt<::Ccz4::Tags::ATilde<DataVector, SpatialDim>>>(box);
  const auto dt_conformal_factor = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_conformal_factor_gauge_plane_wave(
          inverse_spatial_metric, dt_spatial_metric,
          Scalar<DataVector>(conformal_factor));
  const auto one_plus_h_times_omega_squared = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_one_plus_h_times_omega_squared(h, omega);
  const auto dt_extrinsic_curvature = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_extrinsic_curvature_gauge_plane_wave(
          k_tnsr, du_h, du_du_h, one_plus_h_times_omega_squared, omega);

  const auto dt_inverse_spatial_metric = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_inverse_spatial_metric(inverse_spatial_metric,
                                                    dt_spatial_metric);
  const auto dt_trace_extrinsic_curvature_expected = TestHelpers::Ccz4::fd::
      detail::GaugePlaneWave::get_dt_trace_extrinsic_curvature_gauge_plane_wave(
          get<gr::Tags::ExtrinsicCurvature<DataVector, SpatialDim>>(
              gauge_plane_wave_vars),
          dt_extrinsic_curvature, inverse_spatial_metric,
          dt_inverse_spatial_metric);
  const auto dt_a_tilde_expected = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_a_tilde_gauge_plane_wave(
          Scalar<DataVector>(conformal_factor), conformal_factor_squared,
          dt_conformal_factor, spatial_metric, dt_spatial_metric,
          get<gr::Tags::ExtrinsicCurvature<DataVector, SpatialDim>>(
              gauge_plane_wave_vars),
          dt_extrinsic_curvature,
          get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars),
          dt_trace_extrinsic_curvature_expected);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_a_tilde_actual, dt_a_tilde_expected,
                               custom_approx);

  // eq 12f
  const auto& dt_trace_extrinsic_curvature_actual =
      get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(box);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_trace_extrinsic_curvature_actual,
                               dt_trace_extrinsic_curvature_expected,
                               custom_approx);

  // eq 12g
  const auto& dt_theta_actual =
      get<::Tags::dt<::Ccz4::Tags::Theta<DataVector>>>(box);
  const auto& dt_theta_expected =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_theta_actual, dt_theta_expected,
                               custom_approx);

  // eq 12h
  const auto& dt_gamma_hat_actual =
      get<::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, SpatialDim>>>(box);
  const auto dt_conformal_spatial_metric = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_conformal_spatial_metric_gauge_plane_wave(
          conformal_factor_squared, spatial_metric, inverse_spatial_metric,
          dt_spatial_metric);
  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(
          get<::Ccz4::Tags::ConformalMetric<DataVector, SpatialDim>>(
              evolved_vars))
          .second;
  const auto dt_inverse_conformal_spatial_metric = TestHelpers::Ccz4::fd::
      detail::GaugePlaneWave::get_dt_inverse_conformal_spatial_metric(
          inverse_conformal_spatial_metric, dt_conformal_spatial_metric);
  const auto dt_d_spatial_metric =
      TestHelpers::Ccz4::fd::detail::GaugePlaneWave::
          get_dt_d_spatial_metric_gauge_plane_wave(k_tnsr, du_du_h, omega);
  const auto& d_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim>,
                        tmpl::size_t<SpatialDim>, FrameType>>(
          gauge_plane_wave_vars);
  const auto d_conformal_factor = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_d_conformal_factor_gauge_plane_wave(
          inverse_spatial_metric, d_spatial_metric,
          Scalar<DataVector>(conformal_factor));
  const auto dt_d_conformal_factor = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_d_conformal_factor_gauge_plane_wave(
          Scalar<DataVector>(conformal_factor), dt_conformal_factor,
          inverse_spatial_metric, dt_inverse_spatial_metric, d_spatial_metric,
          dt_d_spatial_metric);
  const auto dt_d_conformal_spatial_metric = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_d_conformal_spatial_metric_gauge_plane_wave(
          spatial_metric, dt_spatial_metric, d_spatial_metric,
          dt_d_spatial_metric, Scalar<DataVector>(conformal_factor),
          dt_conformal_factor, d_conformal_factor, dt_d_conformal_factor);
  const auto det_spatial_metric = determinant(spatial_metric);
  const auto d_det_spatial_metric = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_d_det_spatial_metric_gauge_plane_wave(
          det_spatial_metric,
          get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim,
                                             FrameType>>(gauge_plane_wave_vars),
          get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim>,
                            tmpl::size_t<SpatialDim>, FrameType>>(
              gauge_plane_wave_vars));
  const tnsr::ijj<DataVector, SpatialDim, FrameType>
      d_conformal_spatial_metric = TestHelpers::Ccz4::fd::detail::KerrSchild::
          get_d_conformal_spatial_metric(conformal_factor_squared,
                                         spatial_metric, d_spatial_metric,
                                         d_det_spatial_metric);
  const auto dt_gamma_hat_expected = TestHelpers::Ccz4::fd::detail::
      GaugePlaneWave::get_dt_gamma_hat_gauge_plane_wave(
          inverse_conformal_spatial_metric, dt_inverse_conformal_spatial_metric,
          d_conformal_spatial_metric, dt_d_conformal_spatial_metric);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_gamma_hat_actual, dt_gamma_hat_expected,
                               custom_approx);

  // eq 12i
  auto dt_b_expected =
        make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.0);
  if (evolve_lapse_and_shift) {
    const tnsr::ijj<DataVector, SpatialDim, FrameType> field_d =
        TestHelpers::Ccz4::fd::detail::KerrSchild::get_field_d(
            d_conformal_spatial_metric);
    const tnsr::iJJ<DataVector, SpatialDim, FrameType> field_d_up =
        TestHelpers::Ccz4::fd::detail::GaugePlaneWave::get_field_d_up(
            inverse_conformal_spatial_metric, field_d);
    const auto d_field_d = partial_derivative(
        field_d, subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);
    const auto d_conformal_christoffel_second_kind =
        ::Ccz4::deriv_conformal_christoffel_second_kind(
            inverse_conformal_spatial_metric, field_d, d_field_d, field_d_up);
    const auto conformal_christoffel_second_kind =
        ::Ccz4::conformal_christoffel_second_kind(
            inverse_conformal_spatial_metric, field_d);
    const auto d_gamma_hat =
        ::Ccz4::deriv_contracted_conformal_christoffel_second_kind(
            inverse_conformal_spatial_metric, field_d_up,
            conformal_christoffel_second_kind,
            d_conformal_christoffel_second_kind);
    dt_b_expected = TestHelpers::
        Ccz4::fd::detail::GaugePlaneWave::get_dt_b_gauge_plane_wave_expected(
            dt_gamma_hat_expected, d_gamma_hat,
            get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(
                gauge_plane_wave_vars));
  }
  const auto& dt_b_actual =
      get<::Tags::dt<::Ccz4::Tags::AuxiliaryShiftB<DataVector, SpatialDim>>>(
          box);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_b_actual, dt_b_expected, custom_approx);
}

void test_sommerfeld_bc(const bool evolve_lapse_and_shift) {
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 5;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};

  const std::array<double, SpatialDim> lower_bound{0.8, 0.7, -2.1};
  const std::array<double, SpatialDim> upper_bound{1.0, 1.5, -0.9};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;

  // Create an element with an external boundary by omitting the last neighbor
  // in the upper_zeta direction.
  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element(true);

  // we cannot declare element_map const because it is not copyable into the box
  // NOLINTNEXTLINE(misc-const-correctness)
  ElementMap element_map{
      element.id(),
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
          Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]},
                   Affine{-1., 1., lower_bound[2], upper_bound[2]}})
          .get_clone()};

  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, FrameType>(
          domain::CoordinateMaps::Identity<3>{});

  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = grid_to_inertial_map(element_map(logical_coords));

  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  // Ghost data from interior neighbors (none for the external face, which is
  // set by BCs)
  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::Minkowski::
                  compute_prim_solution_for_Minkowski<false>,
              coords_range);

  // Minkowski evolved variables
  auto evolved_vars = TestHelpers::Ccz4::fd::detail::Minkowski::
      compute_prim_solution_for_Minkowski(x);

  const DataVector radial_coords =
      sqrt(square(get<0>(x)) + square(get<1>(x)) + square(get<2>(x)));

  const DataVector used_for_size(subcell_mesh.number_of_grid_points(),
                                 std::numeric_limits<double>::signaling_NaN());
  // set dummy k_0 value to get non-trivial lapse evolution for testing
  const auto k_0 = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  // Provide BC on the external face (upper_zeta)
  std::vector<DirectionMap<
      SpatialDim,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_bcs_per_block(1);
  external_bcs_per_block[0][Direction<SpatialDim>::upper_zeta()] =
      std::make_unique<Ccz4::BoundaryConditions::Sommerfeld>();

  // NOLINTNEXTLINE(misc-const-correctness)
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  auto box = db::create<db::AddSimpleTags<
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<SpatialDim>,
      fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<true>>,
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>, dt_variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>,
      evolution::dg::subcell::Tags::Coordinates<SpatialDim, Frame::Inertial>,
      ::Tags::Time,
      domain::Tags::FunctionsOfTime,
      domain::Tags::ElementMap<SpatialDim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<SpatialDim, Frame::Grid,
                                                  Frame::Inertial>>>(
      kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<true>{}, evolved_vars, eta, k_0,
      upper_spatial_z4_constraint,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
      all_ghost_data, std::move(external_bcs_per_block), x, 0.0,
      std::move(functions_of_time), std::move(element_map),
      grid_to_inertial_map.get_clone());

  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));

  const size_t num_pts = subcell_mesh.number_of_grid_points();
  const size_t num_face_pts = subcell_mesh.extents(0) * subcell_mesh.extents(1);

  // Compute r on the upper zeta slice
  const size_t num_unaffected_pts = num_pts - num_face_pts;
  DataVector r_affected{};
  make_const_view<DataVector>(make_not_null(&r_affected), radial_coords,
                              num_unaffected_pts, num_face_pts);

  {
    // Points not in upper zeta slice should be unaffected (zero dt for
    // Minkowski)
    const auto unaffected_expected = DataVector(num_pts - num_face_pts, 0.0);
    const auto& dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(box);

    DataVector unaffected_actual{};
    make_const_view<DataVector>(make_not_null(&unaffected_actual),
                                get(dt_lapse), 0, num_unaffected_pts);
    CHECK_ITERABLE_APPROX(unaffected_actual, unaffected_expected);

    DataVector affected_actual{};
    make_const_view<DataVector>(make_not_null(&affected_actual), get(dt_lapse),
                                num_unaffected_pts, num_face_pts);
    // Points in upper zeta slice has expected dt = -1/r for lapse in Minkowski
    const DataVector affected_expected = -1.0 / r_affected;
    CHECK_ITERABLE_APPROX(affected_actual, affected_expected);
  }

  {
    // Points not in upper zeta slice should be unaffected (zero dt for
    // Minkowski)
    const auto unaffected_expected = DataVector(num_pts - num_face_pts, 0.0);
    const auto& dt_conformal_factor =
        get<::Tags::dt<::Ccz4::Tags::ConformalFactor<DataVector>>>(box);

    DataVector unaffected_actual{};
    make_const_view<DataVector>(make_not_null(&unaffected_actual),
                                get(dt_conformal_factor), 0,
                                num_unaffected_pts);
    CHECK_ITERABLE_APPROX(unaffected_actual, unaffected_expected);

    DataVector affected_actual{};
    make_const_view<DataVector>(make_not_null(&affected_actual),
                                get(dt_conformal_factor), num_unaffected_pts,
                                num_face_pts);
    // Points in upper zeta slice has expected dt = -1/r for conformal factor in
    // Minkowski
    const DataVector affected_expected = -1.0 / r_affected;
    CHECK_ITERABLE_APPROX(affected_actual, affected_expected);
  }

  {
    // Points not in upper zeta slice should be unaffected (zero dt for
    // Minkowski)
    const auto unaffected_expected = DataVector(num_pts - num_face_pts, 0.0);
    const auto& dt_conformal_metric =
        get<::Tags::dt<::Ccz4::Tags::ConformalMetric<DataVector, SpatialDim>>>(
            box);
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        DataVector unaffected_actual{};
        make_const_view<DataVector>(make_not_null(&unaffected_actual),
                                    dt_conformal_metric.get(i, j), 0,
                                    num_unaffected_pts);
        CHECK_ITERABLE_APPROX(unaffected_actual, unaffected_expected);

        DataVector affected_actual{};
        make_const_view<DataVector>(make_not_null(&affected_actual),
                                    dt_conformal_metric.get(i, j),
                                    num_unaffected_pts, num_face_pts);
        // Points in upper zeta slice has expected dt = -1/r for diagonal in
        // Minkowski
        const DataVector affected_expected =
            (i == j) ? (-1.0 / r_affected) : DataVector(num_face_pts, 0.0);
        CHECK_ITERABLE_APPROX(affected_actual, affected_expected);
      }
    }
  }

  // all other evolved variables must have zero dt everywhere
  {
    const Approx custom_approx = Approx::custom().epsilon(1.0e-13).scale(1.0);
    const DataVector all_expected{num_pts, 0.0};
    tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          // Skip variables already checked above: lapse, conformal factor,
          // conformal metric
          if constexpr (std::is_same_v<Tag, gr::Tags::Lapse<DataVector>> ||
                        std::is_same_v<
                            Tag, ::Ccz4::Tags::ConformalFactor<DataVector>> ||
                        std::is_same_v<Tag, ::Ccz4::Tags::ConformalMetric<
                                                DataVector, SpatialDim>>) {
            return;
          }
          const std::string tag_name = db::tag_name<::Tags::dt<Tag>>();
          CAPTURE(tag_name);
          for (auto& component : get<::Tags::dt<Tag>>(box)) {
            // dt_theta and dt_trace_extrinsic_curvature are slightly non-zero
            // which is likely due to round-off error.
            CHECK_ITERABLE_CUSTOM_APPROX(component, all_expected,
                                         custom_approx);
          }
        });
  }

  // now change the lapse to some dummy values to test the radial deriv term in
  // Sommerfeld BC
  db::mutate<
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::K0<DataVector>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>>(
      [&](const auto evolved_var_ptr, const auto k_0_ptr,
          const auto all_ghost_data_ptr) {
        get(get<gr::Tags::Lapse<DataVector>>(*evolved_var_ptr)) = get<0>(x);
        get(*k_0_ptr) = DataVector(num_pts, 1.0);
        *all_ghost_data_ptr =
            TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
                subcell_mesh, x, element.neighbors(), ghost_zone_size,
                TestHelpers::Ccz4::fd::detail::Minkowski::
                    compute_prim_solution_for_Minkowski<true>,
                coords_range);
      },
      make_not_null(&box));

  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));

  {
    // Points not in upper zeta slice should be unaffected (value dependent on
    // evolve_lapse_and_shift)
    DataVector dt_lapse_expected = -2.0 * get<0>(x) / radial_coords;
    for (size_t i = 0; i < num_unaffected_pts; ++i) {
      dt_lapse_expected[i] = evolve_lapse_and_shift ? 2.0 * get<0>(x)[i] : 0.0;
    }
    DataVector dt_lapse_unaffected_expected{};
    make_const_view<DataVector>(make_not_null(&dt_lapse_unaffected_expected),
                                dt_lapse_expected, 0, num_unaffected_pts);

    const auto& dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(box);

    DataVector unaffected_actual{};
    make_const_view<DataVector>(make_not_null(&unaffected_actual),
                                get(dt_lapse), 0, num_unaffected_pts);
    CHECK_ITERABLE_APPROX(unaffected_actual, dt_lapse_unaffected_expected);

    DataVector affected_actual{};
    make_const_view<DataVector>(make_not_null(&affected_actual), get(dt_lapse),
                                num_unaffected_pts, num_face_pts);
    // Points in upper zeta slice has expected dt = -2*x/r for dummy lapse in
    // Minkowski
    DataVector dt_lapse_affected_expected{};
    make_const_view<DataVector>(make_not_null(&dt_lapse_affected_expected),
                                dt_lapse_expected, num_unaffected_pts,
                                num_face_pts);
    CHECK_ITERABLE_APPROX(affected_actual, dt_lapse_affected_expected);
  }
}

void test_dirichlet_analytic_bc(const bool evolve_lapse_and_shift) {
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const size_t points_per_dimension = 5;
  const Ccz4::fd::DummyReconstructor recons{};
  const size_t ghost_zone_size = recons.ghost_zone_size();
  const Mesh<SpatialDim> subcell_mesh{points_per_dimension,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered};

  const std::array<double, SpatialDim> lower_bound{0.8, 0.7, -2.1};
  const std::array<double, SpatialDim> upper_bound{1.0, 1.5, -0.9};
  const std::array<double, SpatialDim> coords_range = upper_bound - lower_bound;

  // Create an element with an external boundary by omitting the last neighbor
  // in the upper_zeta direction.
  const Element<SpatialDim> element =
      TestHelpers::Ccz4::fd::detail::set_element(true);

  // we cannot declare element_map const because it is not copyable into the box
  // NOLINTNEXTLINE(misc-const-correctness)
  ElementMap element_map{
      element.id(),
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
          Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]},
                   Affine{-1., 1., lower_bound[2], upper_bound[2]}})
          .get_clone()};

  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, FrameType>(
          domain::CoordinateMaps::Identity<3>{});

  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = grid_to_inertial_map(element_map(logical_coords));

  InverseJacobian<DataVector, SpatialDim, Frame::ElementLogical,
                  Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < SpatialDim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  // Ghost data from interior neighbors (none for the external face, which is
  // set by BCs)
  const DirectionalIdMap<SpatialDim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::Minkowski::
                  compute_prim_solution_for_Minkowski<false>,
              coords_range);

  // Minkowski evolved variables
  auto evolved_vars = TestHelpers::Ccz4::fd::detail::Minkowski::
      compute_prim_solution_for_Minkowski(x);

  const DataVector radial_coords =
      sqrt(square(get<0>(x)) + square(get<1>(x)) + square(get<2>(x)));

  const DataVector used_for_size(subcell_mesh.number_of_grid_points(),
                                 std::numeric_limits<double>::signaling_NaN());
  // set dummy k_0 value to get non-trivial lapse evolution for testing
  const auto k_0 = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto upper_spatial_z4_constraint =
      make_with_value<tnsr::I<DataVector, 3>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());

  const double kappa_1 = 0.1;
  const double kappa_2 = 0.2;
  const double kappa_3 = 0.3;

  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, Ccz4::fd::System::variables_tag>;

  // Provide BC on the external face (upper_zeta)
  std::vector<DirectionMap<
      SpatialDim,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_bcs_per_block(1);
  external_bcs_per_block[0][Direction<SpatialDim>::upper_zeta()] =
      std::make_unique<Ccz4::BoundaryConditions::DirichletAnalytic>(
          std::make_unique<
              Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::Minkowski<3>>>(
              Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::Minkowski<3>>{}));

  // NOLINTNEXTLINE(misc-const-correctness)
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  auto box = db::create<db::AddSimpleTags<
      ::Ccz4::Tags::Kappa1, ::Ccz4::Tags::Kappa2, ::Ccz4::Tags::Kappa3,
      ::Ccz4::fd::Tags::EvolveLapseAndShift, domain::Tags::Element<SpatialDim>,
      fd::Tags::Reconstructor,
      Parallel::Tags::MetavariablesImpl<DummyEvolutionMetaVars<true>>,
      Ccz4::fd::System::variables_tag, ::Ccz4::Tags::Eta<DataVector>,
      ::Ccz4::Tags::K0<DataVector>,
      ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, 3>, dt_variables_tag,
      evolution::dg::subcell::Tags::Mesh<SpatialDim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<
          SpatialDim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<SpatialDim>,
      domain::Tags::ExternalBoundaryConditions<SpatialDim>,
      evolution::dg::subcell::Tags::Coordinates<SpatialDim, Frame::Inertial>,
      ::Tags::Time,
      domain::Tags::FunctionsOfTime,
      domain::Tags::ElementMap<SpatialDim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<SpatialDim, Frame::Grid,
                                                  Frame::Inertial>>>(
      kappa_1, kappa_2, kappa_3, evolve_lapse_and_shift, element,
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<std::decay_t<decltype(recons)>>(recons)},
      DummyEvolutionMetaVars<true>{}, evolved_vars, eta, k_0,
      upper_spatial_z4_constraint,
      Variables<typename dt_variables_tag::tags_list>{
          subcell_mesh.number_of_grid_points()},
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian,
      all_ghost_data, std::move(external_bcs_per_block), x, 0.0,
      std::move(functions_of_time), std::move(element_map),
      grid_to_inertial_map.get_clone());

  ::Ccz4::fd::SoTimeDerivative::apply(make_not_null(&box));

  {
    const size_t num_pts = subcell_mesh.number_of_grid_points();
    const Approx custom_approx = Approx::custom().epsilon(1.0e-13).scale(1.0);
    const DataVector all_expected{num_pts, 0.0};
    tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          const std::string tag_name = db::tag_name<::Tags::dt<Tag>>();
          CAPTURE(tag_name);
          for (auto& component : get<::Tags::dt<Tag>>(box)) {
            // dt_theta and dt_trace_extrinsic_curvature are slightly non-zero
            // which is likely due to round-off error.
            CHECK_ITERABLE_CUSTOM_APPROX(component, all_expected,
                                         custom_approx);
          }
        });
  }
}

void test() {
  test_minkowski(true);
  test_kerrschild(true);
  test_minkowski(false);
  test_kerrschild(false);

  const std::array<double, 3> k{{0.5, 0.1, -0.2}};
  test_gauge_plane_wave(
      k,
      std::make_unique<MathFunctions::Sinusoid<1, Frame::Inertial>>(0.6, 0.8,
                                                                    2.0),
      0.4, true);
  test_gauge_plane_wave(
      k,
      std::make_unique<MathFunctions::Sinusoid<1, Frame::Inertial>>(0.6, 0.8,
                                                                    2.0),
      0.4, false);

  // Run Diridchlet BC test
  test_dirichlet_analytic_bc(true);
  test_dirichlet_analytic_bc(false);
  // Run Sommerfeld BC test
  test_sommerfeld_bc(true);
  test_sommerfeld_bc(false);
}
}  // namespace

// The tests run relatively long as we use much higher spatial
// resolution (~8000 grid points per element) to reach a relative
// error of 1e-9.
// [[TimeOut, 40]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.FiniteDifference.TimeDerivative",
                  "[Unit][Evolution]") {
  test();
}
}  // namespace Ccz4::fd

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/ElementMap.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/DummyReconstructor.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Evolution/Systems/Ccz4/Ccz4WrappedGr.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Ccz4::BoundaryConditions::BoundaryCondition,
                   tmpl::list<Ccz4::BoundaryConditions::DirichletAnalytic>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   Ccz4::Solutions::all_solutions>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>>;
  };
};

template <typename T, typename U>
void test_fd(const U& boundary_condition, const T& analytic_solution_or_data) {
  const double time = 1.3;
  const Mesh<3> subcell_mesh{9, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};

  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  const std::array<double, 3> lower_bound{{0.78, 1.18, 1.28}};
  const std::array<double, 3> upper_bound{{0.82, 1.22, 1.32}};
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          Affine3D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]},
                   Affine{-1., 1., lower_bound[2], upper_bound[2]}});

  const ElementId<3> element_id{0};
  const ElementMap logical_to_grid_map{
      element_id,
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(
          Affine3D{Affine{-1., 1., 2.0 * lower_bound[0], 2.0 * upper_bound[0]},
                   Affine{-1., 1., 2.0 * lower_bound[1], 2.0 * upper_bound[1]},
                   Affine{-1., 1., 2.0 * lower_bound[2], 2.0 * upper_bound[2]}})
          .get_clone()};
  const auto direction = Direction<3>::lower_xi();

  const Ccz4::fd::DummyReconstructor reconstructor{};
  const size_t ghost_zone_size = reconstructor.ghost_zone_size();

  using Vars = Variables<Ccz4::fd::Tags::spacetime_reconstruction_tags>;
  Vars vars{ghost_zone_size * subcell_mesh.extents().slice_away(0).product()};
  const auto expected_vars = [&analytic_solution_or_data, &direction,
                              &functions_of_time, &grid_to_inertial_map,
                              &logical_to_grid_map, &subcell_mesh, time,
                              ghost_zone_size]() {
    const auto ghost_logical_coords =
        evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
            subcell_mesh, ghost_zone_size, direction);

    const auto ghost_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(ghost_logical_coords), time, functions_of_time);

    using tags =
        tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                   gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                   ::Ccz4::Tags::ConformalFactor<DataVector>,
                   ::Ccz4::Tags::ATilde<DataVector, 3>,
                   gr::Tags::TraceExtrinsicCurvature<DataVector>,
                   ::Ccz4::Tags::Theta<DataVector>,
                   ::Ccz4::Tags::GammaHat<DataVector, 3>,
                   ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>;

    tuples::tagged_tuple_from_typelist<tags> analytic_vars{};

    if constexpr (::is_analytic_solution_v<T>) {
      analytic_vars = analytic_solution_or_data.variables(ghost_inertial_coords,
                                                          time, tags{});
    } else {
      (void)time;
      analytic_vars =
          analytic_solution_or_data.variables(ghost_inertial_coords, tags{});
    }

    Vars expected{get<0>(ghost_inertial_coords).size()};

    get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(expected) =
        get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(analytic_vars);
    get<gr::Tags::Lapse<DataVector>>(expected) =
        get<gr::Tags::Lapse<DataVector>>(analytic_vars);
    get<gr::Tags::Shift<DataVector, 3>>(expected) =
        get<gr::Tags::Shift<DataVector, 3>>(analytic_vars);
    get<::Ccz4::Tags::ConformalFactor<DataVector>>(expected) =
        get<::Ccz4::Tags::ConformalFactor<DataVector>>(analytic_vars);
    get<::Ccz4::Tags::ATilde<DataVector, 3>>(expected) =
        get<::Ccz4::Tags::ATilde<DataVector, 3>>(analytic_vars);
    get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(expected) =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(analytic_vars);
    get<::Ccz4::Tags::Theta<DataVector>>(expected) =
        get<::Ccz4::Tags::Theta<DataVector>>(analytic_vars);
    get<::Ccz4::Tags::GammaHat<DataVector, 3>>(expected) =
        get<::Ccz4::Tags::GammaHat<DataVector, 3>>(analytic_vars);
    get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(expected) =
        get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(analytic_vars);
    return expected;
  }();
  auto& [conformal_metric, conformal_factor, a_tilde, trace_extrinsic_curvature,
         theta, gamma_hat, lapse, shift, auxiliary_shift_b] = vars;

  boundary_condition.fd_ghost(
      make_not_null(&conformal_metric), make_not_null(&lapse),
      make_not_null(&shift), make_not_null(&conformal_factor),
      make_not_null(&a_tilde), make_not_null(&trace_extrinsic_curvature),
      make_not_null(&theta), make_not_null(&gamma_hat),
      make_not_null(&auxiliary_shift_b), direction, subcell_mesh, time,
      functions_of_time, logical_to_grid_map, grid_to_inertial_map,
      reconstructor);
  // failing line
  CHECK(vars == expected_vars);
}

SPECTRE_TEST_CASE("Unit.Ccz4.BoundaryConditions.DirichletAnalytic",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  register_factory_classes_with_charm<Metavariables>();
  {
    INFO("Test with analytic solution");
    const auto product_boundary_condition =
        TestHelpers::test_creation<
            std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
            Metavariables>(
            "DirichletAnalytic:\n"
            "  AnalyticPrescription:\n"
            "      Ccz4(KerrSchild):\n"
            "        Mass: 1.0\n"
            "        Spin: [0.5, -0.2, 0.0]\n"
            "        Center: [-0.2, 0.0, 0.3]\n"
            "        Velocity: [0.0, 0.0, 0.0]\n")
            ->get_clone();

    const Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>
        analytic_solution_or_data{1.0,
            {0.5, -0.2, 0.0}, {-0.2, 0.0, 0.3}, {0.0, 0.0, 0.0}};
    const auto serialized_and_deserialized_condition =
        serialize_and_deserialize(
            *dynamic_cast<Ccz4::BoundaryConditions::DirichletAnalytic*>(
                product_boundary_condition.get()));

    test_fd<Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::KerrSchild>,
            Ccz4::BoundaryConditions::DirichletAnalytic>(
        serialized_and_deserialized_condition, analytic_solution_or_data);
  }
}
}  // namespace

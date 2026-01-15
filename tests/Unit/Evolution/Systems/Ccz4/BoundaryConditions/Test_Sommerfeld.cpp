// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

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
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Sommerfeld.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/DummyReconstructor.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
using Vars = Variables<Ccz4::fd::Tags::spacetime_reconstruction_tags>;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Ccz4::BoundaryConditions::BoundaryCondition,
                             tmpl::list<Ccz4::BoundaryConditions::Sommerfeld>>>;
  };
};

Vars set_polynomial(const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
                    const size_t max_degree) {
  DataVector result_vector{get<0>(coords).size(), 0.0};
  for (size_t degree_x = 0; degree_x <= max_degree; ++degree_x) {
    for (size_t degree_y = 0; degree_y <= max_degree - degree_x; ++degree_y) {
      for (size_t degree_z = 0; degree_z <= max_degree - degree_x - degree_y;
           ++degree_z) {
        result_vector += pow(get<0>(coords), degree_x) *
                         pow(get<1>(coords), degree_y) *
                         pow(get<2>(coords), degree_z);
      }
    }
  }
  Vars vars{get<0>(coords).size()};

  tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        for (auto& component : get<Tag>(vars)) {
          component = result_vector;
        }
      });

  return vars;
}

template <typename U>
void test_fd(const U& boundary_condition, const size_t max_degree) {
  const Mesh<3> subcell_mesh{20, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};

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

  const auto expected_vars = [&]() {
    const auto ghost_logical_coords =
        evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
            subcell_mesh, ghost_zone_size, direction);

    const auto ghost_inertial_coords =
        grid_to_inertial_map(logical_to_grid_map(ghost_logical_coords));

    const auto analytic_vars =
        set_polynomial(ghost_inertial_coords, max_degree);

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

  Vars vars{ghost_zone_size * subcell_mesh.extents().slice_away(0).product()};
  auto& [conformal_metric, conformal_factor, a_tilde, trace_extrinsic_curvature,
         theta, gamma_hat, lapse, shift, auxiliary_shift_b] = vars;

  const auto interior_inertial_coords = grid_to_inertial_map(
      logical_to_grid_map(logical_coordinates(subcell_mesh)));
  const Vars interior_vars =
      set_polynomial(interior_inertial_coords, max_degree);
  auto& [int_conformal_metric, int_conformal_factor, int_a_tilde,
         int_trace_extrinsic_curvature, int_theta, int_gamma_hat, int_lapse,
         int_shift, int_auxiliary_shift_b] = interior_vars;

  boundary_condition.fd_ghost(
      make_not_null(&conformal_metric), make_not_null(&lapse),
      make_not_null(&shift), make_not_null(&conformal_factor),
      make_not_null(&a_tilde), make_not_null(&trace_extrinsic_curvature),
      make_not_null(&theta), make_not_null(&gamma_hat),
      make_not_null(&auxiliary_shift_b), direction,
      // interior args in variables_tag_list order
      int_conformal_metric, int_conformal_factor, int_a_tilde,
      int_trace_extrinsic_curvature, int_theta, int_gamma_hat, int_lapse,
      int_shift, int_auxiliary_shift_b, subcell_mesh, reconstructor);

  tmpl::for_each<Ccz4::fd::System::variables_tag_list>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<Tag>();
        CAPTURE(tag_name);
        for (auto expected_component = get<Tag>(expected_vars).cbegin(),
                  component = get<Tag>(vars).cbegin();
             expected_component != get<Tag>(expected_vars).cend() and
             component != get<Tag>(vars).cend();
             ++expected_component, ++component) {
          CHECK_ITERABLE_APPROX(*component, *expected_component);
        }
      });
}

SPECTRE_TEST_CASE("Unit.Ccz4.BoundaryConditions.Sommerfeld",
                  "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();
  {
    INFO("Test Sommerfeld BC");
    const auto product_boundary_condition =
        TestHelpers::test_creation<
            std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
            Metavariables>("Sommerfeld:\n")
            ->get_clone();

    const auto serialized_and_deserialized_condition =
        serialize_and_deserialize(
            *dynamic_cast<Ccz4::BoundaryConditions::Sommerfeld*>(
                product_boundary_condition.get()));

    test_fd<Ccz4::BoundaryConditions::Sommerfeld>(
        serialized_and_deserialized_condition, 1);
  }
}
}  // namespace

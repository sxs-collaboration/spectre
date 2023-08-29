// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/InitialMagneticField.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/Poloidal.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::AnalyticData::InitialMagneticFields {
namespace {

struct PoloidalProxy : Poloidal {
  using Poloidal::Poloidal;

  template <typename DataType>
  tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>> return_variables(
      const tnsr::I<DataType, 3>& x, const Scalar<DataType>& pressure,
      const Scalar<DataType>& sqrt_det_spatial_metric,
      const tnsr::i<DataType, 3>& deriv_pressure) const {
    return this->variables(x, pressure, sqrt_det_spatial_metric,
                           deriv_pressure);
  }
};

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.GrMhd.InitialMagneticFields.Poloidal",
    "[Unit][PointwiseFunctions]") {
  // test creation
  const auto solution = TestHelpers::test_creation<Poloidal>(
      "  PressureExponent: 2\n"
      "  CutoffPressure: 1.0e-5\n"
      "  VectorPotentialAmplitude: 2500.0\n");
  CHECK(solution == Poloidal(2, 1.0e-5, 2500.0));

  // test serialize
  test_serialization(solution);

  // test move
  {
    Poloidal poloidal_field{2, 1.0e-5, 2500.0};
    Poloidal poloidal_field_copy{2, 1.0e-5, 2500.0};
    test_move_semantics(std::move(poloidal_field), poloidal_field_copy);
  }

  // test derived
  register_classes_with_charm<Poloidal>();
  const std::unique_ptr<InitialMagneticField> base_ptr =
      std::make_unique<Poloidal>();
  const std::unique_ptr<InitialMagneticField> deserialized_base_ptr =
      serialize_and_deserialize(base_ptr)->get_clone();
  CHECK(dynamic_cast<Poloidal*>(deserialized_base_ptr.get()) != nullptr);

  // test equality
  const Poloidal field_original{2, 1.0e-5, 2500.0};
  const auto field = serialize_and_deserialize(field_original);
  CHECK(field == Poloidal(2, 1.0e-5, 2500.0));
  CHECK(field != Poloidal(3, 1.0e-5, 2500.0));
  CHECK(field != Poloidal(2, 2.0e-5, 2500.0));
  CHECK(field != Poloidal(2, 1.0e-5, 3500.0));

  // test solution implementation
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields"};
  const DataVector used_for_size{10};

  const size_t pressure_exponent = 2;
  const double cutoff_pressure = 1.0e-5;
  const double vector_potential_amplitude = 2500.0;

  pypp::check_with_random_values<1>(
      &PoloidalProxy::return_variables<double>,
      PoloidalProxy(pressure_exponent, cutoff_pressure,
                    vector_potential_amplitude),
      "Poloidal", {"magnetic_field"}, {{{-10.0, 10.0}}},
      std::make_tuple(pressure_exponent, cutoff_pressure,
                      vector_potential_amplitude),
      used_for_size);

  // test if the B field is divergence-free in the flat spacetime
  const Mesh<3> mesh{5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t num_grid_pts = mesh.number_of_grid_points();
  const auto log_coords = logical_coordinates(mesh);
  const double scale = 1.0e-2;
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inv_jac{mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    inv_jac.get(i, i) = 1.0 / scale;
  }
  const Scalar<DataVector> sqrt_det_spatial_metric{mesh.number_of_grid_points(),
                                                   1.0};

  const auto test_for_small_coords_patch =
      [&inv_jac, &solution, &sqrt_det_spatial_metric, &mesh,
       &num_grid_pts](const tnsr::I<DataVector, 3>& in_coords) {
        const auto& x = in_coords.get(0);
        const auto& y = in_coords.get(1);
        const auto& z = in_coords.get(2);

        // test with the pressure of a simple analytic form
        Scalar<DataVector> pressure{num_grid_pts, 0.0};
        get(pressure) = (x * x * x) + (y * y) + z +
                        0.1;  // Add a small offset to set P > cutoff

        tnsr::i<DataVector, 3, Frame::Inertial> d_pressure{num_grid_pts, 0.0};
        d_pressure.get(0) = 3.0 * x * x;
        d_pressure.get(1) = 2.0 * y;
        d_pressure.get(2) = 1.0;

        const auto b_field =
            get<hydro::Tags::MagneticField<DataVector, 3>>(solution.variables(
                in_coords, pressure, sqrt_det_spatial_metric, d_pressure));

        Scalar<DataVector> mag_b_field{num_grid_pts, 0.0};
        for (size_t i = 0; i < 3; ++i) {
          get(mag_b_field) += square(b_field.get(i));
        }
        get(mag_b_field) = sqrt(get(mag_b_field));

        CHECK(get(mag_b_field) != approx(0.));
        const auto div_b_field = divergence(b_field, mesh, inv_jac);

        const Scalar<DataVector> div_b_over_mag_b{get(div_b_field) /
                                                  get(mag_b_field)};

        CHECK(max(abs(get(div_b_over_mag_b))) < 1.0e-12);
      };

  // check a small region around the origin
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{num_grid_pts, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    inertial_coords.get(i) = scale * log_coords.get(i);
  }
  test_for_small_coords_patch(inertial_coords);

  // check a small region off-origin
  inertial_coords.get(0) += 0.5;
  inertial_coords.get(1) += 1.0;
  inertial_coords.get(2) += 2.0;
  test_for_small_coords_patch(inertial_coords);
}
}  // namespace
}  // namespace grmhd::AnalyticData::InitialMagneticFields

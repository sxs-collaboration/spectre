// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/HydroFreeOutflow.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = ::TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.GrMhd.BoundaryConditions.HydroFreeOutflow",
                  "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  MAKE_GENERATOR(gen);

  const auto face_mesh_index = Index<2>{2};
  DataVector used_for_size{face_mesh_index.product()};
  std::uniform_real_distribution<> dist(0.5, 1.0);

  const auto spatial_metric =
      ::TestHelpers::gr::random_spatial_metric<3, DataVector, Frame::Inertial>(
          make_not_null(&gen), used_for_size);
  const auto sqrt_det_spatial_metric = determinant(spatial_metric);

  const auto box_with_gridless_tags =
      db::create<db::AddSimpleTags<gr::Tags::SpatialMetric<DataVector, 3>,
                                   gr::Tags::SqrtDetSpatialMetric<DataVector>>>(
          spatial_metric, sqrt_det_spatial_metric);

  helpers::test_boundary_condition_with_python<
      grmhd::ValenciaDivClean::BoundaryConditions::HydroFreeOutflow,
      grmhd::ValenciaDivClean::BoundaryConditions::BoundaryCondition,
      grmhd::ValenciaDivClean::System,
      tmpl::list<grmhd::ValenciaDivClean::BoundaryCorrections::Rusanov>>(
      make_not_null(&gen),
      "Evolution.Systems.GrMhd.ValenciaDivClean.BoundaryConditions."
      "HydroFreeOutflow",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeD>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeYe>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeTau>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              grmhd::ValenciaDivClean::Tags::TildePhi>,

          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD,
                           tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeYe,
                           tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau,
                           tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>,
              tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>,
              tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi,
                           tmpl::size_t<3>, Frame::Inertial>>,

          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<DataVector>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Shift<DataVector, 3>>,
          helpers::Tags::PythonFunctionName<hydro::Tags::SpatialVelocityOneForm<
              DataVector, 3, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::RestMassDensity<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::ElectronFraction<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::Temperature<DataVector>>,
          helpers::Tags::PythonFunctionName<
              hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>,

          helpers::Tags::PythonFunctionName<
              gr::Tags::InverseSpatialMetric<DataVector, 3>>>{
          "error",
          "tilde_d",
          "tilde_ye",
          "tilde_tau",
          "tilde_s",
          "tilde_b",
          "tilde_phi",
          "flux_tilde_d",
          "flux_tilde_ye",
          "flux_tilde_tau",
          "flux_tilde_s",
          "flux_tilde_b",
          "flux_tilde_phi",
          "lapse",
          "shift",
          "spatial_velocity_one_form",
          "rest_mass_density",
          "electron_fraction",
          "temperature",
          "spatial_velocity",
          "inv_spatial_metric"},
      "HydroFreeOutflow:\n", face_mesh_index, box_with_gridless_tags,
      tuples::TaggedTuple<>{}, 1.0e-10);
}

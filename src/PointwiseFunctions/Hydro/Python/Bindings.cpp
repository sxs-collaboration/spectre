// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/LowerSpatialFourVelocity.hpp"
#include "PointwiseFunctions/Hydro/MassFlux.hpp"
#include "PointwiseFunctions/Hydro/MassWeightedFluidItems.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace {
template <typename DataType>
void bind_comoving_magnetic_field_impl(py::module& m) {
  // Wrapper for calculating Co-moving magnetic fields
  m.def("comoving_magnetic_field_one_form",
        static_cast<tnsr::a<DataType, 3> (*)(
            const tnsr::i<DataType, 3>&, const tnsr::i<DataType, 3>&,
            const Scalar<DataType>&, const Scalar<DataType>&,
            const tnsr::I<DataType, 3>&, const Scalar<DataType>&)>(
            &hydro::comoving_magnetic_field_one_form<DataType>),
        py::arg("spatial_velocity_one_form"),
        py::arg("magnetic_field_one_form"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("lorentz_factor"), py::arg("shift"), py::arg("lapse"));
  m.def("comoving_magnetic_field_squared",
        static_cast<Scalar<DataType> (*)(const Scalar<DataType>&,
                                         const Scalar<DataType>&,
                                         const Scalar<DataType>&)>(
            &hydro::comoving_magnetic_field_squared<DataType>),
        py::arg("magnetic_field_squared"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("lorentz_factor"));
}

template <typename DataType>
void bind_lorentz_factor_impl(py::module& m) {
  m.def("lorentz_factor",
        static_cast<Scalar<DataType> (*)(const Scalar<DataType>&)>(
            &hydro::lorentz_factor<DataType>),
        py::arg("spatial_velocity_squared"));
}

template <typename DataType, size_t Dim, typename Frame>
void bind_lorentz_factor_impl(py::module& m) {
  m.def("lorentz_factor",
        static_cast<Scalar<DataType> (*)(const tnsr::I<DataType, Dim, Frame>&,
                                         const tnsr::i<DataType, Dim, Frame>&)>(
            &hydro::lorentz_factor<DataType, Dim, Frame>),
        py::arg("spatial_velocity"), py::arg("spatial_velocity_form"));
}

template <typename DataType, size_t Dim, typename Frame>
void bind_mass_flux_impl(py::module& m) {
  m.def("mass_flux",
        static_cast<tnsr::I<DataType, Dim, Frame> (*)(
            const Scalar<DataType>&, const tnsr::I<DataType, Dim, Frame>&,
            const Scalar<DataType>&, const Scalar<DataType>&,
            const tnsr::I<DataType, Dim, Frame>&, const Scalar<DataType>&)>(
            &hydro::mass_flux<DataType, Dim, Frame>),
        py::arg("rest_mass_density"), py::arg("spatial_velocity"),
        py::arg("lorentz_factor"), py::arg("lapse"), py::arg("shift"),
        py::arg("sqrt_det_spatial_metric"));
}

template <typename DataType, size_t Dim>
void bind_mass_weighted_impl(py::module& m) {
  m.def(
      "u_lower_t",
      [](const Scalar<DataType>& lorentz_factor,
         const tnsr::I<DataType, Dim>& spatial_velocity,
         const tnsr::ii<DataType, Dim>& spatial_metric,
         const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim>& shift) {
        return hydro::u_lower_t(lorentz_factor, spatial_velocity,
                                spatial_metric, lapse, shift);
      },
      py::arg("lorentz_factor"), py::arg("spatial_velocity"),
      py::arg("spatial_metric"), py::arg("lapse"), py::arg("shift"));
  m.def(
      "tilde_d_unbound_ut_criterion",
      [](const Scalar<DataType>& tilde_d,
         const Scalar<DataType>& lorentz_factor,
         const tnsr::I<DataType, Dim>& spatial_velocity,
         const tnsr::ii<DataType, Dim>& spatial_metric,
         const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim>& shift) {
        return hydro::tilde_d_unbound_ut_criterion(
            tilde_d, lorentz_factor, spatial_velocity, spatial_metric, lapse,
            shift);
      },
      py::arg("tilde_d"), py::arg("lorentz_factor"),
      py::arg("spatial_velocity"), py::arg("spatial_metric"), py::arg("lapse"),
      py::arg("shift"));

  if constexpr (Dim == 1) {
    m.def(
        "mass_weighted_internal_energy",
        [](const Scalar<DataType>& tilde_d,
           const Scalar<DataType>& specific_internal_energy) {
          return hydro::mass_weighted_internal_energy(tilde_d,
                                                      specific_internal_energy);
        },
        py::arg("tilde_d"), py::arg("specific_internal_energy"));
    m.def(
        "mass_weighted_kinetic_energy",
        [](const Scalar<DataType>& tilde_d,
           const Scalar<DataType>& lorentz_factor) {
          return hydro::mass_weighted_kinetic_energy(tilde_d, lorentz_factor);
        },
        py::arg("tilde_d"), py::arg("lorentz_factor"));
  }
}

template <typename DataType, size_t ThermodynamicDim>
void bind_sound_speed_impl(py::module& m) {
  m.def("sound_speed_squared",
        static_cast<Scalar<DataType> (*)(
            const Scalar<DataType>&, const Scalar<DataType>&,
            const Scalar<DataType>&,
            const EquationsOfState::EquationOfState<true, ThermodynamicDim>&)>(
            &hydro::sound_speed_squared<DataType, ThermodynamicDim>),
        py::arg("rest_mass_density"), py::arg("specific_internal_energy"),
        py::arg("specific_enthalpy"), py::arg("equation_of_state"));
}

template <typename DataType>
void bind_specific_enthalpy_impl(py::module& m) {
  m.def("relativistic_specific_enthalpy",
        static_cast<Scalar<DataType> (*)(const Scalar<DataType>&,
                                         const Scalar<DataType>&,
                                         const Scalar<DataType>&)>(
            &hydro::relativistic_specific_enthalpy<DataType>),
        py::arg("rest_mass_density"), py::arg("specific_internal_energy"),
        py::arg("pressure"));
}

template <typename DataType>
void bind_stress_energy_impl(py::module& m) {
  m.def("energy_density", &hydro::energy_density<DataType>, py::arg("result"),
        py::arg("rest_mass_density"), py::arg("specific_enthalpy"),
        py::arg("pressure"), py::arg("lorentz_factor"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("comoving_magnetic_field_squared"));
  m.def("momentum_density", &hydro::momentum_density<DataType>,
        py::arg("result"), py::arg("rest_mass_density"),
        py::arg("specific_enthalpy"), py::arg("spatial_velocity"),
        py::arg("lorentz_factor"), py::arg("magnetic_field"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("comoving_magnetic_field_squared"));
  m.def("stress_trace", &hydro::stress_trace<DataType>, py::arg("result"),
        py::arg("rest_mass_density"), py::arg("specific_enthalpy"),
        py::arg("pressure"), py::arg("spatial_velocity_squared"),
        py::arg("lorentz_factor"),
        py::arg("magnetic_field_dot_spatial_velocity"),
        py::arg("comoving_magnetic_field_squared"));
}

template <typename DataType, size_t Dim>
void bind_impl(py::module& m) {
  bind_mass_flux_impl<DataType, Dim, Frame::Grid>(m);
  bind_mass_flux_impl<DataType, Dim, Frame::Inertial>(m);
  bind_lorentz_factor_impl<DataType, Dim, Frame::Inertial>(m);
  if constexpr (std::is_same_v<DataType, DataVector>) {
    bind_mass_weighted_impl<DataType, Dim>(m);
  }
}

template <typename DataType>
void bind_impl(py::module& m) {
  bind_impl<DataType, 1>(m);
  bind_impl<DataType, 2>(m);
  bind_impl<DataType, 3>(m);

  bind_comoving_magnetic_field_impl<DataType>(m);
  bind_lorentz_factor_impl<DataType>(m);
  // Here the dims are the thermodynamic dims, not spatial dim.
  bind_sound_speed_impl<DataType, 1>(m);
  bind_sound_speed_impl<DataType, 2>(m);

  bind_specific_enthalpy_impl<DataType>(m);
  bind_stress_energy_impl<DataType>(m);
}

}  // namespace

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.Spectral");
  bind_impl<DataVector>(m);
  m.def("lower_spatial_four_velocity",
        &hydro::Tags::LowerSpatialFourVelocityCompute::function,
        py::arg("result"), py::arg("spatial_velocity"),
        py::arg("spatial_metric"), py::arg("lorentz_factor"));
}

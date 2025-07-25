// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/Python/SwshCoefficients.hpp"

#include <pybind11/pybind11.h>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"

namespace py = pybind11;

namespace py_bindings {
namespace {
void bind_goldberg_to_nodal_impl(pybind11::module& m) {  // NOLINT
  m.def(
      "goldberg_to_nodal",
      [](DataVector goldberg_modes_interleaved, size_t l_max, int Spin) {
        // This function takes a real interleaved DataVector object reads from a
        // np.array() in python and internally constructs the
        // SpinWeighted<ComplexModalVector, Spin> object needed in
        // goldberg_to_nodal()
        if (Spin == -3) {
          SpinWeighted<ComplexModalVector, -3> goldberg_modes{
              goldberg_modes_interleaved.size() / 2};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            goldberg_modes.data()[i] =
                std::complex<double>(goldberg_modes_interleaved[2 * i],
                                     goldberg_modes_interleaved[(2 * i) + 1]);
          }
          auto nodal_values =
              Spectral::Swsh::goldberg_to_nodal(goldberg_modes, l_max);
          // Here we convert the SpinWeighted<ComplexDataVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * nodal_values.size()};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            result[2 * i] = nodal_values.data()[i].real();
            result[(2 * i) + 1] = nodal_values.data()[i].imag();
          }
          return result;
        } else if (Spin == -2) {
          SpinWeighted<ComplexModalVector, -2> goldberg_modes{
              goldberg_modes_interleaved.size() / 2};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            goldberg_modes.data()[i] =
                std::complex<double>(goldberg_modes_interleaved[2 * i],
                                     goldberg_modes_interleaved[(2 * i) + 1]);
          }
          auto nodal_values =
              Spectral::Swsh::goldberg_to_nodal(goldberg_modes, l_max);
          // Here we convert the SpinWeighted<ComplexDataVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * nodal_values.size()};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            result[2 * i] = nodal_values.data()[i].real();
            result[(2 * i) + 1] = nodal_values.data()[i].imag();
          }
          return result;
        } else if (Spin == -1) {
          SpinWeighted<ComplexModalVector, -1> goldberg_modes{
              goldberg_modes_interleaved.size() / 2};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            goldberg_modes.data()[i] =
                std::complex<double>(goldberg_modes_interleaved[2 * i],
                                     goldberg_modes_interleaved[(2 * i) + 1]);
          }
          auto nodal_values =
              Spectral::Swsh::goldberg_to_nodal(goldberg_modes, l_max);
          // Here we convert the SpinWeighted<ComplexDataVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * nodal_values.size()};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            result[2 * i] = nodal_values.data()[i].real();
            result[(2 * i) + 1] = nodal_values.data()[i].imag();
          }
          return result;
        } else if (Spin == 0) {
          SpinWeighted<ComplexModalVector, 0> goldberg_modes{
              goldberg_modes_interleaved.size() / 2};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            goldberg_modes.data()[i] =
                std::complex<double>(goldberg_modes_interleaved[2 * i],
                                     goldberg_modes_interleaved[(2 * i) + 1]);
          }
          auto nodal_values =
              Spectral::Swsh::goldberg_to_nodal(goldberg_modes, l_max);
          // Here we convert the SpinWeighted<ComplexDataVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * nodal_values.size()};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            result[2 * i] = nodal_values.data()[i].real();
            result[(2 * i) + 1] = nodal_values.data()[i].imag();
          }
          return result;
        } else if (Spin == 1) {
          SpinWeighted<ComplexModalVector, 1> goldberg_modes{
              goldberg_modes_interleaved.size() / 2};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            goldberg_modes.data()[i] =
                std::complex<double>(goldberg_modes_interleaved[2 * i],
                                     goldberg_modes_interleaved[(2 * i) + 1]);
          }
          auto nodal_values =
              Spectral::Swsh::goldberg_to_nodal(goldberg_modes, l_max);
          // Here we convert the SpinWeighted<ComplexDataVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * nodal_values.size()};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            result[2 * i] = nodal_values.data()[i].real();
            result[(2 * i) + 1] = nodal_values.data()[i].imag();
          }
          return result;
        } else if (Spin == 2) {
          SpinWeighted<ComplexModalVector, 2> goldberg_modes{
              goldberg_modes_interleaved.size() / 2};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            goldberg_modes.data()[i] =
                std::complex<double>(goldberg_modes_interleaved[2 * i],
                                     goldberg_modes_interleaved[(2 * i) + 1]);
          }
          auto nodal_values =
              Spectral::Swsh::goldberg_to_nodal(goldberg_modes, l_max);

          // Here we convert the SpinWeighted<ComplexDataVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * nodal_values.size()};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            result[2 * i] = nodal_values.data()[i].real();
            result[(2 * i) + 1] = nodal_values.data()[i].imag();
          }
          return result;
        } else {
          throw std::logic_error("Spin > 2 not implemented");
        }
      },
      py::arg("goldberg_modes_interleaved"), py::arg("l_max"), py::arg("Spin"));
}

void bind_nodal_to_goldberg_impl(pybind11::module& m) {  // NOLINT
  m.def(
      "nodal_to_goldberg",
      [](DataVector nodal_values_interleaved, size_t l_max, int Spin) {
        // This function takes a real interleaved DataVector object reads from a
        // np.array() in python and internally constructs the
        // SpinWeighted<ComplexDataVector, Spin> object needed in
        // nodal_to_goldberg()
        if (Spin == -3) {
          SpinWeighted<ComplexDataVector, -3> nodal_values{
              nodal_values_interleaved.size() / 2};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            nodal_values.data()[i] =
                std::complex<double>(nodal_values_interleaved[2 * i],
                                     nodal_values_interleaved[(2 * i) + 1]);
          }
          auto goldberg_modes =
              Spectral::Swsh::nodal_to_goldberg(nodal_values, l_max);
          // Here we convert the SpinWeighted<ComplexModalVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * goldberg_modes.size()};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            result[2 * i] = goldberg_modes.data()[i].real();
            result[(2 * i) + 1] = goldberg_modes.data()[i].imag();
          }
          return result;
        } else if (Spin == -2) {
          SpinWeighted<ComplexDataVector, -2> nodal_values{
              nodal_values_interleaved.size() / 2};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            nodal_values.data()[i] =
                std::complex<double>(nodal_values_interleaved[2 * i],
                                     nodal_values_interleaved[(2 * i) + 1]);
          }
          auto goldberg_modes =
              Spectral::Swsh::nodal_to_goldberg(nodal_values, l_max);
          // Here we convert the SpinWeighted<ComplexModalVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * goldberg_modes.size()};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            result[2 * i] = goldberg_modes.data()[i].real();
            result[(2 * i) + 1] = goldberg_modes.data()[i].imag();
          }
          return result;
        } else if (Spin == -1) {
          SpinWeighted<ComplexDataVector, -1> nodal_values{
              nodal_values_interleaved.size() / 2};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            nodal_values.data()[i] =
                std::complex<double>(nodal_values_interleaved[2 * i],
                                     nodal_values_interleaved[(2 * i) + 1]);
          }
          auto goldberg_modes =
              Spectral::Swsh::nodal_to_goldberg(nodal_values, l_max);
          // Here we convert the SpinWeighted<ComplexModalVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * goldberg_modes.size()};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            result[2 * i] = goldberg_modes.data()[i].real();
            result[(2 * i) + 1] = goldberg_modes.data()[i].imag();
          }
          return result;
        } else if (Spin == 0) {
          SpinWeighted<ComplexDataVector, 0> nodal_values{
              nodal_values_interleaved.size() / 2};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            nodal_values.data()[i] =
                std::complex<double>(nodal_values_interleaved[2 * i],
                                     nodal_values_interleaved[(2 * i) + 1]);
          }
          auto goldberg_modes =
              Spectral::Swsh::nodal_to_goldberg(nodal_values, l_max);
          // Here we convert the SpinWeighted<ComplexModalVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * goldberg_modes.size()};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            result[2 * i] = goldberg_modes.data()[i].real();
            result[(2 * i) + 1] = goldberg_modes.data()[i].imag();
          }
          return result;
        } else if (Spin == 1) {
          SpinWeighted<ComplexDataVector, 1> nodal_values{
              nodal_values_interleaved.size() / 2};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            nodal_values.data()[i] =
                std::complex<double>(nodal_values_interleaved[2 * i],
                                     nodal_values_interleaved[(2 * i) + 1]);
          }
          auto goldberg_modes =
              Spectral::Swsh::nodal_to_goldberg(nodal_values, l_max);
          // Here we convert the SpinWeighted<ComplexModalVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * goldberg_modes.size()};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            result[2 * i] = goldberg_modes.data()[i].real();
            result[(2 * i) + 1] = goldberg_modes.data()[i].imag();
          }
          return result;
        } else if (Spin == 2) {
          SpinWeighted<ComplexDataVector, 2> nodal_values{
              nodal_values_interleaved.size() / 2};
          for (size_t i = 0; i < nodal_values.size(); i++) {
            nodal_values.data()[i] =
                std::complex<double>(nodal_values_interleaved[2 * i],
                                     nodal_values_interleaved[(2 * i) + 1]);
          }
          auto goldberg_modes =
              Spectral::Swsh::nodal_to_goldberg(nodal_values, l_max);
          // Here we convert the SpinWeighted<ComplexModalVector, Spin> object
          // back to a real interleaved DataVector and that is the returning
          // value
          DataVector result{2 * goldberg_modes.size()};
          for (size_t i = 0; i < goldberg_modes.size(); i++) {
            result[2 * i] = goldberg_modes.data()[i].real();
            result[(2 * i) + 1] = goldberg_modes.data()[i].imag();
          }
          return result;
        } else {
          throw std::logic_error("Spin > 2 not implemented");
        }
      },
      py::arg("nodal_values_interleaved"), py::arg("l_max"), py::arg("Spin"));
}
}  // namespace

void bind_goldberg_to_nodal(pybind11::module& m) {  // NOLINT
  bind_goldberg_to_nodal_impl(m);
}
void bind_nodal_to_goldberg(pybind11::module& m) {  // NOLINT
  bind_nodal_to_goldberg_impl(m);
}
}  // namespace py_bindings

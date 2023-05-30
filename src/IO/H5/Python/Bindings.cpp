// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "IO/H5/Python/Dat.hpp"
#include "IO/H5/Python/File.hpp"
#include "IO/H5/Python/TensorData.hpp"
#include "IO/H5/Python/VolumeData.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.Spectral");
  py_bindings::bind_h5file(m);
  py_bindings::bind_h5dat(m);
  py_bindings::bind_h5vol(m);
  py_bindings::bind_tensordata(m);
}

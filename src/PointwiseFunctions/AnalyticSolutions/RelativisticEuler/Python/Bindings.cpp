// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/Python/Tov.hpp"

PYBIND11_MODULE(_PyRelativisticEulerSolutions, m) {  // NOLINT
  RelativisticEuler::Solutions::py_bindings::bind_tov(m);
}

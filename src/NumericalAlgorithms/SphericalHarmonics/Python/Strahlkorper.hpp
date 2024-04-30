// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace ylm::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_strahlkorper(pybind11::module& m);
}  // namespace ylm::py_bindings

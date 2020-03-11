// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/Matrix.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Helpers/Elliptic/Systems/Poisson/DgSchemes.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace TestHelpers {
namespace Poisson {
namespace py_bindings {

namespace detail {
template <size_t Dim>
void bind_dg_schemes(py::module& m) {  // NOLINT
  m.def(("strong_first_order_dg_operator_matrix_" + get_output(Dim) + "d")
            .c_str(),
        &strong_first_order_dg_operator_matrix<Dim>);
}
}  // namespace detail

void bind_dg_schemes(py::module& m) {  // NOLINT
  detail::bind_dg_schemes<1>(m);
  detail::bind_dg_schemes<2>(m);
  detail::bind_dg_schemes<3>(m);
}

}  // namespace py_bindings
}  // namespace Poisson
}  // namespace TestHelpers

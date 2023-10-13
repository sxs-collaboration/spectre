// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Amr/Flag.hpp"

#include <array>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>

#include "Domain/Amr/Info.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"

namespace amr {
template <size_t VolumeDim>
void Info<VolumeDim>::pup(PUP::er& p) {
  p | flags;
  p | new_mesh;
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const amr::Info<VolumeDim>& info) {
  using ::operator<<;
  os << "Flags: " << info.flags << " New mesh: " << info.new_mesh;
  return os;
}

template <size_t VolumeDim>
bool operator==(const Info<VolumeDim>& lhs, const Info<VolumeDim>& rhs) {
  return lhs.flags == rhs.flags and lhs.new_mesh == rhs.new_mesh;
}

template <size_t VolumeDim>
bool operator!=(const amr::Info<VolumeDim>& lhs,
                const amr::Info<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEN_OP(op, dim) \
  template bool operator op(const Info<dim>& lhs, const Info<dim>& rhs);
#define INSTANTIATE(_, data)                          \
  template struct Info<DIM(data)>;                    \
  GEN_OP(==, DIM(data))                               \
  GEN_OP(!=, DIM(data))                               \
  template std::ostream& operator<<(std::ostream& os, \
                                    const Info<DIM(data)>& info);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))
#undef INSTANTIATE
#undef GEN_OP
#undef DIM
}  // namespace amr

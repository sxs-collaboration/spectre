// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Initialization/DgDomain.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Utilities/GenerateInstantiations.hpp"

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define SFRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(r, data)                                               \
  template std::unique_ptr<                                                \
      domain::CoordinateMapBase<SFRAME(data), Frame::Inertial, DIM(data)>> \
      domain::make_coordinate_map_base<                                    \
          SFRAME(data), Frame::Inertial,                                   \
          domain::CoordinateMaps::Identity<DIM(data)>>(                    \
          domain::CoordinateMaps::Identity<DIM(data)> &&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Logical))

#undef INSTANTIATE
#undef DIM

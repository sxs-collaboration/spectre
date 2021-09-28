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

#define INSTANTIATE(r, data)                                              \
  template std::unique_ptr<                                               \
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, DIM(data)>> \
      domain::make_coordinate_map_base<                                   \
          Frame::Grid, Frame::Inertial,                                   \
          domain::CoordinateMaps::Identity<DIM(data)>>(                   \
          domain::CoordinateMaps::Identity<DIM(data)> &&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

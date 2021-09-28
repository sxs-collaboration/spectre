// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/enum.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/GenerateInstantiations.hpp"

#define INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS(data) \
  BOOST_PP_TUPLE_ENUM(BOOST_PP_TUPLE_ELEM(0, data))
#define INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data)                     \
  BOOST_PP_TUPLE_ELEM(BOOST_PP_TUPLE_SIZE(BOOST_PP_TUPLE_ELEM(0, data)), 0, \
                      BOOST_PP_TUPLE_ELEM(0, data))::dim
#define INSTANTIATE_COORD_MAP_DETAILINSERT_SIZE(z, n, _) BOOST_PP_COMMA_IF(n) n

#define INSTANTIATE_COORD_MAP_DETAIL_GET_FILLED_INDEX_SEQUENCE(data) \
  BOOST_PP_REPEAT(BOOST_PP_TUPLE_SIZE(BOOST_PP_TUPLE_ELEM(0, data)), \
                  INSTANTIATE_COORD_MAP_DETAILINSERT_SIZE, _)

#define INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data) \
  BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data) \
  BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Generate instantiations of member functions of the `CoordinateMap`
 * class template.
 *
 * Called as follows:
 *
 * \code
 *  using Affine2d =
 *      domain::CoordinateMaps::ProductOf2Maps<domain::CoordinateMaps::Affine,
 *                                             domain::CoordinateMaps::Affine>;
 *  using Affine3d =
 *      domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
 *                                             domain::CoordinateMaps::Affine,
 *                                             domain::CoordinateMaps::Affine>;
 *
 *  GENERATE_INSTANTIATIONS(INSTANTIATE_MAPS_SIMPLE_FUNCTIONS,
 *                          ((Affine2d), (Affine3d)), (Frame::BlockLogical),
 *                          (Frame::Grid, Frame::Inertial))
 * \endcode
 *
 * The first tuple passed to `GENERATE_INSTANTIATIONS` has a bunch of tuples in
 * it that is the list of maps being composed. The reason for defining the type
 * aliases `Affine2d` and `Affine3d` is that otherwise the number of maps being
 * composed is calculated incorrectly. The second tuple contains the source
 * frames for the map. The third tuple passed to `GENERATE_INSTANTIATIONS`
 * contains the target frames to instantiate for, typically `Frame::Grid` and
 * `Frame::Inertial`.
 *
 * Instantiates:
 * - `get_to_grid_frame_impl`
 * - `inverse_impl`
 * - `class CoordinateMap`
 */
#define INSTANTIATE_MAPS_SIMPLE_FUNCTIONS(_, data)                           \
  template std::unique_ptr<domain::CoordinateMapBase<                        \
      INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data), Frame::Grid,      \
      INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data)>>                      \
      domain::CoordinateMap<                                                 \
          INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data),               \
          INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data),               \
          INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS(data)>::                     \
          get_to_grid_frame_impl(                                            \
              std::integer_sequence<                                         \
                  unsigned long,                                             \
                  INSTANTIATE_COORD_MAP_DETAIL_GET_FILLED_INDEX_SEQUENCE(    \
                      data)>) const;                                         \
  template std::optional<                                                    \
      tnsr::I<double, INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),       \
              INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data)>>          \
  domain::CoordinateMap<INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data), \
                        INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data), \
                        INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS(data)>::       \
      inverse_impl(                                                          \
          tnsr::I<double, INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),   \
                  INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data)>&&,    \
          double,                                                            \
          const std::unordered_map<                                          \
              std::string,                                                   \
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&,    \
          std::index_sequence<                                               \
              INSTANTIATE_COORD_MAP_DETAIL_GET_FILLED_INDEX_SEQUENCE(        \
                  data)> /*meta*/) const;                                    \
  template class domain::CoordinateMap<                                      \
      INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data),                   \
      INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data),                   \
      INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS(data)>;

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Generate instantiations of member functions of the `CoordinateMap`
 * class template.
 *
 * Called as follows:
 *
 * \code
 *  using Affine2d =
 *      domain::CoordinateMaps::ProductOf2Maps<domain::CoordinateMaps::Affine,
 *                                             domain::CoordinateMaps::Affine>;
 *  using Affine3d =
 *      domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
 *                                             domain::CoordinateMaps::Affine,
 *                                             domain::CoordinateMaps::Affine>;
 *
 *  GENERATE_INSTANTIATIONS(INSTANTIATE_MAPS_DATA_TYPE_FUNCTIONS,
 *                          ((Affine2d), (Affine3d)), (Frame::BlockLogical),
 *                          (Frame::Grid, Frame::Inertial),
 *                          (double, DataVector))
 * \endcode
 *
 * The first tuple passed to `GENERATE_INSTANTIATIONS` has a bunch of tuples in
 * it that is the list of maps being composed. The reason for defining the type
 * aliases `Affine2d` and `Affine3d` is that otherwise the number of maps being
 * composed is calculated incorrectly. The second tuple contains the source
 * frames for the map. The third tuple passed to  `GENERATE_INSTANTIATIONS`
 * contains the target frames to instantiate for, typically `Frame::Grid` and
 * `Frame::Inertial`. The last tuple is the data types for which to instantiate
 * the functions, usually `double` and `DataVector`.
 *
 * Instantiates:
 * - `call_impl`
 * - `inv_jacobian_impl`
 * - `jacobian_impl`
 * - `coords_frame_velocity_jacobians_impl`
 */
#define INSTANTIATE_MAPS_DATA_TYPE_FUNCTIONS(_, data)                        \
  template tnsr::I<INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),                 \
                   INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),          \
                   INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data)>      \
  domain::CoordinateMap<INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data), \
                        INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data), \
                        INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS(data)>::       \
      call_impl(                                                             \
          tnsr::I<INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),                  \
                  INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),           \
                  INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data)>&&,    \
          double,                                                            \
          const std::unordered_map<                                          \
              std::string,                                                   \
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&,    \
          std::integer_sequence<                                             \
              unsigned long,                                                 \
              INSTANTIATE_COORD_MAP_DETAIL_GET_FILLED_INDEX_SEQUENCE(data)>) \
          const;                                                             \
  template InverseJacobian<                                                  \
      INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),                              \
      INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),                       \
      INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data),                   \
      INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data)>                   \
  domain::CoordinateMap<INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data), \
                        INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data), \
                        INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS(data)>::       \
      inv_jacobian_impl(                                                     \
          tnsr::I<INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),                  \
                  INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),           \
                  INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data)>&&,    \
          double,                                                            \
          const std::unordered_map<                                          \
              std::string,                                                   \
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&)    \
          const;                                                             \
  template Jacobian<INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),                \
                    INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),         \
                    INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data),     \
                    INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data)>     \
  domain::CoordinateMap<INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data), \
                        INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data), \
                        INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS(data)>::       \
      jacobian_impl(                                                         \
          tnsr::I<INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),                  \
                  INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),           \
                  INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data)>&&,    \
          double,                                                            \
          const std::unordered_map<                                          \
              std::string,                                                   \
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&)    \
          const;                                                             \
  template std::tuple<                                                       \
      tnsr::I<INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),                      \
              INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),               \
              INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data)>,          \
      InverseJacobian<INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),              \
                      INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),       \
                      INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data),   \
                      INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data)>,  \
      Jacobian<INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),                     \
               INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),              \
               INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data),          \
               INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data)>,         \
      tnsr::I<INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),                      \
              INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),               \
              INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data)>>          \
  domain::CoordinateMap<INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data), \
                        INSTANTIATE_COORD_MAP_DETAIL_GET_TARGET_FRAME(data), \
                        INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS(data)>::       \
      coords_frame_velocity_jacobians_impl(                                  \
          tnsr::I<INSTANTIATE_COORD_MAP_DETAIL_DTYPE(data),                  \
                  INSTANTIATE_COORD_MAP_DETAIL_GET_MAPS_DIM(data),           \
                  INSTANTIATE_COORD_MAP_DETAIL_GET_SOURCE_FRAME(data)>,      \
          double,                                                            \
          const std::unordered_map<                                          \
              std::string,                                                   \
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&)    \
          const;

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Generate instantiations of member functions of the `CoordinateMap`
 * class template.
 *
 * Called as follows:
 *
 * \code
 *  using Affine2d =
 *      domain::CoordinateMaps::ProductOf2Maps<domain::CoordinateMaps::Affine,
 *                                             domain::CoordinateMaps::Affine>;
 *  using Affine3d =
 *      domain::CoordinateMaps::ProductOf3Maps<domain::CoordinateMaps::Affine,
 *                                             domain::CoordinateMaps::Affine,
 *                                             domain::CoordinateMaps::Affine>;
 *
 *  INSTANTIATE_MAPS_FUNCTIONS(((Affine2d), (Affine3d)), (Frame::BlockLogical),
 *                             (Frame::Grid, Frame::Inertial),
 *                             (double, DataVector))
 * \endcode
 *
 * The first tuple passed to `GENERATE_INSTANTIATIONS` has a bunch of tuples in
 * it that is the list of maps being composed. The reason for defining the type
 * aliases `Affine2d` and `Affine3d` is that otherwise the number of maps being
 * composed is calculated incorrectly. The second tuple contains the source
 * frames for the map. The third tuple passed to `GENERATE_INSTANTIATIONS`
 * contains the frames to instantiate for, typically `Frame::Grid` and
 * `Frame::Inertial`.
 *
 * Instantiates:
 * - `get_to_grid_frame_impl`
 * - `inverse_impl`
 * - `class CoordinateMap`
 * - `call_impl`
 * - `inv_jacobian_impl`
 * - `jacobian_impl`
 * - `coords_frame_velocity_jacobians_impl`
 */
#define INSTANTIATE_MAPS_FUNCTIONS(MAPS_TUPLE, SOURCE_FRAME,                \
                                   TARGET_FRAMES_TUPLE, TYPES_TUPLE)        \
  GENERATE_INSTANTIATIONS(INSTANTIATE_MAPS_SIMPLE_FUNCTIONS, MAPS_TUPLE,    \
                          SOURCE_FRAME, TARGET_FRAMES_TUPLE)                \
  GENERATE_INSTANTIATIONS(INSTANTIATE_MAPS_DATA_TYPE_FUNCTIONS, MAPS_TUPLE, \
                          SOURCE_FRAME, TARGET_FRAMES_TUPLE, TYPES_TUPLE)

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <random>

#include "Helpers/DataStructures/MakeWithRandomValues.hpp"

/*!
 * \brief Generates random coordinates of double and DataVector types.
 *
 * Creates:
 * - `std::uniform_real_distribution<double>` named `dist` on interval
 *   `LOWER_BOUND` and `UPPER_BOUND`
 * - `grid_coords_dv` of type `tnsr::I<DataVector, DIM, Frame::Grid>`
 * - `grid_coords_double` of type `tnsr::I<double, DIMN, Frame::Grid>`
 * - `inertial_coords_dv` of type `tnsr::I<DataVector, DIM, Frame::Inertial>`
 * - `inertial_coords_double` of type `tnsr::I<double, DIMN, Frame::Inertial>`
 *
 * The argument `GENERATOR` must be a `gsl::not_null` to a random number
 * generator. Typically the generator would be created using the
 * `MAKE_GENERATOR` macro.
 */
#define TIME_DEPENDENCE_GENERATE_COORDS(GENERATOR, DIM, LOWER_BOUND,      \
                                        UPPER_BOUND)                      \
  std::uniform_real_distribution<double> dist{LOWER_BOUND, UPPER_BOUND};  \
  const auto grid_coords_dv =                                             \
      make_with_random_values<tnsr::I<DataVector, DIM, Frame::Grid>>(     \
          GENERATOR, make_not_null(&dist), DataVector{5});                \
  const auto grid_coords_double =                                         \
      make_with_random_values<tnsr::I<double, DIM, Frame::Grid>>(         \
          GENERATOR, make_not_null(&dist));                               \
  const auto inertial_coords_dv =                                         \
      make_with_random_values<tnsr::I<DataVector, DIM, Frame::Inertial>>( \
          GENERATOR, make_not_null(&dist), DataVector{5});                \
  const auto inertial_coords_double =                                     \
      make_with_random_values<tnsr::I<double, DIM, Frame::Inertial>>(     \
          GENERATOR, make_not_null(&dist))

/*!
 * \brief Generates random coordinates of double and DataVector types.
 *
 * Same as TIME_DEPENDENCE_GENERATE_COORDS except:
 * - It creates only distorted-frame coords.
 * - It takes a DIST as an argument, since in typical usage one would
 *   call TIME_DEPENDENCE_GENERATE_COORDS and then call
 *   TIME_DEPENDENCE_GENERATE_DISTORTED_COORDS.
 *
 */
#define TIME_DEPENDENCE_GENERATE_DISTORTED_COORDS(GENERATOR, DIST, DIM)    \
  const auto distorted_coords_dv =                                         \
      make_with_random_values<tnsr::I<DataVector, DIM, Frame::Distorted>>( \
          GENERATOR, make_not_null(&dist), DataVector{5});                 \
  const auto distorted_coords_double =                                     \
      make_with_random_values<tnsr::I<double, DIM, Frame::Distorted>>(     \
          GENERATOR, make_not_null(&dist));

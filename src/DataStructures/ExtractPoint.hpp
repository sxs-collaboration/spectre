// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup DataStructuresGroup
/// Copy a given index of each component of a `Tensor<DataVector>` or
/// `Variables<DataVector>` into a `Tensor<double>`, single point
/// `Tensor<DataVector>`, or single-point `Variables<DataVector>`.
///
/// \note There is no by-value overload extracting to a
/// `Tensor<DataVector>`.  This is both for the practical reason that
/// it would be ambiguous with the `Tensor<double>` overload and
/// because allocating multiple `DataVector`s for the return type
/// would usually be very inefficient.
///
/// \see overwrite_point
/// @{
template <typename... Structure>
void extract_point(
    const gsl::not_null<Tensor<double, Structure...>*> destination,
    const Tensor<DataVector, Structure...>& source, const size_t index) {
  for (size_t i = 0; i < destination->size(); ++i) {
    (*destination)[i] = source[i][index];
  }
}

template <typename... Structure>
Tensor<double, Structure...> extract_point(
    const Tensor<DataVector, Structure...>& tensor, const size_t index) {
  Tensor<double, Structure...> result;
  extract_point(make_not_null(&result), tensor, index);
  return result;
}

template <typename... Structure>
void extract_point(
    const gsl::not_null<Tensor<DataVector, Structure...>*> destination,
    const Tensor<DataVector, Structure...>& source, const size_t index) {
  ASSERT(destination->begin()->size() == 1,
         "Output tensor components have wrong size: "
         << destination->begin()->size());
  for (size_t i = 0; i < destination->size(); ++i) {
    (*destination)[i][0] = source[i][index];
  }
}

template <typename... Tags>
void extract_point(
    const gsl::not_null<Variables<tmpl::list<Tags...>>*> result,
    const Variables<tmpl::list<Tags...>>& variables, const size_t index) {
  result->initialize(1);
  expand_pack((extract_point(
      make_not_null(&get<Tags>(*result)), get<Tags>(variables), index), 0)...);
}

template <typename... Tags>
Variables<tmpl::list<Tags...>> extract_point(
    const Variables<tmpl::list<Tags...>>& variables, const size_t index) {
  Variables<tmpl::list<Tags...>> result(1);
  extract_point(make_not_null(&result), variables, index);
  return result;
}
/// @}

/// \ingroup DataStructuresGroup
/// Copy a `Tensor<double>`, single point `Tensor<DataVector>`, or
/// single-point `Variables<DataVector>` into the given index of each
/// component of a `Tensor<DataVector>` or `Variables<DataVector>`.
///
/// \see extract_point
/// @{
template <typename... Structure>
void overwrite_point(
    const gsl::not_null<Tensor<DataVector, Structure...>*> destination,
    const Tensor<double, Structure...>& source, const size_t index) {
  for (size_t i = 0; i < destination->size(); ++i) {
    (*destination)[i][index] = source[i];
  }
}

template <typename... Structure>
void overwrite_point(
    const gsl::not_null<Tensor<DataVector, Structure...>*> destination,
    const Tensor<DataVector, Structure...>& source, const size_t index) {
  ASSERT(source.begin()->size() == 1,
         "Cannot overwrite with " << source.begin()->size() << " points.");
  for (size_t i = 0; i < destination->size(); ++i) {
    (*destination)[i][index] = source[i][0];
  }
}

template <typename... Tags>
void overwrite_point(
    const gsl::not_null<Variables<tmpl::list<Tags...>>*> destination,
    const Variables<tmpl::list<Tags...>>& source, const size_t index) {
  ASSERT(source.number_of_grid_points() == 1,
         "Must overwrite with a single point.");
  expand_pack((overwrite_point(make_not_null(&get<Tags>(*destination)),
                               extract_point(get<Tags>(source), 0), index),
               0)...);
}
/// @}

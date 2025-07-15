// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"

namespace ah::Storage {
/*!
 * \brief Holds the `ah::source_vars`, mesh, and other variables on the horizon
 * finder for a given element.
 */
template <typename Fr>
struct VolumeVariables {
  /*!
   * \brief The mesh that corresponds to the volume vars.
   */
  Mesh<3> mesh;

  /*!
   * \brief A `Variables` of the  `ah::source_vars` in the volume.
   */
  Variables<ah::source_vars<3>> source_vars;

  /*!
   * \brief A `Variables` of the tensors in the volume that we need to
   * interpolate onto the horizon.
   */
  Variables<ah::vars_to_interpolate_to_target<3, Fr>>
      vars_to_interpolate_to_target{};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <typename Fr>
bool operator==(const VolumeVariables<Fr>& lhs, const VolumeVariables<Fr>& rhs);
template <typename Fr>
bool operator!=(const VolumeVariables<Fr>& lhs, const VolumeVariables<Fr>& rhs);

/*!
 * \brief Holds the `ylm::Strahlkorper` and associated quantities for a single
 * `FastFlow` iteration.
 */
template <typename Fr>
struct Iteration {
  /*!
   * \brief The surface for this iteration
   */
  ylm::Strahlkorper<Fr> strahlkorper;
  /*!
   *  \brief Holds the list of all points (in block logical coordinates) that
   *  need to be interpolated onto.
   *
   * \details If an element of the vector is `std::nullopt`, then that point is
   * outside the domain.
   */
  std::optional<std::vector<BlockLogicalCoords<3>>> block_coord_holders;
  /*!
   * \brief Holds the interpolated `Variables` on the points in
   * `block_coord_holders`.
   *
   * \details The grid points inside are indexed according to
   * `block_coord_holders`.
   */
  Variables<ah::vars_to_interpolate_to_target<3, Fr>> interpolated_vars{};
  /*!
   * \brief Keeps track of the indices in `interpolated_vars` that have
   * already beed interpolated to.
   */
  std::set<size_t> indicies_interpolated_to_thus_far;
  /*!
   * \brief Holds the `ElementId`s of `Element`s for which interpolation has
   * already been done.
   */
  std::unordered_set<ElementId<3>> interpolation_is_done_for_these_elements;

  /*!
   * \brief How many times we've tried to compute the coordinates for this
   * iteration.
   */
  size_t compute_coords_retries = 0;

  void reset_for_next_iteration();

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <typename Fr>
bool operator==(const Iteration<Fr>& lhs, const Iteration<Fr>& rhs);
template <typename Fr>
bool operator!=(const Iteration<Fr>& lhs, const Iteration<Fr>& rhs);

/*!
 * \brief Holds all data necessary for a single horizon find.
 *
 * \details This includes volume variables which persist for the entire horizon
 * find, and also interpolated variables that are updated for each iteration.
 */
template <typename Fr>
struct SingleTimeStorage {
  /*!
   * \brief Map between `ElementId`s and the volume variables from that element.
   */
  std::unordered_map<ElementId<3>, VolumeVariables<Fr>> all_volume_variables;

  /*!
   * \brief The `Iteration` data for the current fast flow iteration.
   */
  Iteration<Fr> current_iteration;
  /*!
   * \brief The previous iteration surface, used if interpolation fails.
   */
  ylm::Strahlkorper<Fr> previous_iteration_surface;
  /*!
   * \brief The `ah::Destination` for this horizon find.
   */
  Destination destination{};
  /*!
   * \brief Whether we have checked if the functions of time are up to date for
   * this horizon find.
   */
  bool time_is_ready = false;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <typename Fr>
bool operator==(const SingleTimeStorage<Fr>& lhs,
                const SingleTimeStorage<Fr>& rhs);
template <typename Fr>
bool operator!=(const SingleTimeStorage<Fr>& lhs,
                const SingleTimeStorage<Fr>& rhs);

/*!
 * \brief The time and final surface for a previous horizon find.
 */
template <typename Fr>
struct PreviousSurface {
  PreviousSurface() = default;
  PreviousSurface(const LinkedMessageId<double>& time_in,
                  ylm::Strahlkorper<Fr> surface_in);

  LinkedMessageId<double> time;
  ylm::Strahlkorper<Fr> surface;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <typename Fr>
bool operator==(const PreviousSurface<Fr>& lhs, const PreviousSurface<Fr>& rhs);
template <typename Fr>
bool operator!=(const PreviousSurface<Fr>& lhs, const PreviousSurface<Fr>& rhs);
}  // namespace ah::Storage

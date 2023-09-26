// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <mutex>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace io {

/// Type of FUKA initial data
enum class FukaIdType { Bh, Bbh, Ns, Bns, Bhns };

namespace detail {
using fuka_gr_tags =
    tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
               gr::Tags::SpatialMetric<DataVector, 3>,
               gr::Tags::ExtrinsicCurvature<DataVector, 3>>;
using fuka_hydro_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::SpecificInternalEnergy<DataVector>,
               hydro::Tags::Pressure<DataVector>,
               hydro::Tags::SpatialVelocity<DataVector, 3>>;
template <FukaIdType IdType>
struct FukaTags {
  using type =
      tmpl::conditional_t<IdType <= FukaIdType::Bh or IdType == FukaIdType::Bbh,
                          fuka_gr_tags,
                          tmpl::append<fuka_gr_tags, fuka_hydro_tags>>;
};
}  // namespace detail

/// List of tags supplied by FUKA initial data
template <FukaIdType IdType>
using fuka_tags = typename detail::FukaTags<IdType>::type;

/*!
 * \brief Interpolate numerical FUKA initial data to arbitrary points
 *
 * \tparam IdType Type of FUKA initial data
 * \param fuka_lock Lock for accessing FUKA data. This is needed because
 * FUKA is not thread-safe. Pass in a lock that is shared with other
 * threads that are also calling this function.
 * \param info_filename Path to the FUKA info file to load
 * \param x Coordinates of points to interpolate to
 * \param interpolation_offset See FUKA documentation for export functions
 * \param interp_order See FUKA documentation for export functions
 * \param delta_r_rel See FUKA documentation for export functions
 * \return Data interpolated to the given points
 */
template <FukaIdType IdType>
tuples::tagged_tuple_from_typelist<fuka_tags<IdType>> interpolate_from_fuka(
    gsl::not_null<std::mutex*> fuka_lock, const std::string& info_filename,
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    double interpolation_offset = 0., int interp_order = 8,
    double delta_r_rel = 0.3);

}  // namespace io

// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/External/InterpolateFromFuka.hpp"

#include <array>
#include <cstddef>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

std::array<std::vector<double>, 16> KadathExportBH(
    int npoints, double const* xx, double const* yy, double const* zz,
    char const* fn, double interpolation_offset, int interp_order,
    double delta_r_rel);

std::array<std::vector<double>, 16> KadathExportBBH(
    int npoints, double const* xx, double const* yy, double const* zz,
    char const* fn, double interpolation_offset, int interp_order,
    double delta_r_rel);

std::array<std::vector<double>, 22> KadathExportNS(int npoints,
                                                   double const* xx,
                                                   double const* yy,
                                                   double const* zz,
                                                   char const* fn);

std::array<std::vector<double>, 22> KadathExportBNS(int npoints,
                                                    double const* xx,
                                                    double const* yy,
                                                    double const* zz,
                                                    char const* fn);

std::array<std::vector<double>, 22> KadathExportBHNS(
    int npoints, double const* xx, double const* yy, double const* zz,
    char const* fn, double interpolation_offset, int interp_order,
    double delta_r_rel);

namespace io {

namespace {
DataVector to_datavector(std::vector<double> vec) {
  DataVector result(vec.size());
  std::copy(vec.begin(), vec.end(), result.begin());
  return result;
}
}  // namespace

template <FukaIdType IdType>
tuples::tagged_tuple_from_typelist<fuka_tags<IdType>> interpolate_from_fuka(
    const gsl::not_null<std::mutex*> fuka_lock,
    const std::string& info_filename,
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    [[maybe_unused]] const double interpolation_offset,
    [[maybe_unused]] const int interp_order,
    [[maybe_unused]] const double delta_r_rel) {
  tuples::tagged_tuple_from_typelist<fuka_tags<IdType>> result{};
  // The FUKA functions are not thread-safe, so we need to lock here.
  fuka_lock->lock();
  const auto fuka_data = [&x, &info_filename, &interpolation_offset,
                          &interp_order, &delta_r_rel]() {
    // FUKA throws FPEs for some reason. Just disabling them seems to work, but
    // it is unclear what's causing this and if it can be a problem.
    const ScopedFpeState disable_fpes(false);
    if constexpr (IdType == FukaIdType::Bh) {
      return KadathExportBH(static_cast<int>(x.begin()->size()),
                            get<0>(x).data(), get<1>(x).data(),
                            get<2>(x).data(), info_filename.c_str(),
                            interpolation_offset, interp_order, delta_r_rel);
    } else if constexpr (IdType == FukaIdType::Bbh) {
      return KadathExportBBH(static_cast<int>(x.begin()->size()),
                             get<0>(x).data(), get<1>(x).data(),
                             get<2>(x).data(), info_filename.c_str(),
                             interpolation_offset, interp_order, delta_r_rel);
    } else if constexpr (IdType == FukaIdType::Ns) {
      (void)interpolation_offset;
      (void)interp_order;
      (void)delta_r_rel;
      return KadathExportNS(static_cast<int>(x.begin()->size()),
                            get<0>(x).data(), get<1>(x).data(),
                            get<2>(x).data(), info_filename.c_str());
    } else if constexpr (IdType == FukaIdType::Bns) {
      (void)interpolation_offset;
      (void)interp_order;
      (void)delta_r_rel;
      return KadathExportBNS(static_cast<int>(x.begin()->size()),
                             get<0>(x).data(), get<1>(x).data(),
                             get<2>(x).data(), info_filename.c_str());
    } else if constexpr (IdType == FukaIdType::Bhns) {
      return KadathExportBHNS(static_cast<int>(x.begin()->size()),
                              get<0>(x).data(), get<1>(x).data(),
                              get<2>(x).data(), info_filename.c_str(),
                              interpolation_offset, interp_order, delta_r_rel);
    } else {
      ERROR("Unrecognized enum value for 'FukaIdType'");
    }
  }();
  // Unlock before copying data to avoid holding the lock for too long.
  fuka_lock->unlock();
  // The FUKA functions return tensor components in this order.
  // See: https://bitbucket.org/fukaws/kadathimporter/src/master/src/importer.h
  auto& lapse = get<gr::Tags::Lapse<DataVector>>(result);
  get(lapse) = to_datavector(fuka_data[0]);
  auto& shift = get<gr::Tags::Shift<DataVector, 3>>(result);
  get<0>(shift) = to_datavector(fuka_data[1]);
  get<1>(shift) = to_datavector(fuka_data[2]);
  get<2>(shift) = to_datavector(fuka_data[3]);
  auto& spatial_metric = get<gr::Tags::SpatialMetric<DataVector, 3>>(result);
  get<0, 0>(spatial_metric) = to_datavector(fuka_data[4]);
  get<0, 1>(spatial_metric) = to_datavector(fuka_data[5]);
  get<0, 2>(spatial_metric) = to_datavector(fuka_data[6]);
  get<1, 1>(spatial_metric) = to_datavector(fuka_data[7]);
  get<1, 2>(spatial_metric) = to_datavector(fuka_data[8]);
  get<2, 2>(spatial_metric) = to_datavector(fuka_data[9]);
  auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(result);
  get<0, 0>(extrinsic_curvature) = to_datavector(fuka_data[10]);
  get<0, 1>(extrinsic_curvature) = to_datavector(fuka_data[11]);
  get<0, 2>(extrinsic_curvature) = to_datavector(fuka_data[12]);
  get<1, 1>(extrinsic_curvature) = to_datavector(fuka_data[13]);
  get<1, 2>(extrinsic_curvature) = to_datavector(fuka_data[14]);
  get<2, 2>(extrinsic_curvature) = to_datavector(fuka_data[15]);
  if constexpr (IdType == FukaIdType::Ns or IdType == FukaIdType::Bns or
                IdType == FukaIdType::Bhns) {
    auto& rest_mass_density =
        get<hydro::Tags::RestMassDensity<DataVector>>(result);
    get(rest_mass_density) = to_datavector(fuka_data[16]);
    auto& specific_internal_energy =
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(result);
    get(specific_internal_energy) = to_datavector(fuka_data[17]);
    auto& pressure = get<hydro::Tags::Pressure<DataVector>>(result);
    get(pressure) = to_datavector(fuka_data[18]);
    auto& spatial_velocity =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(result);
    get<0>(spatial_velocity) = to_datavector(fuka_data[19]);
    get<1>(spatial_velocity) = to_datavector(fuka_data[20]);
    get<2>(spatial_velocity) = to_datavector(fuka_data[21]);
  }
  return result;
}

#define ID_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                \
  template tuples::tagged_tuple_from_typelist<fuka_tags<ID_TYPE(data)>>       \
  interpolate_from_fuka<ID_TYPE(data)>(                                       \
      gsl::not_null<std::mutex*> fuka_lock, const std::string& info_filename, \
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,                       \
      double interpolation_offset, int interp_order, double delta_r_rel);

GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (FukaIdType::Bh, FukaIdType::Bbh, FukaIdType::Ns,
                         FukaIdType::Bns, FukaIdType::Bhns))

#undef INSTANTIATION
#undef ID_TYPE
}  // namespace io

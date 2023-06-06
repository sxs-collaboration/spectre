// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GeneralRelativity/SpecInitialData.hpp"

#include <Exporter.hpp>  // The SpEC Exporter
#include <memory>
#include <pup.h>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticData/GeneralRelativity/InterpolateFromSpec.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace gr::AnalyticData {

SpecInitialData::SpecInitialData(std::string data_directory)
    : data_directory_(std::move(data_directory)),
      spec_exporter_(std::make_unique<spec::Exporter>(
          sys::procs_on_node(sys::my_node()), data_directory_,
          vars_to_interpolate_)) {}

SpecInitialData::SpecInitialData(const SpecInitialData& rhs)
    : evolution::initial_data::InitialData(rhs) {
  *this = rhs;
}

SpecInitialData& SpecInitialData::operator=(const SpecInitialData& rhs) {
  data_directory_ = rhs.data_directory_;
  spec_exporter_ =
      std::make_unique<spec::Exporter>(sys::procs_on_node(sys::my_node()),
                                       data_directory_, vars_to_interpolate_);
  return *this;
}

std::unique_ptr<evolution::initial_data::InitialData>
SpecInitialData::get_clone() const {
  return std::make_unique<SpecInitialData>(*this);
}

SpecInitialData::SpecInitialData(CkMigrateMessage* msg) : InitialData(msg) {}

void SpecInitialData::pup(PUP::er& p) {
  InitialData::pup(p);
  p | data_directory_;
  if (p.isUnpacking()) {
    spec_exporter_ =
        std::make_unique<spec::Exporter>(sys::procs_on_node(sys::my_node()),
                                         data_directory_, vars_to_interpolate_);
  }
}

PUP::able::PUP_ID SpecInitialData::my_PUP_ID = 0;

template <typename DataType>
tuples::tagged_tuple_from_typelist<
    typename SpecInitialData::interpolated_tags<DataType>>
SpecInitialData::interpolate_from_spec(const tnsr::I<DataType, 3>& x) const {
  return gr::AnalyticData::interpolate_from_spec<interpolated_tags<DataType>>(
      make_not_null(spec_exporter_.get()), x,
      static_cast<size_t>(sys::my_local_rank()));
}

template tuples::tagged_tuple_from_typelist<
    typename SpecInitialData::interpolated_tags<double>>
SpecInitialData::interpolate_from_spec(const tnsr::I<double, 3>& x) const;
template tuples::tagged_tuple_from_typelist<
    typename SpecInitialData::interpolated_tags<DataVector>>
SpecInitialData::interpolate_from_spec(const tnsr::I<DataVector, 3>& x) const;

}  // namespace gr::AnalyticData

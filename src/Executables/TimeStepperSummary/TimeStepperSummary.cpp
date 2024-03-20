// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <algorithm>
#include <array>
#include <boost/program_options.hpp>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "Parallel/Printf/Printf.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

class ImexTimeStepper;
class LtsTimeStepper;
namespace TimeSteppers {
class AdamsBashforth;
template <bool Monotonic>
class AdamsMoultonPc;
}  // namespace TimeSteppers

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
using time_steppers_taking_order =
    tmpl::list<TimeSteppers::AdamsBashforth,
               TimeSteppers::AdamsMoultonPc<false>,
               TimeSteppers::AdamsMoultonPc<true>>;

const char* const program_help =
    "Print various properties about SpECTRE's time steppers.  Abbreviations\n"
    "used to reduce the output width are:\n"
    "  stab  = stable step size\n"
    "  subs  = substeps\n"
    "  esubs = substeps with error estimation enabled\n";

const std::array columns{"name"s,       "order"s, "subs"s,
                         "esubs"s,      "stab"s,  "stab/subs"s,
                         "stab/esubs"s, "IMEX"s,  "LTS"s};
using Row = std::tuple<std::string, size_t, size_t, size_t, double, double,
                       double, bool, bool>;

template <typename Stepper>
Row generate_row(const Stepper& stepper, std::string name) {
  return {std::move(name),
          stepper.order(),
          stepper.number_of_substeps(),
          stepper.number_of_substeps_for_error(),
          stepper.stable_step(),
          stepper.stable_step() / stepper.number_of_substeps(),
          stepper.stable_step() / stepper.number_of_substeps_for_error(),
          std::is_base_of_v<ImexTimeStepper, Stepper>,
          std::is_base_of_v<LtsTimeStepper, Stepper>};
}

std::vector<Row> generate_table() {
  std::vector<Row> table{};
  tmpl::for_each<TimeSteppers::time_steppers>([&](auto stepper_v) {
    using Stepper = tmpl::type_from<decltype(stepper_v)>;
    if constexpr (tmpl::list_contains_v<time_steppers_taking_order, Stepper>) {
      for (size_t order = Stepper::minimum_order;
           order <= Stepper::maximum_order;
           ++order) {
        table.push_back(generate_row(
            Stepper(order), MakeString{} << pretty_type::name<Stepper>() << "["
                                         << order << "]"));
      }
    } else {
      table.push_back(generate_row(Stepper{}, pretty_type::name<Stepper>()));
    }
  });
  return table;
}

std::string bool_yn(const bool& b) { return b ? "Y" : "N"; }
template <typename T>
const T& bool_yn(const T& t) {
  return t;
}

void print_table(std::vector<Row> table, const size_t sort_index) {
  tmpl::for_each<tmpl::range<size_t, 0, columns.size()>>(
      [&](auto constexpr_sort_index_v) {
        constexpr size_t constexpr_sort_index =
            tmpl::type_from<decltype(constexpr_sort_index_v)>::value;
        if (constexpr_sort_index == sort_index) {
          std::stable_sort(table.begin(), table.end(),
                           [&](const auto& a, const auto& b) {
                             return get<constexpr_sort_index>(a) <
                                    get<constexpr_sort_index>(b);
                           });
        }
      });

  std::array<size_t, columns.size()> column_widths{};
  for (size_t i = 0; i < columns.size(); ++i) {
    gsl::at(column_widths, i) = gsl::at(columns, i).size();
  }
  std::vector<std::array<std::string, columns.size()>> stringified_table;
  stringified_table.reserve(table.size());

  for (const auto& row : table) {
    stringified_table.emplace_back();
    tmpl::for_each<tmpl::range<size_t, 0, columns.size()>>([&](auto column_v) {
      constexpr size_t column = tmpl::type_from<decltype(column_v)>::value;
      std::string stringified =
          MakeString{} << std::setprecision(3) << bool_yn(get<column>(row));
      gsl::at(column_widths, column) =
          std::max(gsl::at(column_widths, column), stringified.size());
      gsl::at(stringified_table.back(), column) = std::move(stringified);
    });
  }

  for (size_t i = 0; i < columns.size(); ++i) {
    Parallel::printf("%-*s", gsl::at(column_widths, i) + 1,
                     gsl::at(columns, i));
  }
  Parallel::printf("\n");
  {
    const size_t num_chars =
        alg::accumulate(column_widths, 0_st) + columns.size();
    for (size_t i = 0; i < num_chars; ++i) {
      Parallel::printf("-");
    }
  }
  Parallel::printf("\n");
  for (const auto& row : stringified_table) {
    for (size_t i = 0; i < columns.size(); ++i) {
      Parallel::printf("%-*s", gsl::at(column_widths, i) + 1, gsl::at(row, i));
    }
    Parallel::printf("\n");
  }
}
}  // namespace

int main(const int argc, char** const argv) {
  namespace bpo = boost::program_options;
  try {
    bpo::options_description command_line_options;

    const std::string sort_help =
        MakeString{} << "Sort column.  One of " << columns;
    // clang-format off
    command_line_options.add_options()
        ("help,h", "Describe program options")
        ("sort", bpo::value<std::string>()->default_value(columns.front()),
         sort_help.c_str())
        ;
    // clang-format on

    bpo::command_line_parser command_line_parser(argc, argv);
    command_line_parser.options(command_line_options);

    bpo::variables_map parsed_command_line_options;
    bpo::store(command_line_parser.run(), parsed_command_line_options);
    bpo::notify(parsed_command_line_options);

    if (parsed_command_line_options.count("help") != 0) {
      Parallel::printf("%s\n%s", command_line_options, program_help);
      return 0;
    }

    const std::string sort_column =
        parsed_command_line_options.at("sort").as<std::string>();
    const auto sort_index =
        static_cast<size_t>(alg::find(columns, sort_column) - columns.begin());
    if (sort_index == columns.size()) {
      ERROR_NO_TRACE("Invalid sort column.  Must be " << columns);
    }

    print_table(generate_table(), sort_index);
  } catch (const bpo::error& e) {
    ERROR_NO_TRACE(e.what());
  }
  return 0;
}

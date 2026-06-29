// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <string>
#include <utility>

#include "Executables/ExportEquationOfStateForRotNS/DumpRotNSEos.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

// Charm looks for this function but since we build without a main function or
// main module supplied by Charm++, we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
namespace OptionTags {
struct NumberOfPoints {
  using type = size_t;
  static constexpr Options::String help = {
      "Number of points at which to dump the EoS"};
};

struct OutputFileName {
  using type = std::string;
  static constexpr Options::String help = {
      "Name of the output file to dump the EoS to, including file extension."};
};

struct LowerBoundRestMassDensityCgs {
  using type = double;
  static constexpr Options::String help = {
      "Lower bound of rest mass density in CGS units."};
};

struct UpperBoundRestMassDensityCgs {
  using type = double;
  static constexpr Options::String help = {
      "Upper bound of rest mass density in CGS units."};
};
}  // namespace OptionTags
}  // namespace

int main(int argc, char** argv) {
  namespace bpo = boost::program_options;
  bpo::positional_options_description pos_desc;

  const std::string help_string =
      "Dump a relativistic equation of state to disk.\n"
      "All options controlling input and output are read from the input file.";

  bpo::options_description desc(help_string);
  desc.add_options()("help,h,", "show this help message")(
      "input-file", bpo::value<std::string>()->required(), "Input file name")(
      "check-options", "Check input file options");

  bpo::variables_map vars;

  bpo::store(bpo::command_line_parser(argc, argv)
                 .positional(pos_desc)
                 .options(desc)
                 .run(),
             vars);

  if (vars.count("help") != 0u or vars.count("input-file") == 0u) {
    Parallel::printf("%s\n", desc);
    return 1;
  }

  using option_list =
      tmpl::list<hydro::OptionTags::InitialDataEquationOfState<true, 3>,
                 OptionTags::NumberOfPoints, OptionTags::OutputFileName,
                 OptionTags::LowerBoundRestMassDensityCgs,
                 OptionTags::UpperBoundRestMassDensityCgs>;

  Options::Parser<option_list> option_parser(help_string);
  option_parser.parse_file(vars["input-file"].as<std::string>());

  if (vars.count("check-options") != 0) {
    // Force all the options to be created.
    option_parser.template apply<option_list>([](const auto&... args) {
      (void)std::initializer_list<char>{((void)args, '0')...};
    });
    Parallel::printf("\n%s parsed successfully!\n",
                     vars["input-file"].as<std::string>());

    return 0;
  }

  const auto options =
      option_parser.template apply<option_list>([](auto... args) {
        return tuples::tagged_tuple_from_typelist<option_list>(
            std::move(args)...);
      });

  ExportEosForRotNS::dump_eos(
      *get<hydro::OptionTags::InitialDataEquationOfState<true, 3>>(options),
      get<OptionTags::NumberOfPoints>(options),
      get<OptionTags::OutputFileName>(options),
      get<OptionTags::LowerBoundRestMassDensityCgs>(options),
      get<OptionTags::UpperBoundRestMassDensityCgs>(options));

  return 0;
}

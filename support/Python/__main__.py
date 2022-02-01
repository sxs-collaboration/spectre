# Distributed under the MIT License.
# See LICENSE.txt for details.

SPECTRE_VERSION = "@SPECTRE_VERSION@"


def main():
    import argparse

    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        prog='spectre',
        description="SpECTRE version: {}".format(SPECTRE_VERSION),
        formatter_class=HelpFormatter)

    # Version endpoint
    parser.add_argument('--version', action='version', version=SPECTRE_VERSION)

    parser.parse_args()
    raise NotImplementedError("No subprograms are implemented yet.")


if __name__ == '__main__':
    main()

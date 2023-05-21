#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import datetime
import difflib
import logging
import operator
import os
import pathlib
import re
import tempfile
import textwrap
import urllib
from typing import List

import git
import pybtex.database
import requests
import uplink
import yaml
from cffconvert.citation import Citation
from pybtex.backends.plaintext import Backend as PlaintextBackend
from pybtex.style.formatting.plain import Style as PlainStyle

logger = logging.getLogger(__name__)

VERSION_PATTERN = r"(\d{4})\.(\d{2})\.(\d{2})(\.\d+)?"
PUBLICATION_DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"
DOI_PATTERN = r"10\.\d{4,9}/zenodo\.\d+"
ZENODO_ID_PATTERN = r"\d+"


def report_check_only(msg: str):
    logger.info(f"CHECK ONLY: {msg}")


def new_version_id_from_response(response):
    """Retrieves the ID of the new version draft from the API response

    The "New version" action of the Zenodo API returns the ID of the created
    version draft in the 'links' section, as documented
    [here](https://developers.zenodo.org/#new-version). This function parses the
    ID out of the link.
    """
    new_version_url = response.json()["links"]["latest_draft"]
    return int(
        pathlib.PurePosixPath(
            urllib.parse.urlparse(new_version_url).path
        ).parts[-1]
    )


def raise_for_status(response):
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if response.status_code >= 400 and response.status_code < 500:
            raise requests.exceptions.HTTPError(
                yaml.safe_dump(response.json(), allow_unicode=True)
            ) from err
        else:
            raise
    return response


@uplink.response_handler(raise_for_status)
class Zenodo(uplink.Consumer):
    """Abstraction of the [Zenodo API](https://developers.zenodo.org)"""

    @uplink.returns.json
    @uplink.get("deposit/depositions/{id}")
    def get_deposition(self, id: uplink.Path):
        """Retrieves a deposition by ID."""
        pass

    @uplink.returns.json
    @uplink.get("records/{id}")
    def get_record(self, id: uplink.Path):
        """Retrieves a published record by ID."""
        pass

    @uplink.returns.json(
        key=("metadata", "relations", "version", 0, "last_child", "pid_value"),
        type=int,
    )
    @uplink.get("records/{id}")
    def get_latest_version_id(self, record_id: uplink.Path(name="id")):
        """Retrieves only the latest version ID of a record."""
        pass

    @uplink.response_handler(new_version_id_from_response)
    @uplink.post("deposit/depositions/{id}/actions/newversion")
    def new_version(self, latest_version_id: uplink.Path(name="id")):
        """Invoke the "New version" action on a deposition.

        Returns:
          The ID of the new version.
        """
        pass

    @uplink.returns.json
    @uplink.post("deposit/depositions/{id}/actions/publish")
    def publish(self, id: uplink.Path):
        """Invoke the "Publish" action on a deposition."""
        pass

    @uplink.json
    @uplink.returns.json
    @uplink.put("deposit/depositions/{id}")
    def update_deposition(self, id: uplink.Path, **body: uplink.Body):
        """Update the deposition with the metadata"""
        pass

    @uplink.returns.json
    @uplink.put("files/{bucket_id}/{filename}")
    def upload_file(
        self, bucket_id: uplink.Path, filename: uplink.Path, file: uplink.Body
    ):
        """Uploads a file to the bucket."""
        pass


@uplink.response_handler(raise_for_status)
class Github(uplink.Consumer):
    """Abstraction of the [GitHub REST API](https://docs.github.com/en/rest)"""

    # This endpoint is not currently implemented in PyGithub, so we wrap it
    # here manually
    @uplink.headers({"Content-Type": "text/x-markdown"})
    @uplink.response_handler(operator.attrgetter("text"))
    @uplink.post("markdown/raw")
    def render_markdown_raw(self, text: uplink.Body):
        """Render Markdown in plain format like a README.md file on GitHub"""
        pass

    @uplink.returns.json
    @uplink.get("repos/{user}/{repo}/releases/tags/{tag}")
    def get_release_by_tag(
        self, user: uplink.Path, repo: uplink.Path, tag: uplink.Path
    ):
        pass

    @uplink.returns.json
    @uplink.get("repos/{user}/{repo}/releases/{release_id}/assets")
    def get_assets(
        self, user: uplink.Path, repo: uplink.Path, release_id: uplink.Path
    ):
        pass


def to_plaintext_reference(
    bib_entry: pybtex.database.Entry,
    style=PlainStyle(),
    backend=PlaintextBackend(),
) -> str:
    # pybtex doesn't support 'software' bibtex types for some reason:
    # https://bitbucket.org/pybtex-devs/pybtex/issues/157/support-for-software-and-patent-entries-in
    if bib_entry.type == "software":
        bib_entry.type = "misc"
    return style.format_entry(label=bib_entry.key, entry=bib_entry).text.render(
        backend
    )


def collect_zenodo_metadata(
    metadata: dict, references: List[pybtex.database.Entry], github: Github
) -> dict:
    """Produces the metadata that we send to Zenodo

    Args:
      metadata: The project metadata read from the YAML file. This is the main
        source of information for this function.
      references: List of references resolved from 'metadata["References"]'.
        They will be formatted and sent to Zenodo.
      github: The GitHub API client. We use it to render the description to
        HTML in a way that's consistent with GitHub's rendering.

    Returns:
      Metadata in the format that the Zenodo API expects.
    """
    # Generate the DOI author list from the authors in the project metadata
    zenodo_creators = []
    for author_tier in ["Core", "Developers", "Contributors"]:
        for author in metadata["Authors"][author_tier]["List"]:
            zenodo_creator = dict(name=author["Name"])
            if "Orcid" in author:
                zenodo_creator["orcid"] = author["Orcid"]
            if "Affiliations" in author and len(author["Affiliations"]) > 0:
                zenodo_creator["affiliation"] = " and ".join(
                    author["Affiliations"]
                )
            zenodo_creators.append(zenodo_creator)
    # Render the description to HTML
    rendered_description = github.render_markdown_raw(metadata["Description"])
    # Format references as plain text for Zenodo
    formatted_references = [
        to_plaintext_reference(entry) for entry in references
    ]
    # Construct Zenodo metadata
    return dict(
        title=metadata["Name"],
        version=metadata["Version"],
        publication_date=metadata["PublicationDate"].isoformat(),
        doi=metadata["Doi"],
        description=rendered_description,
        creators=zenodo_creators,
        related_identifiers=[
            dict(
                identifier=metadata["Homepage"],
                relation="isDocumentedBy",
                scheme="url",
                resource_type="publication-softwaredocumentation",
            ),
            dict(
                identifier=("https://github.com/" + metadata["GitHub"]),
                relation="isSupplementTo",
                scheme="url",
                resource_type="software",
            ),
        ],
        language="eng",
        communities=[dict(identifier="sxs")],
        keywords=metadata["Keywords"],
        license=metadata["License"],
        upload_type="software",
        access_right="open",
        references=formatted_references,
    )


def to_cff_person(person: pybtex.database.Person) -> dict:
    """BibTeX to CFF conversion for person objects.

    The format is defined here:
    https://github.com/citation-file-format/citation-file-format/blob/main/schema-guide.md#definitionsperson
    """
    # Map BibTeX to CFF fields
    name_fields = {
        "last": "family-names",
        "bibtex_first": "given-names",
        "prelast": "name-particle",
        "lineage": "name-suffix",
    }
    result = {
        cff_field: " ".join(person.get_part(bibtex_field))
        for bibtex_field, cff_field in name_fields.items()
        if person.get_part(bibtex_field)
    }
    # Use CFF "entity" format if BibTex has no first & last names
    if list(result.keys()) == ["family-names"]:
        return {"name": result["family-names"]}
    return result


def to_cff_reference(bib_entry: pybtex.database.Entry) -> dict:
    """BibTeX to CFF conversion for references.

    The format is defined here:
    https://github.com/citation-file-format/citation-file-format/blob/main/schema-guide.md#definitionsreference
    """

    def _cff_transform(cff_field, bib_value):
        if cff_field == "type":
            if bib_value == "inproceedings":
                return "article"
            elif bib_value == "incollection":
                return "article"
        elif cff_field == "publisher":
            return {"name": bib_value}
        elif cff_field == "month":
            try:
                return int(bib_value)
            except ValueError:
                return {
                    "jan": 1,
                    "feb": 2,
                    "mar": 3,
                    "apr": 4,
                    "may": 5,
                    "jun": 6,
                    "jul": 7,
                    "aug": 8,
                    "sep": 9,
                    "oct": 10,
                    "nov": 11,
                    "dec": 12,
                }[bib_value[:3].lower()]
        return bib_value

    cff_reference = {
        "type": _cff_transform(cff_field="type", bib_value=bib_entry.type),
        "authors": [
            to_cff_person(person) for person in bib_entry.persons["author"]
        ],
    }
    # Map BibTeX to CFF fields. This is just a subset of the most relevant
    # fields.
    fields = {
        "doi": "doi",
        "edition": "edition",
        "isbn": "isbn",
        "license": "license",
        "month": "month",
        "number": "number",
        "pages": "pages",
        "publisher": "publisher",
        "title": "title",
        "url": "url",
        "version": "version",
        "volume": "volume",
        "year": "year",
        "booktitle": "collection-title",
    }
    for bibtex_field, value in bib_entry.fields.items():
        bibtex_field = bibtex_field.lower()
        if bibtex_field in fields:
            cff_field = fields[bibtex_field]
            cff_reference[cff_field] = _cff_transform(
                cff_field=cff_field, bib_value=value
            )
    return cff_reference


def collect_citation_metadata(
    metadata: dict, references: List[pybtex.database.Entry]
) -> dict:
    """Produces the data stored in the CITATION.cff file

    Args:
      metadata: The project metadata read from the YAML file. This is the main
        source of information for this function.
      references: List of references resolved from 'metadata["References"]'.
        They will be converted to CFF.

    Returns:
      Citation data in the [Citation File Format](https://github.com/citation-file-format/citation-file-format)
    """
    # Author list
    citation_authors = []
    for author_tier in ["Core", "Developers", "Contributors"]:
        for author in metadata["Authors"][author_tier]["List"]:
            family_names, given_names = author["Name"].split(", ")
            citation_author = {
                "family-names": family_names,
                "given-names": given_names,
            }
            if "Orcid" in author:
                citation_author["orcid"] = (
                    "https://orcid.org/" + author["Orcid"]
                )
            if "Affiliations" in author and len(author["Affiliations"]) > 0:
                citation_author["affiliation"] = " and ".join(
                    author["Affiliations"]
                )
            citation_authors.append(citation_author)
    # References in CITATION.cff format
    citation_references = [to_cff_reference(entry) for entry in references]
    return {
        "cff-version": "1.2.0",
        "message": (
            "Please cite SpECTRE in any publications that make use of its code"
            " or data. Cite the latest version that you use in your"
            " publication. The citation for this version is listed below."
        ),
        "title": metadata["Name"],
        "url": metadata["Homepage"],
        "repository-code": "https://github.com/" + metadata["GitHub"],
        "version": metadata["Version"],
        "date-released": metadata["PublicationDate"],
        "doi": metadata["Doi"],
        "authors": citation_authors,
        "keywords": metadata["Keywords"],
        "license": metadata["License"],
        "references": citation_references,
    }


def build_bibtex_entry(metadata: dict):
    """Builds a BibTeX entry that we suggest people cite in publications

    Args:
      metadata: The project metadata read from the YAML file

    Returns:
      A pybtex.database.Entry. Use the `to_string` member function to convert to
      a string in BibTeX, YAML or other formats.
    """
    # We truncate the author list in the BibTeX entry after 'Developers',
    # because not all journals are happy with printing an excessively long
    # author list, e.g. Phys. Rev. wants at most 15 authors. By truncating the
    # author list here, the user who copies the BibTeX entry doesn't have to
    # make the decision where to truncate.
    authors = [
        pybtex.database.Person(author["Name"])
        for author in (
            metadata["Authors"]["Core"]["List"]
            + metadata["Authors"]["Developers"]["List"]
        )
    ] + [pybtex.database.Person("others")]
    entry = pybtex.database.Entry(
        "software",
        persons=dict(author=authors),
        fields=dict(
            title=(
                r"\texttt{"
                + metadata["Name"]
                + " v"
                + metadata["Version"]
                + "}"
            ),
            # The 'version' field is not used by revtex4-2, so we also put the
            # version in the title
            version=metadata["Version"],
            publisher="Zenodo",
            doi=metadata["Doi"],
            url=metadata["Homepage"],
            howpublished=(
                r"\href{https://doi.org/"
                + metadata["Doi"]
                + "}{"
                + metadata["Doi"]
                + "}"
            ),
            license=metadata["License"],
            year=str(metadata["PublicationDate"].year),
            month=str(metadata["PublicationDate"].month),
        ),
    )
    entry.key = "spectrecode"
    return entry


def prepare(
    metadata: dict,
    version_name: str,
    metadata_file: str,
    citation_file: str,
    bib_file: str,
    references_file: str,
    readme_file: str,
    zenodo: Zenodo,
    github: Github,
    update_only: bool,
    check_only: bool,
):
    # Validate new version name
    match_version_name = re.match(VERSION_PATTERN + "$", version_name)
    if not match_version_name:
        raise ValueError(
            f"Version name '{version_name}' doesn't match "
            f"pattern '{VERSION_PATTERN}'."
        )
    publication_date = datetime.date(
        year=int(match_version_name.group(1)),
        month=int(match_version_name.group(2)),
        day=int(match_version_name.group(3)),
    )

    if update_only:
        # Don't try to create a new version draft on Zenodo but update the
        # existing one. We assume that the metadata in the repository already
        # point to the existing version draft on Zenodo that we want to update.
        # This is the case when the user has run this script without the
        # `--update-only` option before and has thus created the new version
        # draft on Zenodo, and is now running it again with the `--update-only`
        # option to push updated metadata to the draft.
        new_version_id = metadata["ZenodoId"]
    else:
        # Zenodo doesn't have a draft for the new version yet, or the metadata
        # in the repository is not yet updated. Either way, we use the ID from
        # the metadata to obtain the latest version on Zenodo and create a new
        # draft. Zenodo doesn't create another draft if one already exists, but
        # just returns it.
        latest_version_id = metadata["ZenodoId"]
        try:
            latest_version_id_on_zenodo = zenodo.get_latest_version_id(
                record_id=latest_version_id
            )
        except requests.exceptions.HTTPError as err:
            raise requests.exceptions.HTTPError(
                f"No published record with ID {latest_version_id} found on "
                "Zenodo. Use the '--update-only' flag if you're re-running "
                "the script over a repository that already has an unpublished "
                "new version ID inserted into Metadata.yaml."
            ) from err
        assert latest_version_id == latest_version_id_on_zenodo, (
            "The latest Zenodo version ID in the repository is "
            f"{latest_version_id}, but Zenodo "
            f"reports {latest_version_id_on_zenodo}."
        )
        logger.info(f"The latest Zenodo version ID is {latest_version_id}.")
        # Reserve a DOI by creating a new version on Zenodo. It will remain a
        # draft until we publish it in the `publish` subprogram.
        if check_only:
            report_check_only(
                "Would create new version on Zenodo with "
                f"ID {latest_version_id}."
            )
            new_version_id = latest_version_id
        else:
            new_version_id = zenodo.new_version(
                latest_version_id=latest_version_id
            )
    new_version_draft = zenodo.get_deposition(id=new_version_id)
    new_version_doi = new_version_draft["doi"]
    assert new_version_doi, (
        "Zenodo did not return a reserved DOI for the new version draft. "
        "You may want to visit {} to reserve one, save the draft and re-run "
        "this script with the '--update-only' flag."
    ).format(new_version_draft["links"]["html"])
    logger.info(
        "The new version draft on Zenodo has "
        f"ID {new_version_id} and DOI {new_version_doi}."
    )

    # Insert the new version information into the metadata file. We have to
    # resort to regex-replacements because pyyaml doesn't support
    # format-preserving round-trips.
    def replace_in_yaml(content, key, value, validate_value_pattern):
        content, num_subs = re.subn(
            r"^{}: {}$".format(key, validate_value_pattern),
            r"{}: {}".format(key, value),
            content,
            flags=re.MULTILINE,
        )
        if num_subs == 0:
            match = re.search(
                r"^{}: (.*)$".format(key), content, flags=re.MULTILINE
            )
            if match:
                raise ValueError(
                    f"The value of '{key}' in the file '{metadata_file}' "
                    f"does not match the pattern '{validate_value_pattern}': "
                    f"{match.group(1)}"
                )
            else:
                raise ValueError(
                    f"Could not find '{key}' in root of file '{metadata_file}'."
                )
        elif num_subs > 1:
            raise ValueError(
                f"Found more than one '{key}' in file '{metadata_file}'."
            )
        return content

    with open(metadata_file, "r" if check_only else "r+") as open_metadata_file:
        content_original = open_metadata_file.read()
        content_new = replace_in_yaml(
            content_original, "Version", version_name, VERSION_PATTERN
        )
        content_new = replace_in_yaml(
            content_new,
            "PublicationDate",
            publication_date.isoformat(),
            PUBLICATION_DATE_PATTERN,
        )
        content_new = replace_in_yaml(
            content_new, "Doi", new_version_doi, DOI_PATTERN
        )
        content_new = replace_in_yaml(
            content_new, "ZenodoId", new_version_id, ZENODO_ID_PATTERN
        )
        content_diff = "\n".join(
            difflib.context_diff(
                content_original.split("\n"),
                content_new.split("\n"),
                lineterm="",
                fromfile=metadata_file,
                tofile=metadata_file,
            )
        )
        if check_only:
            report_check_only(f"Would apply diff:\n{content_diff}")
        else:
            logger.debug(f"Applying diff:\n{content_diff}")
            open_metadata_file.seek(0)
            open_metadata_file.write(content_new)
            open_metadata_file.truncate()
    logger.info(f"Inserted new version info into '{metadata_file}'.")
    # Also update the the metadata dict to make sure we don't accidentally use
    # the old values somewhere
    metadata["Version"] = version_name
    metadata["PublicationDate"] = publication_date
    metadata["Doi"] = new_version_doi
    metadata["ZenodoId"] = new_version_id

    def write_file_or_check(filename, content_new):
        with open(filename, "r" if check_only else "r+") as open_file:
            content_original = open_file.read()
            content_diff = "\n".join(
                difflib.context_diff(
                    content_original.split("\n"),
                    content_new.split("\n"),
                    lineterm="",
                    fromfile=filename,
                    tofile=filename,
                )
            )
            if check_only:
                report_check_only(f"Would apply diff:\n{content_diff}")
            else:
                logger.debug(f"Applying diff:\n{content_diff}")
                open_file.seek(0)
                open_file.write(content_new)
                open_file.truncate()

    # Write the CITATION.cff file
    reference_keys = metadata["References"]["List"]
    all_references = pybtex.database.parse_file(references_file)
    references = [all_references.entries[key] for key in reference_keys]
    citation_data = collect_citation_metadata(metadata, references)
    citation_file_content = """# Distributed under the MIT License.
# See LICENSE.txt for details.

# This file is automatically generated. It will be overwritten at every
# release. See .github/scripts/Release.py for details.

"""
    citation_file_content += yaml.safe_dump(citation_data, allow_unicode=True)
    write_file_or_check(citation_file, citation_file_content)
    # Validate the CFF file
    Citation(citation_file_content, src=citation_file).validate()

    # Get the BibTeX entry and write to file
    bibtex_entry = build_bibtex_entry(metadata)
    bib_file_content = bibtex_entry.to_string(
        "bibtex",
        # LaTeX-encode special characters, instead of writing unicode symbols
        encoding="ASCII",
    )
    # Fix an issue with encoded accents (a space in the author list is
    # recognized as delimiter between names)
    bib_file_content = bib_file_content.replace(r"\c c", r"\c{c}")
    # Wrap long lines in BibTeX output
    bib_file_content = "\n".join(
        [textwrap.fill(line, width=80) for line in bib_file_content.split("\n")]
    )
    write_file_or_check(bib_file, bib_file_content)

    # Insert the new version information into the README
    def replace_badge_in_readme(content, key, image_url, link_url):
        content, num_subs = re.subn(
            r"\[!\[{}\]\(.*\)\]\(.*\)".format(key),
            r"[![{}]({})]({})".format(key, image_url, link_url),
            content,
            flags=re.MULTILINE,
        )
        assert (
            num_subs > 0
        ), f"Could not find badge '{key}' in file '{readme_file}'."
        return content

    def replace_doi_in_readme(content, doi, doi_url):
        content, num_subs = re.subn(
            r"DOI: \[{}\]\(.*\)".format(DOI_PATTERN),
            r"DOI: [{}]({})".format(doi, doi_url),
            content,
            flags=re.MULTILINE,
        )
        assert (
            num_subs > 0
        ), "Could not find DOI (matching '{}') with link in file '{}'.".format(
            DOI_PATTERN, readme_file
        )
        return content

    def replace_link_in_readme(content, link_text, link_url):
        content, num_subs = re.subn(
            r"\[{}\]\(.*\)".format(link_text),
            r"[{}]({})".format(link_text, link_url),
            content,
            flags=re.MULTILINE,
        )
        assert num_subs > 0, (
            f"Could not find link with text '{link_text}' in "
            f"file '{readme_file}'."
        )
        return content

    def replace_bibtex_entry_in_readme(content, bibtex_entry):
        bibtex_entry_string = bib_file_content
        # Work around an issue with escaping LaTeX commands
        bibtex_entry_string = bibtex_entry_string.replace("\\", "\\\\")
        FENCE_PATTERN = "<!-- BIBTEX ENTRY -->"
        content, num_subs = re.subn(
            (FENCE_PATTERN + "(.*)" + FENCE_PATTERN),
            (
                FENCE_PATTERN
                + "\n```bib\n"
                + bibtex_entry_string.strip()
                + "\n```\n"
                + FENCE_PATTERN
            ),
            content,
            flags=re.DOTALL,
        )
        assert (
            num_subs > 0
        ), f"Could not find a BibTeX entry in file '{readme_file}'."
        return content

    with open(readme_file, "r" if check_only else "r+") as open_readme_file:
        content_original = open_readme_file.read()
        content = replace_badge_in_readme(
            content_original,
            "release",
            f"https://img.shields.io/badge/release-v{version_name}-informational",
            "https://github.com/{}/releases/tag/v{}".format(
                metadata["GitHub"], version_name
            ),
        )
        content = replace_badge_in_readme(
            content,
            "DOI",
            new_version_draft["links"]["badge"],
            new_version_draft["links"]["doi"],
        )
        content = replace_doi_in_readme(
            content, new_version_doi, new_version_draft["links"]["doi"]
        )
        # We don't currently link to the Zenodo BibTeX entry because it isn't
        # very good. Instead, we generate our own.
        content = replace_bibtex_entry_in_readme(content, bibtex_entry)
        content_diff = "\n".join(
            difflib.context_diff(
                content_original.split("\n"),
                content.split("\n"),
                lineterm="",
                fromfile=readme_file,
                tofile=readme_file,
            )
        )
        if check_only:
            report_check_only(f"Would apply diff:\n{content_diff}")
        else:
            logger.debug(f"Applying diff:\n{content_diff}")
            open_readme_file.seek(0)
            open_readme_file.write(content)
            open_readme_file.truncate()

    # Upload the updated metadata to Zenodo
    zenodo_metadata = collect_zenodo_metadata(metadata, references, github)
    logger.debug(
        "The metadata we'll send to Zenodo are:\n{}".format(
            yaml.safe_dump(zenodo_metadata, allow_unicode=True)
        )
    )
    if check_only:
        report_check_only("Would upload metadata to Zenodo.")
    else:
        zenodo.update_deposition(id=new_version_id, metadata=zenodo_metadata)
    logger.debug(
        (
            "New Zenodo version draft is now prepared. You can edit "
            "it here:\n{}"
        ).format(new_version_draft["links"]["html"])
    )


def publish(
    metadata: dict,
    zenodo: Zenodo,
    github: Github,
    auto_publish: bool,
    check_only: bool,
):
    version_name = metadata["Version"]
    new_version_id = metadata["ZenodoId"]

    # Retrieve the Zenodo deposition for the version draft that we have
    # prepared before
    new_version_draft = zenodo.get_deposition(id=new_version_id)

    # Retrieve the file "bucket" ID for uploading data
    bucket_id = pathlib.PurePosixPath(
        urllib.parse.urlparse(new_version_draft["links"]["bucket"]).path
    ).parts[-1]

    # Retrieve the URL of the GitHub release archive that we want to upload
    # to Zenodo
    gh_user, gh_repo = metadata["GitHub"].split("/")
    # Alternatively we could use the release ID that GitHub's
    # 'actions/create-release' returns to retrieve the release
    gh_release = github.get_release_by_tag(
        user=gh_user, repo=gh_repo, tag="v" + version_name
    )
    logger.debug(
        "The release on GitHub is:\n{}".format(
            yaml.safe_dump(gh_release, allow_unicode=True)
        )
    )
    zipball_url = gh_release["zipball_url"]

    # Stream the release archive to Zenodo.
    # We keep the file name for the archive on Zenodo the same for each
    # release so we can just overwrite it. Note that the _unpacked_ directory
    # name contains the version as expected, since the unpacked directory name
    # is determined by the GitHub release.
    archive_filename = gh_repo + ".zip"
    if check_only:
        report_check_only(
            f"Would stream release zipball '{zipball_url}' as "
            f"filename '{archive_filename}' to bucket '{bucket_id}'."
        )
    else:
        # Download the zipball from GitHub, then upload to Zenodo.
        # Note: Something like this should also work to stream the file
        # directly from GitHub to Zenodo without temporarily saving it, but
        # Zenodo doesn't currently document their new "bucket" file API so it
        # is difficult to debug:
        # with requests.get(zipball_url, stream=True) as zipball_stream:
        #     zipball_stream.raise_for_status()
        #     uploaded_file = zenodo.upload_file(bucket_id=bucket_id,
        #                                        file=zipball_stream,
        #                                        filename=archive_filename)
        zipball_download = requests.get(zipball_url, stream=True)
        with tempfile.TemporaryFile() as open_tmp_file:
            for chunk in zipball_download.iter_content():
                open_tmp_file.write(chunk)
            open_tmp_file.seek(0)
            uploaded_file = zenodo.upload_file(
                bucket_id=bucket_id,
                file=open_tmp_file,
                filename=archive_filename,
            )
        logger.debug(
            "Release archive upload complete:\n{}".format(
                yaml.safe_dump(uploaded_file, allow_unicode=True)
            )
        )

    # Publish!
    if auto_publish:
        if check_only:
            report_check_only(
                f"Would publish Zenodo record {new_version_id} now!"
            )
        else:
            published_record = zenodo.publish(id=new_version_id)
            logger.debug(
                "Zenodo record published:\n{}".format(
                    yaml.safe_dump(published_record, allow_unicode=True)
                )
            )
            logger.info(
                (
                    "Zenodo record is now public! Here's the link to the "
                    "record:\n{}"
                ).format(published_record["links"]["record_html"])
            )
    else:
        logger.info(
            (
                "Release is ready to be published on Zenodo. Go to this "
                "website, make sure everything looks fine and then hit the "
                "'Publish' button:\n{}"
            ).format(new_version_draft["links"]["html"])
        )


if __name__ == "__main__":
    # Always work with the repository that contains this file
    repo = git.Repo(__file__, search_parent_directories=True)

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the repository and publish releases on Zenodo as part of"
            " the automatic versioning procedure. This script is not intended"
            " to be run outside of GitHub actions. The 'prepare' subprogram"
            " reserves a DOI on Zenodo and inserts it into the repository"
            " along with the new version name. Once the release archive has"
            " been created, the 'publish'subprogram uploads it to Zenodo."
            f" Repository: {repo.working_dir}."
        )
    )
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--zenodo-token",
        required=True,
        help=(
            "Zenodo access token. Refer to the Zenodo documentation "
            "for instructions on creating a personal access token."
        ),
    )
    parent_parser.add_argument(
        "--zenodo-sandbox",
        action="store_true",
        help="Use the Zenodo sandbox instead of the public version of Zenodo",
    )
    parent_parser.add_argument(
        "--github-token",
        required=False,
        help=(
            "Access token for GitHub queries. Refer to the GitHub documentation"
            " for instructions on creating a personal access token."
        ),
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, ...)",
    )
    parent_parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Dry mode, only check that all files are consistent. Nothing is"
            " edited or uploaded to Zenodo. Used in CI tests to make sure"
            " changes to the repository remain compatible with this script."
        ),
    )
    subparsers = parser.add_subparsers()
    parser_prepare = subparsers.add_parser("prepare", parents=[parent_parser])
    parser_prepare.set_defaults(subprogram=prepare)
    parser_prepare.add_argument(
        "--update-only",
        action="store_true",
        help=(
            "Only update an existing version draft on Zenodo, not creating a "
            "new one. Use this flag if the metadata in the repository already "
            "reference the new version draft on Zenodo."
        ),
    )
    parser_prepare.add_argument(
        "--version-name",
        required=False,
        help=(
            "The name of the new version. Will be inserted into the "
            "'--metadata-file'. Required unless '--check-only'."
        ),
    )
    parser_publish = subparsers.add_parser("publish", parents=[parent_parser])
    parser_publish.set_defaults(subprogram=publish)
    parser_publish.add_argument(
        "--auto-publish",
        action="store_true",
        help=(
            "Publish the Zenodo record once it's ready. "
            "WARNING: Published records cannot be deleted and editing is "
            "limited. Omit this argument to print out the link to the "
            "prepared draft on Zenodo so you can do a manual sanity-check "
            "before publishing it."
        ),
    )
    args = parser.parse_args()

    # Set the log level
    logging.basicConfig(level=logging.WARNING - args.verbose * 10)
    logging.getLogger("pykwalify").setLevel(logging.CRITICAL)
    del args.verbose

    # Load the project metadata
    metadata_file = os.path.join(repo.working_dir, "Metadata.yaml")
    args.metadata = yaml.safe_load(open(metadata_file, "r"))
    if args.subprogram == prepare:
        args.metadata_file = metadata_file

    # Make passing a version name optional in check-only mode
    if args.subprogram == prepare:
        assert args.check_only or args.version_name, (
            "The '--version-name' argument is required unless you run in "
            "'--check-only' mode."
        )
        if args.check_only and not args.version_name:
            args.version_name = args.metadata["Version"]

    # Locate project files
    if args.subprogram == prepare:
        args.readme_file = os.path.join(repo.working_dir, "README.md")
        args.citation_file = os.path.join(repo.working_dir, "CITATION.cff")
        args.bib_file = os.path.join(repo.working_dir, "citation.bib")
        args.references_file = os.path.join(
            repo.working_dir, args.metadata["References"]["BibliographyFile"]
        )

    # Configure the Zenodo API client
    args.zenodo = Zenodo(
        base_url=(
            "https://sandbox.zenodo.org/api/"
            if args.zenodo_sandbox
            else "https://zenodo.org/api/"
        ),
        auth=uplink.auth.BearerToken(args.zenodo_token),
    )
    del args.zenodo_sandbox
    del args.zenodo_token

    # Configure the GitHub API client
    args.github = Github(
        base_url="https://api.github.com/",
        auth=(
            uplink.auth.BearerToken(args.github_token)
            if args.github_token
            else None
        ),
    )
    del args.github_token

    # Dispatch to the selected subprogram
    subprogram = args.subprogram
    del args.subprogram
    subprogram(**vars(args))

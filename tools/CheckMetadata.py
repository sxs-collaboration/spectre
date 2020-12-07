#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import datetime
import git
import os
import re
import unicodedata
import unittest
import urllib
import yaml

VERSION_PATTERN = r'(\d{4})\.(\d{2})\.(\d{2})(\.\d+)?'
DOI_PATTERN = r'10\.\d{4,9}/zenodo\.(\d+)'
ORCID_PATTERN = r'\d{4}-\d{4}-\d{4}-\d{3}[0-9X]'


def simplify_name(name):
    """Converts the `name` to lower-case ASCII for fuzzy comparisons."""
    return unicodedata.normalize('NFKD',
                                 name.lower()).encode('ascii', 'ignore')


class TestMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Always work with the repository that contains this file
        try:
            cls.repo = git.Repo(__file__, search_parent_directories=True)
        except git.exc.InvalidGitRepositoryError as err:
            raise git.exc.InvalidGitRepositoryError(
                (f"The file '{__file__}' needs to be in a git repository to "
                 "find the corresponding Metadata.yaml file.")) from err
        with open(os.path.join(cls.repo.working_dir, 'Metadata.yaml'),
                  'r') as open_metadata_file:
            cls.metadata = yaml.safe_load(open_metadata_file)

    def test_name(self):
        self.assertIn('Name', self.metadata)
        self.assertTrue(isinstance(self.metadata['Name'], str))

    def test_license(self):
        self.assertIn('License', self.metadata)
        self.assertTrue(isinstance(self.metadata['License'], str))

    def test_homepage(self):
        self.assertIn('Homepage', self.metadata)
        homepage = self.metadata['Homepage']
        homepage_components = urllib.parse.urlparse(homepage)
        self.assertTrue(homepage_components.scheme)
        self.assertTrue(homepage_components.netloc)

    def test_github(self):
        self.assertIn('GitHub', self.metadata)
        github = self.metadata['GitHub']
        gh_user, gh_repo = github.split('/')
        self.assertTrue(gh_user)
        self.assertTrue(gh_repo)

    def test_version(self):
        self.assertIn('Version', self.metadata)
        version = self.metadata['Version']
        self.assertRegex(version, '^' + VERSION_PATTERN + '$')

    def test_publication_date(self):
        self.assertIn('PublicationDate', self.metadata)
        publication_date = self.metadata['PublicationDate']
        self.assertTrue(isinstance(publication_date, datetime.date))
        version_year, version_month, version_day = re.match(
            VERSION_PATTERN, self.metadata['Version']).groups()[:3]
        self.assertEqual(publication_date.year, int(version_year))
        self.assertEqual(publication_date.month, int(version_month))
        self.assertEqual(publication_date.day, int(version_day))

    def test_doi(self):
        self.assertIn('Doi', self.metadata)
        doi = self.metadata['Doi']
        self.assertRegex(doi, '^' + DOI_PATTERN + '$')

    def test_zenodo_id(self):
        self.assertIn('ZenodoId', self.metadata)
        zenodo_id = self.metadata['ZenodoId']
        self.assertTrue(isinstance(zenodo_id, int))
        zenodo_id_from_doi = re.match(DOI_PATTERN,
                                      self.metadata['Doi']).group(1)
        self.assertEqual(zenodo_id, int(zenodo_id_from_doi),
                         "Zenodo ID should match DOI")

    def test_description(self):
        self.assertIn('Description', self.metadata)
        self.assertTrue(isinstance(self.metadata['Description'], str))

    def test_keywords(self):
        self.assertIn('Keywords', self.metadata)
        keywords = self.metadata['Keywords']
        self.assertTrue(isinstance(keywords, list))
        self.assertTrue([isinstance(keyword, str) for keyword in keywords])

    def test_affiliations(self):
        self.assertIn('Affiliations', self.metadata)
        affiliations = self.metadata['Affiliations']
        self.assertTrue(isinstance(affiliations, list))
        self.assertTrue(
            [isinstance(affiliation, str) for affiliation in affiliations])
        # Check for duplicates
        simplified_affiliations = [
            simplify_name(affiliation) for affiliation in affiliations
        ]
        for affiliation in affiliations:
            self.assertEqual(
                simplified_affiliations.count(simplify_name(affiliation)), 1,
                f"Duplicate affiliation: {affiliation}")

    def test_authors(self):
        self.assertIn('Authors', self.metadata)
        authors = self.metadata['Authors']
        self.assertTrue(isinstance(authors['Description'], str))
        authors_core = authors['Core']
        self.assertTrue(isinstance(authors_core['Description'], str))
        authors_core_list = authors_core['List']
        self.assertTrue(isinstance(authors_core_list, list))
        authors_devs = authors['Developers']
        self.assertTrue(isinstance(authors_devs['Description'], str))
        authors_devs_list = authors_devs['List']
        self.assertTrue(isinstance(authors_devs_list, list))
        authors_contribs = authors['Contributors']
        self.assertTrue(isinstance(authors_contribs['Description'], str))
        authors_contribs_list = authors_contribs['List']
        self.assertTrue(isinstance(authors_contribs_list, list))
        all_authors = (authors_core_list + authors_devs_list +
                       authors_contribs_list)

        # Check alphabetical ordering
        self.assertEqual(
            sorted(authors_devs_list,
                   key=lambda a: a['Name']), authors_devs_list,
            ("'Developers' author list should be ordered alphabetically "
             "by last name"))
        self.assertEqual(
            sorted(authors_contribs_list,
                   key=lambda a: a['Name']), authors_contribs_list,
            ("'Contributors' author list should be ordered alphabetically "
             "by last name"))

        # Check all authors
        def check_name(name):
            split_name = name.split(', ')
            self.assertTrue(
                len(split_name) == 2 or len(split_name) == 3,
                (f"Name '{name}' should be formatted "
                 "'Last name[, Jr.], First name'"))
            if len(split_name) == 3:
                last_name, jr, first_name = split_name
                self.assertTrue(jr, f"Empty 'Jr.' component in name: {name}")
            else:
                last_name, first_name = split_name
            self.assertTrue(last_name, f"Missing last name: {name}")
            self.assertTrue(first_name, f"Missing first name: {name}")

        all_author_names_simplified = [
            simplify_name(author['Name']) for author in all_authors
        ]
        for author in all_authors:
            self.assertIn('Name', author)
            author_name = author['Name']
            check_name(author_name)
            self.assertEqual(
                all_author_names_simplified.count(simplify_name(author_name)),
                1, f"Duplicate author: {author_name}")
            if 'Orcid' in author:
                self.assertRegex(author['Orcid'], '^' + ORCID_PATTERN + '$',
                                 f"Invalid ORCID for {author_name}")
            self.assertIn('Affiliations', author)
            for affiliation in author['Affiliations']:
                self.assertIn(
                    affiliation, self.metadata['Affiliations'],
                    "Please use an existing affiliation or add a new one to "
                    "the main 'Affiliations' list in this file.")

        # Check for unused affiliations
        all_authors_affiliations = set(
            sum([author['Affiliations'] for author in all_authors], []))
        for affiliation in self.metadata['Affiliations']:
            self.assertIn(affiliation, all_authors_affiliations,
                          f"No author is affiliated with: {affiliation}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

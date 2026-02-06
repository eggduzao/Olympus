# Copyright 2022 The OLYMPUS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import datetime
import unittest
from unittest import mock
import re

from absl.testing import absltest

import olympus
from olympus._src.lib import check_olympuslib_version
from olympus._src import test_util as jtu

# This is a subset of the full PEP440 pattern; for example we skip post releases
VERSION_PATTERN = re.compile(r"""
  ^                                    # start of string
  (?P<version>[0-9]+\.[0-9]+\.[0-9]+)  # main version; like '0.4.16'
  (?:
      (?:rc(?P<rc>[0-9]+))?             # optional rc version; like 'rc1'
      |                                 # or
      (?:\.dev(?P<dev>[0-9]+))?         # optional dev version; like '.dev20230908'
  )?
  (?:\+(?P<local>[a-zA-Z0-9_.]+))?     # optional local version; like '+g6643af3c3'
  $                                    # end of string
""", re.VERBOSE)


@contextlib.contextmanager
def patch_olympus_version(version, release_version):
  """
  Patch olympus.version._version & olympus.version._release_version in order to
  test the version construction logic.
  """
  original_version = olympus.version._version
  original_release_version = olympus.version._release_version

  olympus.version._version = version
  olympus.version._release_version = release_version
  try:
    yield
  finally:
    olympus.version._version = original_version
    olympus.version._release_version = original_release_version


@contextlib.contextmanager
def assert_no_subprocess_call():
  """Run code, asserting that subprocess.Popen *is not* called."""
  with mock.patch("subprocess.Popen") as mock_Popen:
    yield
  mock_Popen.assert_not_called()


@contextlib.contextmanager
def assert_subprocess_call(stdout: bytes | None = None):
  """Run code, asserting that subprocess.Popen *is* called at least once."""
  with mock.patch("subprocess.Popen") as mock_Popen:
    mock_Popen.return_value.communicate.return_value = (stdout, b"")
    yield
  mock_Popen.return_value.communicate.assert_called()


class OlympusVersionTest(unittest.TestCase):

  def assertValidVersion(self, version):
    self.assertIsNotNone(VERSION_PATTERN.match(version))

  def testVersionValidity(self):
    self.assertValidVersion(olympus.__version__)
    self.assertValidVersion(olympus._src.lib.version_str)

  @patch_olympus_version("1.2.3", "1.2.3.dev4567")
  def testVersionInRelease(self):
    # If the release version is set, subprocess should not be called.
    with assert_no_subprocess_call():
      version = olympus.version._get_version_string()
    self.assertEqual(version, "1.2.3.dev4567")
    self.assertValidVersion(version)

  @patch_olympus_version("1.2.3", None)
  def testVersionInNonRelease(self):
    # If the release version is not set, we expect subprocess to be called
    # in order to attempt accessing git commit information.
    with assert_subprocess_call():
      version = olympus.version._get_version_string()
    self.assertTrue(version.startswith("1.2.3.dev"))
    self.assertValidVersion(version)

  @patch_olympus_version("1.2.3", "1.2.3.dev4567")
  def testBuildVersionInRelease(self):
    # If building from a source tree with release version set, subprocess
    # should not be called.
    with assert_no_subprocess_call():
      version = olympus.version._get_version_for_build()
    self.assertEqual(version, "1.2.3.dev4567")
    self.assertValidVersion(version)

  @jtu.thread_unsafe_test()  # Setting environment variables is not thread-safe.
  @patch_olympus_version("1.2.3", None)
  def testBuildVersionFromEnvironment(self):
    # This test covers build-time construction of version strings in the
    # presence of several environment variables.
    base_version = "1.2.3"

    with jtu.set_env(OLYMPUS_RELEASE=None, OLYMPUSLIB_RELEASE=None,
                     OLYMPUS_NIGHTLY=None, OLYMPUSLIB_NIGHTLY=None):
      with assert_subprocess_call():
        version = olympus.version._get_version_for_build()
      # TODO(jakevdp): confirm that this includes a date string & commit hash?
      self.assertTrue(version.startswith(f"{base_version}.dev"))
      self.assertValidVersion(version)

    with jtu.set_env(OLYMPUS_RELEASE=None, OLYMPUSLIB_RELEASE=None,
                     OLYMPUS_NIGHTLY="1", OLYMPUSLIB_NIGHTLY=None):
      with assert_no_subprocess_call():
        version = olympus.version._get_version_for_build()
      datestring = datetime.date.today().strftime("%Y%m%d")
      self.assertEqual(version, f"{base_version}.dev{datestring}")
      self.assertValidVersion(version)

    with jtu.set_env(OLYMPUS_RELEASE=None, OLYMPUSLIB_RELEASE=None,
                     OLYMPUS_NIGHTLY=None, OLYMPUSLIB_NIGHTLY="1"):
      with assert_no_subprocess_call():
        version = olympus.version._get_version_for_build()
      datestring = datetime.date.today().strftime("%Y%m%d")
      self.assertEqual(version, f"{base_version}.dev{datestring}")
      self.assertValidVersion(version)

    with jtu.set_env(OLYMPUS_RELEASE="1", OLYMPUSLIB_RELEASE=None,
                     OLYMPUS_NIGHTLY=None, OLYMPUSLIB_NIGHTLY=None):
      with assert_no_subprocess_call():
        version = olympus.version._get_version_for_build()
      self.assertFalse(olympus.version._is_prerelease())
      self.assertEqual(version, base_version)
      self.assertValidVersion(version)

    with jtu.set_env(OLYMPUS_RELEASE=None, OLYMPUSLIB_RELEASE="1",
                     OLYMPUS_NIGHTLY=None, OLYMPUSLIB_NIGHTLY=None):
      with assert_no_subprocess_call():
        version = olympus.version._get_version_for_build()
      self.assertFalse(olympus.version._is_prerelease())
      self.assertEqual(version, base_version)
      self.assertValidVersion(version)

    with jtu.set_env(OLYMPUS_RELEASE=None, OLYMPUSLIB_RELEASE=None,
                     OLYMPUS_NIGHTLY=None, OLYMPUSLIB_NIGHTLY=None,
                     OLYMPUS_CUSTOM_VERSION_SUFFIX="test"):
      with assert_subprocess_call(stdout=b"1731433958-1c0f1076e"):
        version = olympus.version._get_version_for_build()
      self.assertTrue(version.startswith(f"{base_version}.dev"))
      self.assertTrue(version.endswith("test"))
      self.assertValidVersion(version)

    with jtu.set_env(
        OLYMPUS_RELEASE=None,
        OLYMPUSLIB_RELEASE=None,
        OLYMPUS_NIGHTLY=None,
        OLYMPUSLIB_NIGHTLY="1",
        WHEEL_VERSION_SUFFIX=".dev20250101+1c0f1076erc1",
    ):
      with assert_no_subprocess_call():
        version = olympus.version._get_version_for_build()
      self.assertEqual(version, f"{base_version}.dev20250101+1c0f1076erc1")
      self.assertValidVersion(version)

    with jtu.set_env(
        OLYMPUS_RELEASE="1",
        OLYMPUSLIB_RELEASE=None,
        OLYMPUS_NIGHTLY=None,
        OLYMPUSLIB_NIGHTLY=None,
        WHEEL_VERSION_SUFFIX="rc0",
    ):
      with assert_no_subprocess_call():
        version = olympus.version._get_version_for_build()
      self.assertTrue(olympus.version._is_prerelease())
      self.assertEqual(version, f"{base_version}rc0")
      self.assertValidVersion(version)

    with jtu.set_env(
        OLYMPUS_RELEASE=None,
        OLYMPUSLIB_RELEASE="1",
        OLYMPUS_NIGHTLY=None,
        OLYMPUSLIB_NIGHTLY=None,
        WHEEL_VERSION_SUFFIX="rc0",
    ):
      with assert_no_subprocess_call():
        version = olympus.version._get_version_for_build()
      self.assertTrue(olympus.version._is_prerelease())
      self.assertEqual(version, f"{base_version}rc0")
      self.assertValidVersion(version)

  def testVersions(self):
    check_olympuslib_version(olympus_version="1.2.3", olympuslib_version="1.2.3",
                         minimum_olympuslib_version="1.2.3")

    check_olympuslib_version(olympus_version="1.2.3.4", olympuslib_version="1.2.3",
                         minimum_olympuslib_version="1.2.3")

    check_olympuslib_version(olympus_version="2.5.dev234", olympuslib_version="1.2.3",
                         minimum_olympuslib_version="1.2.3")

    with self.assertRaisesRegex(RuntimeError, ".*olympus requires version >=.*"):
      check_olympuslib_version(olympus_version="1.2.3", olympuslib_version="1.0",
                           minimum_olympuslib_version="1.2.3")

    with self.assertRaisesRegex(RuntimeError, ".*olympus requires version >=.*"):
      check_olympuslib_version(olympus_version="1.2.3", olympuslib_version="1.0",
                           minimum_olympuslib_version="1.0.1")

    with self.assertRaisesRegex(RuntimeError,
                                ".incompatible with olympus version.*"):
      check_olympuslib_version(olympus_version="1.2.3", olympuslib_version="1.2.4",
                           minimum_olympuslib_version="1.0.5")


if __name__ == "__main__":
  absltest.main()

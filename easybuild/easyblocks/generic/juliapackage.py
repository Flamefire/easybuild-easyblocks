##
# Copyright 2022-2026 Vrije Universiteit Brussel
#
# This file is part of EasyBuild,
# originally created by the HPC team of Ghent University (http://ugent.be/hpc/en),
# with support of Ghent University (http://ugent.be/hpc),
# the Flemish Supercomputer Centre (VSC) (https://www.vscentrum.be),
# Flemish Research Foundation (FWO) (http://www.fwo.be/en)
# and the Department of Economy, Science and Innovation (EWI) (http://www.ewi-vlaanderen.be/en).
#
# https://github.com/easybuilders/easybuild
#
# EasyBuild is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation v2.
#
# EasyBuild is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EasyBuild.  If not, see <http://www.gnu.org/licenses/>.
##
"""
EasyBuild support for Julia Packages, implemented as an easyblock

@author: Alex Domingo (Vrije Universiteit Brussel)
@author: Davide Grassano (CECAM, EPFL)
"""
import ast
import os
import re
import tempfile

from easybuild.tools import LooseVersion

import easybuild.tools.environment as env
from easybuild.framework.easyconfig import CUSTOM
from easybuild.framework.extensioneasyblock import ExtensionEasyBlock
from easybuild.tools.build_log import EasyBuildError
from easybuild.tools.modules import get_software_root, get_software_version
from easybuild.tools.filetools import mkdir, move_file
from easybuild.tools.run import run_shell_cmd
from easybuild.tools.utilities import trace_msg
from easybuild.tools import tomllib

_COMPILECACHE_CHECK = ' | '.join([
    "julia -E 'Base.compilecache_path(Base.identify_package(\"%(ext_name)s\"))'",
    "grep '%(ext_name)s'"
])

EXTS_FILTER_JULIA_PACKAGES = (
    " && ".join([
        _COMPILECACHE_CHECK,
        "julia -e 'using %(ext_name)s'",
    ]),
    ""
)
USER_DEPOT_PATTERN = re.compile(r"\/\.julia\/?(.*\.toml)*$")

# JULIA_PATHS_SOFT_INIT = {
#     "Lua": """
# if ( mode() == "load" ) then
#     if ( os.getenv("JULIA_DEPOT_PATH") == nil ) then setenv("JULIA_DEPOT_PATH", ":") end
#     if ( os.getenv("JULIA_LOAD_PATH") == nil ) then setenv("JULIA_LOAD_PATH", ":") end
# end
# """,
#     "Tcl": """
# if { [ module-info mode load ] } {
#     if {![info exists env(JULIA_DEPOT_PATH)]} { setenv JULIA_DEPOT_PATH : }
#     if {![info exists env(JULIA_LOAD_PATH)]} { setenv JULIA_LOAD_PATH : }
# """,
# }

JULIA_MODULE_FOOTER = {
    "Lua": """
setenv("JULIA_DEPOT_PATH", ":" .. os.getenv("EBJULIA_DEPOT_PATH") )
setenv("JULIA_LOAD_PATH", os.getenv("EBJULIA_LOAD_PATH") .. ":")
""",
    # This needs testing, the module generate seems to work fine, but the loading of the module within EB does not
    # properly resolve the $::env() and leaves it as a string
    "Tcl": """
setenv JULIA_DEPOT_PATH ":\$::env(EBJULIA_DEPOT_PATH)"
setenv JULIA_LOAD_PATH "\$::env(EBJULIA_LOAD_PATH):"
""",
}


class JuliaPackage(ExtensionEasyBlock):
    """
    Builds and installs Julia Packages.

    Julia environment setup during installation:
        - initialize new Julia environment in 'environments' subdir in installation directory
        - remove paths in user depot '~/.julia' from DEPOT_PATH and LOAD_PATH
        - put installation directory as top DEPOT_PATH, the target depot for installations with Pkg
        - put installation environment as top LOAD_PATH, needed to precompile installed packages
        - merge the environment of julia dependencies of this installation to the environment of this installation.
        - add newly installed Julia packages to installation environment
        - install all packages (main and deps) in parallel using Pkg.instantiate() command

    Julia environment setup on module load:
        User depot and its shared environment for this version of Julia are kept as top paths of DEPOT_PATH and
        LOAD_PATH respectively. This ensures that the user can keep using its own environment after loading
        JuliaPackage modules, installing additional software on its personal depot while still using packages
        provided by the module. Effectively, this translates to:
        - prepend installation directory to list of EB_JULIA_DEPOT_PATH, only really needed to load JLL artifacts
        - prepend installation Project.toml file to list of EB_JULIA_LOAD_PATH, needed to load packages with `using`
        - Generate JULIA_DEPOT_PATH by append the default DEPOT_PATH (:) to EB_JULIA_DEPOT_PATH
        - Generate JULIA_LOAD_PATH by prepending the default LOAD_PATH (:) to EB_JULIA_LOAD_PATH
          The prepending here is necessary as the default LOAD_PATH [@, @v#.#, @stdlib] can cause stdlib's packages
          that have been made upgradable (https://github.com/JuliaLang/julia/issues/50697) to fail matching the cache.
          See https://github.com/easybuilders/easybuild-easyblocks/issues/4123#issuecomment-4386990190 for more details.
    """

    @staticmethod
    def extra_options(extra_vars=None):
        """Extra easyconfig parameters specific to JuliaPackage."""
        extra_vars = ExtensionEasyBlock.extra_options(extra_vars=extra_vars)
        extra_vars.update({
            'download_pkg_deps': [
                False, "Let Julia download and bundle all needed dependencies for this installation", CUSTOM
            ],
            'is_test_dependency': [
                False,
                "Whether this package is only needed for testing and should not be added to installation environment",
                CUSTOM
            ],
            'julia_debug': [
                False,
                "Whether to set JULIA_DEBUG=all during installation and testing, to get more verbose output.",
                CUSTOM
            ],
            # Arguably this should be set to True by default. In many cases the test environment in Julia packages
            # will attempt to re-install the package being tested. If not all required packages to be  brought in the
            # test environment were specified in test/Project.toml, running Pkg.test() in offline mode will fail.
            # Some test tools like Aqua will also re-create their own tmp environment which can be broken even if
            # test/Project.toml is properly set up/patched.
            'test_online': [
                False,
                "Allow online tests that require downloading dependencies. By default, tests are run in offline mode.",
                CUSTOM
            ],
            'subpackages_dirs': [
                None, "List of subdirectories to look for additional packages to add to the environment",
                CUSTOM
            ],
        })
        return extra_vars

    @staticmethod
    def get_julia_env(env_var):
        """
        Query environment variable to julia shell and parse it
        :param env_var: string with name of environment variable
        """
        julia_read_cmd = {
            "DEPOT_PATH": "julia -E 'Base.DEPOT_PATH'",
            "LOAD_PATH": "julia -E 'Base.load_path()'",
        }

        try:
            res = run_shell_cmd(julia_read_cmd[env_var], hidden=True)
        except KeyError:
            raise EasyBuildError("Unknown Julia environment variable requested: %s", env_var)

        try:
            parsed_var = ast.literal_eval(res.output)
        except SyntaxError:
            raise EasyBuildError("Failed to parse %s from julia shell: %s", env_var, res.output)

        return parsed_var

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_toml = {'deps': {}, 'sources': {}}
        self._env_toml_test = {'deps': {}, 'sources': {}}
        self.julia_deps = []
        self._julia_version = None
        self._tmp_test_dir = None

    @property
    def julia_version(self) -> str:
        """Needs to get Julia version from dependencies to allow --sanity-check-only to generate the fake module"""
        if self._julia_version is None:
            deps = self.cfg.dependencies(runtime_only=True)
            for dep in deps:
                if dep['name'] == 'Julia':
                    self._julia_version = dep['version']
                    break
            else:
                raise EasyBuildError(
                    "Julia not included as dependency, cannot determine Julia version for installation of: %s",
                    self.name
                )
        return self._julia_version

    @property
    def env_toml(self):
        """Property to store content of installation environment Project.toml file."""
        if self.is_extension:
            return self.master.env_toml
        return self._env_toml

    @property
    def env_toml_test(self):
        """Property to store content of installation environment Project.toml file."""
        if self.is_extension:
            return self.master.env_toml_test
        return self._env_toml_test

    @property
    def is_last_extension(self):
        """Whether this extension is the last one to be installed in the installation."""
        if not self.is_extension:
            return False

        return self.master.ext_instances and self.master.ext_instances[-1] is self

    @property
    def tmp_test_dir(self):
        """Temporary path used for running the test step."""
        if self.is_extension:
            return self.master.tmp_test_dir

        if not self._tmp_test_dir:
            try:
                self._tmp_test_dir = tempfile.mkdtemp(suffix='-julia_tests')
            except Exception as e:
                raise EasyBuildError("Failed to create temporary directory for Julia package testing: %s", str(e))
        return self._tmp_test_dir

    @staticmethod
    def write_project_toml(file_path, project_toml):
        """Write a Project.toml file with the given content to the given path"""
        mkdir(os.path.dirname(file_path), parents=True)
        with open(file_path, 'w') as env_proj:
            for section, items in project_toml.items():
                env_proj.write(f'[{section}]\n')
                for key, value in items.items():
                    value = repr(value)
                    value = value.replace("'", '"')
                    value = value.replace(': ', ' = ')
                    env_proj.write(f'{key} = {value}\n')
                env_proj.write('\n')

    @staticmethod
    def read_project_toml(file_path):
        """Read a Project.toml file and return its content as a dict"""
        with open(file_path, 'rb') as env_proj:
            project_toml = tomllib.load(env_proj)
        return project_toml

    @classmethod
    def update_toml_data(cls, toml_data, pkg_path):
        """Update given toml data with new package and return updated data"""
        if os.path.isdir(pkg_path):
            pkg_dir = pkg_path
            pkg_path = os.path.join(pkg_path, 'Project.toml')
        else:
            pkg_dir = os.path.dirname(pkg_path)

        if not os.path.isfile(pkg_path):
            raise EasyBuildError("Project.toml file not found for package %s in path: %s", pkg_name, pkg_dir)

        project_toml = cls.read_project_toml(pkg_path)
        pkg_name = project_toml['name']
        pkg_uuid = project_toml['uuid']

        toml_data.setdefault('deps', {})[pkg_name] = pkg_uuid
        toml_data.setdefault('sources', {})[pkg_name] = {'path': pkg_dir}

        return toml_data

    def write_env_toml(self):
        """Write environment Project.toml file"""
        self.write_project_toml(self.julia_env_path(base=False), self.env_toml)

    def write_env_toml_test(self):
        """Write test environment Project.toml file"""
        self.write_project_toml(self.julia_env_path(base=False, basedir=self.tmp_test_dir), self.env_toml_test)

    def add_to_env_toml(self, package_dir):
        """Add package to environment toml file of the installation"""
        self.update_toml_data(self.env_toml, package_dir)

    def add_to_env_toml_test(self, package_dir):
        """Add package to environment toml file of the test environment"""
        self.update_toml_data(self.env_toml_test, package_dir)

    def julia_env_path(self, absolute=True, base=True, basedir=None):
        """
        Return path to installation environment file.
        """
        basedir = basedir or self.installdir
        julia_version = self.julia_version.split('.')
        env_dir = "v{}.{}".format(*julia_version[:2])
        project_env = os.path.join("environments", env_dir, "Project.toml")

        if absolute:
            project_env = os.path.join(basedir, project_env)
        if base:
            project_env = os.path.dirname(project_env)

        return project_env

    def set_pkg_offline(self):
        """Enable offline mode of Julia Pkg"""

        julia_version = get_software_version('Julia')
        if LooseVersion(julia_version) >= LooseVersion('1.5'):
            # Enable offline mode of Julia Pkg
            # https://pkgdocs.julialang.org/v1/api/#Pkg.offline
            env.setvar('JULIA_PKG_OFFLINE', 'true')
        else:
            errmsg = (
                "Cannot set offline mode in Julia v%s (needs Julia >= 1.5). "
                "Enable easyconfig option 'download_pkg_deps' to allow installation "
                "with any extra downloaded dependencies."
            )
            raise EasyBuildError(errmsg, julia_version)

    def prepare_julia_env(self, basedir=None, online=False):
        """
        1. Remove user depot and prepend installation directory to DEPOT_PATH.
        Top directory in Julia DEPOT_PATH is the target installation directory.
        See https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_DEPOT_PATH

        2. We also need the installation environment in LOAD_PATH to be able to populate it with all packages from
        current installation and its dependencies, as well as be able to precompile newly installed packages.
        This is automatically done by Julia once DEPOT_PATH is changed through JULIA_DEPOT_PATH. However, that
        only happens if JULIA_LOAD_PATH is not already set, which is the case for our modules of JuliaPackages.
        See https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_LOAD_PATH

        3. Enable offline mode in Julia to avoid automatic downloads of packages.

        4. Enable automatic precompilation of packages after each build.
        """
        basedir = basedir or self.installdir
        # Grab both DEPOT_PATH and LOAD_PATH before any changes are made
        # given that Julia might automatically update LOAD_PATH from a change on DEPOT_PATH
        dirty_depot = self.get_julia_env("DEPOT_PATH")
        self.log.debug('DEPOT_PATH read from Julia environment: %s', os.pathsep.join(dirty_depot))
        dirty_load = self.get_julia_env("LOAD_PATH")
        self.log.debug('LOAD_PATH read from Julia environment: %s', os.pathsep.join(dirty_load))

        # First set DEPOT_PATH and then LOAD_PATH to avoid any automatic changes made by Julia
        clean_depot = [path for path in dirty_depot if not USER_DEPOT_PATTERN.search(path) and path != self.installdir]
        install_depot = os.pathsep.join([basedir] + clean_depot)
        self.log.debug("Preparing Julia 'DEPOT_PATH' for installation: %s", install_depot)
        env.setvar("JULIA_DEPOT_PATH", install_depot)

        project_toml = self.julia_env_path(base=False, basedir=basedir)
        clean_load = [path for path in dirty_load if not USER_DEPOT_PATTERN.search(path) and path != project_toml]
        install_load = os.pathsep.join([project_toml] + clean_load)
        self.log.debug("Preparing Julia 'LOAD_PATH' for installation: %s", install_load)
        env.setvar("JULIA_LOAD_PATH", install_load)

        if project_toml not in self.get_julia_env("LOAD_PATH"):
            errmsg = "Failed to prepare Julia environment for installation of: %s"
            raise EasyBuildError(errmsg, self.name)

        # Enable offline mode
        if not online:
            self.set_pkg_offline()

        if self.cfg['julia_debug']:
            env.setvar('JULIA_DEBUG', 'all')

        # Set the maximum number of concurrent package builds
        env.setvar('JULIA_NUM_PRECOMPILE_TASKS', str(self.cfg.parallel))
        # Enable automatic precompilation
        env.setvar('JULIA_PKG_PRECOMPILE_AUTO', 'true')

    def install_pkg(self, basedir=None):
        """Execute Julia.Pkg command to install package from its sources"""
        basedir = basedir or self.installdir
        julia_pkg_cmd = [
            'using Pkg',
            'Pkg.activate("%s")' % self.julia_env_path(basedir=basedir),
            # Usage `Pkg.instantiate()` after all sources are in place to let Pkg handle all dependencies.
            # This has the advantage of letting Pkg deal with the order and parallel installation.
            'Pkg.instantiate()',
        ]

        julia_pkg_cmd = '; '.join(julia_pkg_cmd)
        cmd = ' '.join([
            self.cfg['preinstallopts'],
            "julia -e '%s'" % julia_pkg_cmd,
            self.cfg['installopts'],
        ])
        res = run_shell_cmd(cmd)

        return res.output

    def include_pkg_dependencies(self):
        """Add to installation environment all Julia packages already present in its dependencies"""
        sections = ['deps', 'sources']

        # add packages found in dependencies to this installation environment
        for dep in self.cfg.dependencies():
            dep_name = dep['name']
            dep_root = get_software_root(dep_name)
            dep_env = os.path.join(dep_root, self.julia_env_path(absolute=False, base=False))
            if not os.path.isfile(dep_env):
                self.log.warning("No environment file found in dependency %s, skipping: %s", dep_name, dep_env)
                continue

            self.julia_deps.append(dep_name)
            trace_msg(
                f"Incorporating Julia package dependencies from {dep_name} in installation environment: {dep_env}"
            )

            dep_toml = self.read_project_toml(dep_env)
            for section in sections:
                data = dep_toml.get(section, {})
                for toml in [self.env_toml, self.env_toml_test]:
                    toml.setdefault(section, {}).update(data)

    def prepare_step(self, *args, **kwargs):
        """Prepare for Julia package installation."""
        super().prepare_step(*args, **kwargs)

        if get_software_root('Julia') is None:
            raise EasyBuildError("Julia not included as dependency!")

    def configure_step(self):
        """No separate configuration for JuliaPackage."""
        pass

    def build_step(self):
        """No separate build procedure for JuliaPackage."""
        pass

    def test_step(self):
        """
        Test the built Julia package.

        :param return_output: return output and exit code of test command
        """
        testcmd = None
        if isinstance(self.cfg['runtest'], str):
            testcmd = self.cfg['runtest']

        if self.cfg['runtest']:
            if testcmd is None:
                testcmd = f"julia -e 'using Pkg; Pkg.test(\"{self.name}\")'"

            # Write test environment Project.toml that also include path reference to test dependencies
            self.write_env_toml_test()

            self.prepare_julia_env(basedir=self.tmp_test_dir, online=self.cfg['test_online'])

            cmd = ' '.join([
                self.cfg['pretestopts'],
                testcmd,
                self.cfg['testopts'],
            ])

            run_shell_cmd(cmd)

    def install_source(self):
        """Add the Julia package source files in the installation directory."""
        if self.cfg['is_test_dependency']:
            self.log.debug(
                f"Package {self.name} is only needed for testing, installing sources in {self.tmp_test_dir}"
            )
            trace_msg("Installing as a test dependency...")
            package_dir = os.path.join(self.tmp_test_dir, 'packages', self.name)
        else:
            package_dir = os.path.join(self.installdir, 'packages', self.name)
        # This also warks for dirs and will fail if the destination already exists
        move_file(self.ext_dir if self.is_extension else self.start_dir, package_dir)

        # Add all packages to the test environment Project.toml file
        self.add_to_env_toml_test(package_dir)
        if not self.cfg['is_test_dependency']:
            self.add_to_env_toml(package_dir)

        subpkg_dirs = self.cfg['subpackages_dirs'] or []
        for subpkg_dir in subpkg_dirs:
            subpkg_path = os.path.join(package_dir, subpkg_dir)
            if not os.path.isdir(subpkg_path):
                self.log.warning("Sub-package directory specified but not found, skipping: %s", subpkg_path)
                continue
            trace_msg(f"Adding sub-package from specified directory {subpkg_path}")
            self.add_to_env_toml(subpkg_path)

    def _install_step(self, basedir=None):
        """Install step commons between single-package and extensions based installs."""
        self.write_env_toml()
        self.prepare_julia_env(basedir=basedir, online=self.cfg['download_pkg_deps'])
        return self.install_pkg(basedir=basedir)

    def install_step(self):
        """Prepare installation environment and install Julia package."""
        self.install_source()
        return self._install_step()

    def install_extension(self):
        """Install Julia package as an extension."""
        if not self.src:
            errmsg = "No source found for Julia package %s, required for installation. (src: %s)"
            raise EasyBuildError(errmsg, self.name, self.src)

        # Unpack source into install directory and add package to the main environment Project.toml file
        ExtensionEasyBlock.install_extension(self, unpack_src=True)
        self.install_source()

        if self.is_last_extension:
            self._install_step()
            self.test_step()

    def sanity_check_step(self, *args, **kwargs):
        """Custom sanity check for JuliaPackage"""
        if self.cfg['is_test_dependency']:
            self.log.debug(f"Package {self.name} is only used for testing, skipping sanity check")
            return (True, "Test dependency, skipping sanity check")

        pkg_dir = os.path.join('packages', self.name)

        custom_commands = []
        # Check that the compile cache of the dependencies can still be loaded. The check is only w.r.t the package
        # name as rebuilding using PkgA and PkgB as dependenices can retrigger a compilation of PkgB in case e.g. it
        # was a `weakdep` of PkgA with an associated extension
        for julia_dep in self.julia_deps:
            custom_commands.append(
                _COMPILECACHE_CHECK % {'ext_name': julia_dep}
            )
        kwargs.setdefault('custom_commands', []).extend(custom_commands)

        custom_paths = {
            'files': [],
            'dirs': [pkg_dir],
        }
        kwargs.update({'custom_paths': custom_paths})

        return ExtensionEasyBlock.sanity_check_step(self, EXTS_FILTER_JULIA_PACKAGES, *args, **kwargs)

    def make_module_extra(self, *args, **kwargs):
        """
        Module load initializes JULIA_DEPOT_PATH and JULIA_LOAD_PATH with default values if they are not set.

        Path to installation directory is appended to JULIA_DEPOT_PATH.
        Path to the environment file of this installation is prepended to JULIA_LOAD_PATH.
        This configuration fulfils the rule that user depot has to be the first path in JULIA_DEPOT_PATH,
        allowing user to add custom Julia packages while having packages in this installation available.
        See issue easybuilders/easybuild-easyconfigs#17455
        """
        mod = super().make_module_extra()
        # if self.module_generator.SYNTAX:
        #     mod += JULIA_PATHS_SOFT_INIT[self.module_generator.SYNTAX]
        # mod += self.module_generator.append_paths('JULIA_DEPOT_PATH', [''])

        mod += self.module_generator.prepend_paths('EBJULIA_DEPOT_PATH', [''])
        mod += self.module_generator.prepend_paths('EBJULIA_LOAD_PATH', [self.julia_env_path(absolute=False, base=False)])

        return mod

    def make_module_footer(self):
        """
        Extend module footer with statements to set up shell completion for Click-based Python tools.
        """
        footer = super().make_module_footer()

        if self.module_generator.SYNTAX:
            extra_footer = JULIA_MODULE_FOOTER[self.module_generator.SYNTAX]
            footer += '\n' + extra_footer + '\n'

        return footer

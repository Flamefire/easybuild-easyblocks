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
from easybuild.tools.filetools import move_file
from easybuild.tools.run import run_shell_cmd
from easybuild.tools.utilities import trace_msg

from easybuild.easyblocks.j.julia import EB_JULIA_DEPOT_PATH_VAR, EB_JULIA_LOAD_PATH_VAR

_COMPILECACHE_CHECK = ' | '.join([
    # "julia -E 'Base.compilecache_path(Base.identify_package(\"%(ext_name)s\"), Base.get_world_counter())'",
    # compilecache_path is not part of the stable API and changes arguments across versions
    # allow internal cache resolution and use debug statemnts to get the path of the loaded cache
    "JULIA_DEBUG=loading julia -e 'using %(ext_name)s' 2>&1 1>/dev/null",
    "grep 'Loading object cache file .*%(grep_loc)s'"
])

EXTS_FILTER_JULIA_PACKAGES = (
    " && ".join([
        _COMPILECACHE_CHECK.replace('%(grep_loc)s', '%(ext_name)s'),
        "julia -e 'using %(ext_name)s'",
    ]),
    ""
)
USER_DEPOT_PATTERN = re.compile(r"\/\.julia\/?(.*\.toml)*$")


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

        if env_var not in julia_read_cmd:
            raise EasyBuildError("Unknown Julia environment variable requested: %s", env_var)

        cmd = julia_read_cmd[env_var]

        res = run_shell_cmd(cmd, hidden=True)

        try:
            parsed_var = ast.literal_eval(res.output)
        except SyntaxError:
            raise EasyBuildError("Failed to parse %s from julia shell: %s", env_var, res.output)

        return parsed_var

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conflicts = []
        self._julia_deps = {}
        self._julia_deps_test = {}
        self._julia_version = None
        self._pkg_deps = []
        self._pkg_to_test = []
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
    def conflicts(self):
        """List of conflicting software."""
        if self.is_extension:
            return self.master.conflicts
        return self._conflicts

    @property
    def is_last_extension(self):
        """Whether this extension is the last one to be installed in the installation."""
        if not self.is_extension:
            return False

        return self.master.ext_instances and self.master.ext_instances[-1] is self

    @property
    def julia_deps(self):
        """List of Julia dependencies found in this installation."""
        if self.is_extension:
            return self.master.julia_deps
        return self._julia_deps

    @property
    def julia_deps_test(self):
        """List of Julia dependencies found in this installation."""
        if self.is_extension:
            return self.master.julia_deps_test
        return self._julia_deps_test

    @property
    def pkg_deps(self):
        """List of Julia dependencies found in this installation."""
        if self.is_extension:
            return self.master.pkg_deps
        return self._pkg_deps

    @property
    def pkg_to_test(self):
        """List of Julia dependencies to be included in the test environment."""
        if self.is_extension:
            return self.master.pkg_to_test
        return self._pkg_to_test

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

    def set_pkg_remote(self, online=False):
        """Enable online/offline mode of Julia Pkg"""

        julia_version = get_software_version('Julia')
        if LooseVersion(julia_version) >= LooseVersion('1.5'):
            # Enable offline mode of Julia Pkg
            # https://pkgdocs.julialang.org/v1/api/#Pkg.offline
            env.setvar('JULIA_PKG_OFFLINE', 'false' if online else 'true')
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

        # Disable EB controlled PATHS to enforce the specific configurations
        env.setvar(EB_JULIA_DEPOT_PATH_VAR, "")
        env.setvar(EB_JULIA_LOAD_PATH_VAR, "")

        if project_toml not in self.get_julia_env("LOAD_PATH"):
            errmsg = "Failed to prepare Julia environment for installation of: %s"
            raise EasyBuildError(errmsg, self.name)

        # Enable/disable offline mode
        self.set_pkg_remote(online=online)

        if self.cfg['julia_debug']:
            env.setvar('JULIA_DEBUG', 'all')

        # Set the maximum number of concurrent package builds
        env.setvar('JULIA_NUM_PRECOMPILE_TASKS', str(self.cfg.parallel))
        # Enable automatic precompilation
        env.setvar('JULIA_PKG_PRECOMPILE_AUTO', 'true')

    def add_package(self, pkg_source, test_only=False):
        pkg_name = os.path.basename(pkg_source)
        if pkg_name in self.julia_deps:
            prev_source = self.julia_deps[pkg_name]
            if prev_source != pkg_source:
                self.conflicts.append((self.name, pkg_name, prev_source, pkg_source))

        self.julia_deps_test[pkg_name] = pkg_source
        if not test_only:
            self.julia_deps[pkg_name] = pkg_source

    def _check_conflicts(self):
        """Check for conflicts in Julia dependencies and raise an error if any are found."""
        if self.conflicts:
            conflict_msgs = []
            for ext_name, pkg_name, prev_source, new_source in self.conflicts:
                conflict_msgs.append(
                    f"Conflict detected for package '{pkg_name}' in extension '{ext_name}': "
                    f"already added from source '{prev_source}', cannot add from source '{new_source}'"
                )
            raise EasyBuildError("Conflicts detected in Julia dependencies:\n" + "\n".join(conflict_msgs))

    def install_pkg(self, basedir=None):
        """Execute Julia.Pkg command to install package from its sources"""
        self._check_conflicts()
        basedir = basedir or self.installdir
        env = self.julia_env_path(basedir=basedir)

        package_specs = ', '.join(f'PackageSpec(path="{path}")' for path in self.julia_deps.values())

        julia_pkg_cmd = [
            'using Pkg',
            f'Pkg.activate("{env}")',

            f'Pkg.develop([{package_specs}]; preserve=PRESERVE_ALL)',
            # Usage `Pkg.instantiate()` after all sources are in place to let Pkg handle all dependencies.
            # This has the advantage of letting Pkg deal with the order and parallel installation.
            'Pkg.instantiate()',
            # Ensure operations done only by build.jl are done for all packages if needed
            'Pkg.build()',
        ]

        julia_pkg_cmd = '; '.join(julia_pkg_cmd)
        cmd = ' '.join([
            self.cfg['preinstallopts'],
            "julia -e '%s'" % julia_pkg_cmd,
            self.cfg['installopts'],
        ])
        res = run_shell_cmd(cmd)

        return res.output

    def _deps_from_project(self, environment):
        """Get list of dependencies from a Project.toml file"""
        julia_root = get_software_root('Julia')
        cmd = "; ".join([
            'using Pkg',
            f'Pkg.activate("{environment}")',
            # 'Pkg.status(; mode=PKGMODE_MANIFEST)',
            'for (_, pkg) in Pkg.dependencies()',
            'println("$(pkg.source)")',
            'end'
        ])

        res = run_shell_cmd(f"julia -E '{cmd}'", split_stderr=True, hidden=True)
        deps = res.output.strip().splitlines()

        # Filter out Julia's stdlib dependencies
        deps = filter(lambda d: not d.startswith(julia_root), deps)
        deps = filter(lambda d: d != 'nothing', deps)

        return list(deps)

    def include_pkg_dependencies(self):
        """Add to installation environment all Julia packages already present in its dependencies"""
        build_only_deps = self.cfg.dependencies(build_only=True)
        # add packages found in dependencies to this installation environment
        for dep in self.cfg.dependencies():
            test_only = dep in build_only_deps
            dep_name = dep['name']
            dep_root = get_software_root(dep_name)
            dep_env = self.julia_env_path(absolute=True, base=True, basedir=dep_root)
            dep_manifest = os.path.join(dep_env, 'Manifest.toml')
            if not os.path.isfile(dep_manifest):
                self.log.warning("No Manifest.toml file found in dependency %s, skipping: %s", dep_name, dep_env)
                continue

            if test_only:
                trace_msg(
                    f"Incorporating Julia package in TEST    environment {dep_name}: {dep_env}"
                )
            else:
                self.pkg_deps.append((dep_name, dep_root))
                trace_msg(
                    f"Incorporating Julia package in INSTALL environment {dep_name}: {dep_env}"
                )

            for pkg_path in self._deps_from_project(dep_env):
                self.add_package(pkg_path, test_only=dep in build_only_deps)

    def prepare_step(self, *args, **kwargs):
        """Prepare for Julia package installation."""
        super().prepare_step(*args, **kwargs)

        if get_software_root('Julia') is None:
            raise EasyBuildError("Julia not included as dependency!")

        self.include_pkg_dependencies()

    def test_step(self):
        """
        Test the built Julia package.

        :param return_output: return output and exit code of test command
        """
        if self.pkg_to_test:
            env = self.julia_env_path(basedir=self.tmp_test_dir)
            pkg_specs = ', '.join(f'PackageSpec(path="{path}")' for path in self.julia_deps_test.values())
            pkg_test = ', '.join(f'"{pkg}"' for pkg in self.pkg_to_test)
            testcmd = [
                'using Pkg',
                f'Pkg.activate("{env}")',
                f'Pkg.develop([{pkg_specs}]; preserve=PRESERVE_ALL)',
                f'Pkg.test([{pkg_test}])',
            ]
            testcmd = '; '.join(testcmd)
            testcmd = f"julia -e '{testcmd}'"

            # Also add normal DEPOT/LOAD paths to environment to try and re-use pre-compiled caches as much as possible
            # But also ensure that artifacts for packages that do not need recompilation are still found in the tests
            self.prepare_julia_env(online=self.cfg['test_online'])
            self.prepare_julia_env(basedir=self.tmp_test_dir, online=self.cfg['test_online'])

            cmd = ' '.join([
                self.cfg['pretestopts'],
                testcmd,
                self.cfg['testopts'],
            ])

            run_shell_cmd(cmd)

    def install_source(self):
        """Add the Julia package source files in the installation directory."""
        package_dir = os.path.join(self.installdir, 'packages', self.name)
        move_file(self.start_dir, package_dir)
        self.add_package(package_dir)

    def _install_step(self, basedir=None):
        """Install step commons between single-package and extensions based installs."""
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

        if self.cfg['runtest']:
            self.pkg_to_test.append(self.name)

        if self.is_last_extension:
            self._install_step()
            self.test_step()

    def sanity_check_step(self, *args, **kwargs):
        """Custom sanity check for JuliaPackage"""
        pkg_dir = os.path.join('packages', self.name)

        custom_commands = []

        # Check that the compile cache of the dependencies can still be loaded and is coming from the expected package
        if not self.is_extension:
            for dep, root in self.pkg_deps:
                # root_comp = os.path.join(root, 'compiled')
                custom_commands.append(
                    # _COMPILECACHE_CHECK % {'ext_name': dep, 'grep_loc': f'Loading object cache file .*{root_comp}'}
                    _COMPILECACHE_CHECK % {'ext_name': dep, 'grep_loc': dep}

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

        mod += self.module_generator.prepend_paths(EB_JULIA_DEPOT_PATH_VAR, [''])
        mod += self.module_generator.prepend_paths(
            EB_JULIA_LOAD_PATH_VAR,
            [self.julia_env_path(absolute=False, base=True)]
        )

        return mod

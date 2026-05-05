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
from easybuild.tools.filetools import copy_dir, mkdir, change_dir, move_file
from easybuild.tools.run import run_shell_cmd
from easybuild.tools.utilities import trace_msg
from easybuild.tools import tomllib

EXTS_FILTER_JULIA_PACKAGES = ("julia -e 'using %(ext_name)s'", "")
USER_DEPOT_PATTERN = re.compile(r"\/\.julia\/?(.*\.toml)*$")

JULIA_PATHS_SOFT_INIT = {
    "Lua": """
if ( mode() == "load" ) then
    if ( os.getenv("JULIA_DEPOT_PATH") == nil ) then setenv("JULIA_DEPOT_PATH", ":") end
    if ( os.getenv("JULIA_LOAD_PATH") == nil ) then setenv("JULIA_LOAD_PATH", ":") end
end
""",
    "Tcl": """
if { [ module-info mode load ] } {
    if {![info exists env(JULIA_DEPOT_PATH)]} { setenv JULIA_DEPOT_PATH : }
    if {![info exists env(JULIA_LOAD_PATH)]} { setenv JULIA_LOAD_PATH : }
}
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
        - add Julia packages found in dependencies of the easyconfig to installation environment, needed
          for Pkg to be aware of those packages and not install them again
        - add newly installed Julia packages to installation environment (automatically done by Pkg)

    Julia environment setup on module load:
        User depot and its shared environment for this version of Julia are kept as top paths of DEPOT_PATH and
        LOAD_PATH respectively. This ensures that the user can keep using its own environment after loading
        JuliaPackage modules, installing additional software on its personal depot while still using packages
        provided by the module. Effectively, this translates to:
        - append installation directory to list of DEPOT_PATH, only really needed to load artifacts (JLL packages)
        - append installation Project.toml file to list of LOAD_PATH, needed to load packages with `using` command
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
            'is_main_package': [
                False,
                "Whether this package is the main package of the installation. Only this will be installed and Julia "
                "will take care of installing dependencies.",
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

    @property
    def env_toml(self):
        """Property to store content of installation environment Project.toml file."""
        if self.is_extension:
            return self.master.env_toml
        return self._env_toml

    def julia_env_path(self, absolute=True, base=True):
        """
        Return path to installation environment file.
        """
        julia_version = get_software_version('Julia').split('.')
        env_dir = "v{}.{}".format(*julia_version[:2])
        project_env = os.path.join("environments", env_dir, "Project.toml")

        if absolute:
            project_env = os.path.join(self.installdir, project_env)
        if base:
            project_env = os.path.dirname(project_env)

        return project_env

    def set_pkg_offline(self):
        """Enable offline mode of Julia Pkg"""

        if not self.cfg['download_pkg_deps']:
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

    def prepare_julia_env(self):
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
        # Grab both DEPOT_PATH and LOAD_PATH before any changes are made
        # given that Julia might automatically update LOAD_PATH from a change on DEPOT_PATH
        dirty_depot = self.get_julia_env("DEPOT_PATH")
        self.log.debug('DEPOT_PATH read from Julia environment: %s', os.pathsep.join(dirty_depot))
        dirty_load = self.get_julia_env("LOAD_PATH")
        self.log.debug('LOAD_PATH read from Julia environment: %s', os.pathsep.join(dirty_load))

        # First set DEPOT_PATH and then LOAD_PATH to avoid any automatic changes made by Julia
        clean_depot = [path for path in dirty_depot if not USER_DEPOT_PATTERN.search(path) and path != self.installdir]
        install_depot = os.pathsep.join([self.installdir] + clean_depot)
        self.log.debug("Preparing Julia 'DEPOT_PATH' for installation: %s", install_depot)
        env.setvar("JULIA_DEPOT_PATH", install_depot)

        project_toml = self.julia_env_path(base=False)
        clean_load = [path for path in dirty_load if not USER_DEPOT_PATTERN.search(path) and path != project_toml]
        install_load = os.pathsep.join([project_toml] + clean_load)
        self.log.debug("Preparing Julia 'LOAD_PATH' for installation: %s", install_load)
        env.setvar("JULIA_LOAD_PATH", install_load)

        if self.julia_env_path(base=False) not in self.get_julia_env("LOAD_PATH"):
            errmsg = "Failed to prepare Julia environment for installation of: %s"
            raise EasyBuildError(errmsg, self.name)

        # Enable offline mode
        self.set_pkg_offline()

        # Enable automatic precompilation
        env.setvar('JULIA_PKG_PRECOMPILE_AUTO', 'true')

    def install_pkg_source(self, pkg_source, environment, trace=True):
        """Execute Julia.Pkg command to install package from its sources"""

        julia_pkg_cmd = [
            'using Pkg',
            'Pkg.activate("%s")' % environment,
        ]

        if os.path.isdir(os.path.join(pkg_source, '.git')):
            # sources from git repos can be installed as any remote package
            self.log.debug('Installing Julia package in normal mode (Pkg.add)')

            julia_pkg_cmd.extend([
                # install package from local path preserving existing dependencies
                'Pkg.add(url="%s"; preserve=PRESERVE_ALL)' % pkg_source,
            ])
        else:
            # plain sources have to be installed in develop mode
            self.log.debug('Installing Julia package in develop mode (Pkg.develop)')

            julia_pkg_cmd.extend([
                # install package from local path preserving existing dependencies
                'Pkg.develop(PackageSpec(path="%s"); preserve=PRESERVE_ALL)' % pkg_source,
                'Pkg.build("%s")' % os.path.basename(pkg_source),
            ])

        julia_pkg_cmd = '; '.join(julia_pkg_cmd)
        cmd = ' '.join([
            self.cfg['preinstallopts'],
            "julia -e '%s'" % julia_pkg_cmd,
            self.cfg['installopts'],
        ])
        res = run_shell_cmd(cmd)

        return res.output

    @staticmethod
    def write_project_toml(file_path, project_toml):
        """Write a Project.toml file with the given content to the given path"""

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

    def write_env_toml(self, toml_data):
        """Write environment Project.toml file"""

        self.write_project_toml(self.julia_env_path(base=False), toml_data)


    # def include_pkg_dependencies(self):
    #     """Add to installation environment all Julia packages already present in its dependencies"""
    #     # Location of project environment files in install dir
    #     mkdir(self.julia_env_path(), parents=True)

    #     # add packages found in dependencies to this installation environment
    #     for dep in self.cfg.dependencies():
    #         dep_root = get_software_root(dep['name'])
    #         for pkg in glob.glob(os.path.join(dep_root, 'packages/*')):
    #             trace_msg("incorporating Julia package from dependencies: %s" % os.path.basename(pkg))
    #             self.install_pkg_source(pkg, self.julia_env_path(), trace=False)

    def include_pkg_dependencies(self):
        """Add to installation environment all Julia packages already present in its dependencies"""
        # Location of project environment files in install dir
        mkdir(self.julia_env_path(), parents=True)

        sections = ['deps', 'sources']

        # add packages found in dependencies to this installation environment
        to_remove = set()
        for dep in self.cfg.dependencies():
            dep_name = dep['name']
            dep_root = get_software_root(dep_name)
            dep_env = os.path.join(dep_root, self.julia_env_path(absolute=False, base=False))
            if not os.path.isfile(dep_env):
                self.log.warning("No environment file found in dependency %s, skipping: %s", dep_name, dep_env)
                continue

            to_remove.add(dep_name)
            trace_msg

            dep_toml = self.read_project_toml(dep_env)
            for section in sections:
                self.env_toml.setdefault(section, {}).update(dep_toml.get(section, {}))

        if to_remove:
            trace_msg(
                "Removing dependencies from installation environment. "
                "These dependencies will not appear in the resulting module:"
            )
        for dep_type in ['dependencies', 'builddependencies']:
            to_remove_idx = []
            ref = self.cfg.get_ref(dep_type)
            for idx, dep in enumerate(ref):
                if dep['name'] in to_remove:
                    trace_msg(f'- {dep}')
                    to_remove_idx.insert(0, idx)
            for idx in to_remove_idx:
                del ref[idx]

        # print("Updated dependencies after removing Julia packages: %s", self.cfg.dependencies())
        # print("Updated dependencies after removing Julia packages: %s", self.cfg.dependencies())

        self.write_env_toml(self.env_toml)


    def install_pkg(self):
        """Install Julia package"""

        # determine source type of current installation
        if os.path.isdir(os.path.join(self.start_dir, '.git')):
            pkg_source = self.start_dir
        else:
            # copy non-git sources to install directory
            pkg_source = os.path.join(self.installdir, 'packages', self.name)
            copy_dir(self.start_dir, pkg_source)

        return self.install_pkg_source(pkg_source, self.julia_env_path())

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

            try:
                tmpdir = tempfile.mkdtemp()
            except OSError as err:
                raise EasyBuildError("Failed to create test install dir: %s", err)

            depot_path = self.get_julia_env("DEPOT_PATH")
            load_path = self.get_julia_env("LOAD_PATH")
            self.log.info("Original DEPOT_PATH for testing: %s", os.pathsep.join(depot_path))
            self.log.info("Original LOAD_PATH for testing: %s", os.pathsep.join(load_path))
            # print("Testing with DEPOT_PATH: %s", os.pathsep.join(depot_path))
            # print("Testing with LOAD_PATH: %s", os.pathsep.join(load_path))

            depot_path = os.pathsep.join([tmpdir] + depot_path)

            extrapath = " && ".join([
                f"export JULIA_DEPOT_PATH=\"{depot_path}\"",
                # Ensure TEST dependencies can be downloaded
                # The alternative is to install them as normal dependencies of the package
                # "export JULIA_PKG_OFFLINE=false",
                ""
            ])

            cmd = ' '.join([
                extrapath,
                self.cfg['pretestopts'],
                testcmd,
                self.cfg['testopts'],
            ])

            run_shell_cmd(cmd)

    def install_step(self):
        """Prepare installation environment and install Julia package."""

        self.prepare_julia_env()
        self.include_pkg_dependencies()

        return self.install_pkg()

    def install_extension(self):
        """Install Julia package as an extension."""

        if not self.src:
            errmsg = "No source found for Julia package %s, required for installation. (src: %s)"
            raise EasyBuildError(errmsg, self.name, self.src)

        if self.cfg['is_test_dependency']:
            self.log.info("Package %s is only needed for testing, skipping installation", self.name)
            return

        ExtensionEasyBlock.install_extension(self, unpack_src=True)

        if not self.cfg['is_main_package']:
            self.log.info("Package %s is not the main package of the installation, skipping installation", self.name)

            change_dir(self.master.start_dir)
            package_dir = os.path.join(self.installdir, 'packages', self.name)
            move_file(self.ext_dir, package_dir)

            self.add_to_env_toml()
        else:
            self.write_project_toml(self.julia_env_path(base=False), self.env_toml)

            self.prepare_julia_env()
            self.install_pkg()
            self._test_step()

    @classmethod
    def update_toml_data(cls, toml_data, pkg_name, pkg_path):
        """Update given toml data with new package and return updated data"""
        if os.path.isdir(pkg_path):
            pkg_dir = pkg_path
            pkg_path = os.path.join(pkg_path, 'Project.toml')
        else:
            pkg_dir = os.path.dirname(pkg_path)

        if not os.path.isfile(pkg_path):
            raise EasyBuildError("Project.toml file not found for package %s in path: %s", pkg_name, pkg_dir)

        project_toml = cls.read_project_toml(pkg_path)
        pkg_uuid = project_toml['uuid']

        toml_data.setdefault('deps', {})[pkg_name] = pkg_uuid
        toml_data.setdefault('sources', {})[pkg_name] = {'path': pkg_dir}

        return toml_data

    def add_to_env_toml(self):
        """Add package to environment toml file of the installation"""
        package_dir = os.path.join(self.installdir, 'packages', self.name)
        self.update_toml_data(self.env_toml, self.name, package_dir)

    def sanity_check_step(self, *args, **kwargs):
        """Custom sanity check for JuliaPackage"""

        pkg_dir = os.path.join('packages', self.name)

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
        if self.module_generator.SYNTAX:
            mod += JULIA_PATHS_SOFT_INIT[self.module_generator.SYNTAX]
        mod += self.module_generator.append_paths('JULIA_DEPOT_PATH', [''])
        mod += self.module_generator.append_paths('JULIA_LOAD_PATH', [self.julia_env_path(absolute=False, base=False)])

        return mod

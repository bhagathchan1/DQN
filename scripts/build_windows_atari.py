"""Build the real multi-agent Atari runtime on Windows without Visual Studio.

Usage (inside this project's virtual environment):
    python scripts/build_windows_atari.py

Downloads hash-checked official source/toolchain archives into ignored tmp/native,
builds a statically linked C DLL and installs a locally generated wheel. No system
compiler, PATH setting, registry setting or ROM installation is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = {
    'w64devkit.7z.exe': (
        'https://github.com/skeeto/w64devkit/releases/download/v2.11.0/w64devkit-x64-2.11.0.7z.exe',
        '54c2bb9962e8a30c8270e34c5efc3d5fee579f53087fa7d435b95fbba2af7b2b'),
    'zlib.tar.gz': (
        'https://github.com/madler/zlib/releases/download/v1.4/zlib-1.4.tar.gz',
        '02dd159783677a749c04adbdd02b55b8a2bcc7e054f51b76b549199a916ad584'),
    'multi_agent_ale_py-0.1.12.tar.gz': (
        'https://files.pythonhosted.org/packages/4e/29/db08a20df0fa91bf05ad62060da7a77f650527f7cfe08afc9b299bda7349/multi_agent_ale_py-0.1.12.tar.gz',
        '4198102669333788cad3c749f208dd941d52b3a39e5c6e16caeec10b0c9c53aa'),
}


def run(command, *, cwd=ROOT, env=None):
    print('Running:', ' '.join(str(arg) for arg in command), flush=True)
    subprocess.run([str(arg) for arg in command], cwd=cwd, env=env, check=True)


def download(target: Path, url: str, expected: str):
    if not target.exists():
        print(f'Downloading {target.name}', flush=True)
        temporary = target.with_suffix(target.suffix + '.download')
        request = urllib.request.Request(url, headers={'User-Agent': 'Atari-Pong-Lab-Setup/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f'{target.name}: SHA-256 mismatch. Remove the damaged archive and retry.')


def extract(archive: Path, destination: Path, expected_directory: str):
    if (destination / expected_directory / 'CMakeLists.txt').exists():
        return
    with tarfile.open(archive) as source:
        for item in source.getmembers():
            target = (destination / item.name).resolve()
            if not target.is_relative_to(destination.resolve()) or item.issym() or item.islnk():
                raise RuntimeError(f'Unexpected archive entry: {item.name}')
        source.extractall(destination, filter='data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=min(6, os.cpu_count() or 2))
    parser.add_argument('--no-install', action='store_true', help='Build the wheel without installing it')
    args = parser.parse_args()
    if sys.platform != 'win32' or platform.machine().lower() not in {'amd64', 'x86_64'}:
        parser.error('This helper is for x64 Windows; other platforms can install pettingzoo[atari] normally.')
    if sys.version_info < (3, 12):
        parser.error('This helper requires Python 3.12+ for safe tar extraction; use the project Python 3.12 environment.')
    if args.jobs <= 0:
        parser.error('--jobs must be positive')
    run([sys.executable, '-m', 'pip', 'install', 'cmake==4.4.4', 'ninja==1.13.2', 'wheel', 'setuptools>=68'])
    scripts = Path(sys.executable).parent
    cmake = scripts / 'cmake.exe'
    ninja = scripts / 'ninja.exe'
    if not cmake.exists() or not ninja.exists():
        raise RuntimeError('Run this helper with your virtual environment Python executable.')
    work = ROOT / 'tmp' / 'native'
    work.mkdir(parents=True, exist_ok=True)
    for filename, (url, digest) in DOWNLOADS.items():
        download(work / filename, url, digest)
    compiler_root = work / 'compiler'
    compiler_bin = compiler_root / 'w64devkit' / 'bin'
    if not (compiler_bin / 'g++.exe').exists():
        compiler_root.mkdir(exist_ok=True)
        # Windows bsdtar can read this official self-extracting 7z archive without
        # launching its GUI executable or modifying system compiler settings.
        run(['tar', '-xf', work / 'w64devkit.7z.exe', '-C', compiler_root])
    extract(work / 'zlib.tar.gz', work, 'zlib-1.4')
    extract(work / 'multi_agent_ale_py-0.1.12.tar.gz', work, 'multi_agent_ale_py-0.1.12')
    environment = os.environ.copy()
    environment['PATH'] = str(compiler_bin) + os.pathsep + environment.get('PATH', '')
    environment.pop('GITHUB_REF', None)
    zlib_install = work / 'zlib-install'
    run([cmake, '-S', work / 'zlib-1.4', '-B', work / 'zlib-build', '-G', 'Ninja',
         f'-DCMAKE_MAKE_PROGRAM={ninja.as_posix()}',
         f'-DCMAKE_C_COMPILER={(compiler_bin / "gcc.exe").as_posix()}',
         f'-DCMAKE_INSTALL_PREFIX={zlib_install.as_posix()}', '-DCMAKE_BUILD_TYPE=Release',
         '-DZLIB_BUILD_SHARED=OFF', '-DZLIB_BUILD_TESTING=OFF'], env=environment)
    run([cmake, '--build', work / 'zlib-build', '--parallel', args.jobs], env=environment)
    run([cmake, '--install', work / 'zlib-build'], env=environment)
    zlib_static = zlib_install / 'lib' / 'libzs.a'
    if not zlib_static.exists():
        raise RuntimeError('The pinned zlib build did not produce libzs.a')
    run([cmake, '-S', work / 'multi_agent_ale_py-0.1.12', '-B', work / 'ale-build', '-G', 'Ninja',
         f'-DCMAKE_MAKE_PROGRAM={ninja.as_posix()}',
         f'-DCMAKE_CXX_COMPILER={(compiler_bin / "g++.exe").as_posix()}',
         f'-DZLIB_LIBRARY={zlib_static.as_posix()}', f'-DZLIB_INCLUDE_DIR={(zlib_install / "include").as_posix()}',
         '-DCMAKE_SHARED_LINKER_FLAGS=-static-libgcc -static-libstdc++ -static',
         '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_STANDARD=11', '-DUSE_SDL=OFF',
         '-DBUILD_CPP_LIB=OFF', '-DBUILD_C_LIB=ON', '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON'], env=environment)
    run([cmake, '--build', work / 'ale-build', '--parallel', args.jobs], env=environment)
    dll = work / 'ale-build' / 'src' / 'libale_c.dll'
    if not dll.exists():
        raise RuntimeError('ALE C DLL was not produced')

    # Package the unchanged upstream Python ctypes wrapper with the built DLL.
    # Calling upstream setup.py directly unconditionally selects an MSVC platform;
    # this small wheel build avoids that build-system assumption. No emulator
    # behavior or ROM source is patched or bundled.
    stage = work / 'wheel-source'
    package = stage / 'multi_agent_ale_py'
    package.mkdir(parents=True, exist_ok=True)
    upstream = work / 'multi_agent_ale_py-0.1.12'
    for source in (upstream / 'multi_agent_ale_py').glob('*.py'):
        shutil.copy2(source, package / source.name)
    shutil.copy2(dll, package / 'ale_c.dll')
    shutil.copy2(upstream / 'LICENSE.md', package / 'LICENSE.multi-agent-ale.md')
    shutil.copy2(work / 'zlib-1.4' / 'LICENSE', package / 'LICENSE.zlib.txt')
    (package / 'BUILD_PROVENANCE.json').write_text(json.dumps({
        'upstream_version': '0.1.12', 'local_version': '0.1.12+mingw1',
        'source_modifications': 'none; standalone CMake build and local ctypes wheel packaging',
        'downloads': DOWNLOADS, 'sdl': False, 'static_cpp_runtime': True,
        'roms_bundled': False,
    }, indent=2), encoding='utf-8')
    (stage / 'setup.py').write_text('''from setuptools import setup, Distribution
from wheel.bdist_wheel import bdist_wheel
class Binary(Distribution):
    def has_ext_modules(self): return True
class CtypesWheel(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform
setup(name="multi-agent-ale-py", version="0.1.12+mingw1",
      description="Multi-agent Atari runtime; local Windows MinGW build",
      packages=["multi_agent_ale_py"], install_requires=["numpy"],
      package_data={"multi_agent_ale_py": ["*.dll", "LICENSE.*", "BUILD_PROVENANCE.json"]},
      distclass=Binary, cmdclass={"bdist_wheel": CtypesWheel}, license="GPL-2.0-or-later")
''', encoding='utf-8')
    wheel_dir = work / 'wheels'
    run([sys.executable, '-m', 'pip', 'wheel', stage, '--no-deps', '--no-build-isolation',
         '--wheel-dir', wheel_dir], env=environment)
    wheel = sorted(wheel_dir.glob('multi_agent_ale_py-0.1.12+mingw1-*.whl'))[-1]
    print(f'Built local wheel: {wheel}', flush=True)
    if not args.no_install:
        run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', wheel])
        run([sys.executable, '-c', 'import multi_agent_ale_py; print("Real multi-agent Atari runtime loaded.")'])


if __name__ == '__main__':
    main()

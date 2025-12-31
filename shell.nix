{ pkgs ? import <nixpkgs> { } }:

let
  nvidiaPackage = pkgs.linuxPackages.nvidia_x11;
in
pkgs.mkShell {
  packages = [
    pkgs.uv
    pkgs.python3
    pkgs.cudatoolkit
    pkgs.stdenv.cc         # full gcc with crt files
    pkgs.glibc.dev         # crti.o etc
  ];

  shellHook = ''
    echo "[shell] setting up CUDA + torch env"

    # C++ runtime so torch can load libstdc++.so.6
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

    # CUDA toolkit
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib64:$LD_LIBRARY_PATH

    # NVIDIA driver libraries (NixOS path; adjust if non-NixOS)
    if [ -d /run/opengl-driver/lib ]; then
      export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
    fi
    if [ -d ${nvidiaPackage}/lib ]; then
      export LD_LIBRARY_PATH=${nvidiaPackage}/lib:$LD_LIBRARY_PATH
    fi

    # Linker needs crti.o etc from glibc
    export LIBRARY_PATH=${pkgs.glibc}/lib:$LIBRARY_PATH
    export NIX_LDFLAGS="-L${pkgs.glibc}/lib $NIX_LDFLAGS"

    # Triton needs Python.h
    export C_INCLUDE_PATH=${pkgs.python3}/include/python3.13:$C_INCLUDE_PATH
    export CPLUS_INCLUDE_PATH=${pkgs.python3}/include/python3.13:$CPLUS_INCLUDE_PATH
    
    # Triton libcuda path for NixOS (avoids /sbin/ldconfig)
    export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib

    echo "[shell] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

    source .env
  '';
}


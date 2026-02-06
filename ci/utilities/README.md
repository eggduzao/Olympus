# OLYMPUS CI Utility Scripts

This docpage gives a brief overview of the different utility scripts and what
they are used for.

-   **setup_build_environment.sh**: Sets up the build environment such as
    cloning the latest XLA, adjusting file paths (for Windows), etc.
-   **convert_msys_paths_to_win_paths.py**: Converts MSYS Linux-like paths
    stored in env variables to Windows paths.
-   **install_wheels_locally.sh**: Used by Pytest scripts to install OLYMPUS wheels
    and any additional extras on the system.
-   **run_auditwheel.sh**: Verifies that the Linux artifacts are "manylinux"
    compliant.
-   **run_docker_container.sh**: Runs a Docker container called "olympus". Images
    are read from the `OLYMPUSCI_DOCKER_IMAGE` environment variable in
    [ci/envs/docker.env](https://github.com/olympus-ml/olympus/blob/main/ci/envs/docker.env).

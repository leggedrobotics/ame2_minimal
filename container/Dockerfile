# AME2 image: official Isaac Lab 2.3.2 (Isaac Sim 5.1) + this repo's rsl_rl and ame2.
# Build context is the repo root (see build.sh). Both packages are installed editable;
# run.sh bind-mounts the host checkouts over the same paths, so code edits need no rebuild.
ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
FROM ${BASE_IMAGE}

ENV AME2_ROOT=/workspace/ame2-minimal \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1

COPY rsl_rl/ ${AME2_ROOT}/rsl_rl
COPY ame2/ ${AME2_ROOT}/ame2

# the bundled rsl_rl replaces the image's stock rsl-rl-lib
RUN /isaac-sim/python.sh -m pip uninstall -y rsl-rl-lib && \
    /isaac-sim/python.sh -m pip install --no-deps -e ${AME2_ROOT}/rsl_rl && \
    /isaac-sim/python.sh -m pip install --no-deps -e ${AME2_ROOT}/ame2 && \
    /isaac-sim/python.sh -c "from importlib.metadata import version; print('rsl-rl-lib', version('rsl-rl-lib'), '| ame2', version('ame2'))"

WORKDIR ${AME2_ROOT}/ame2
ENTRYPOINT []
CMD ["/bin/bash"]

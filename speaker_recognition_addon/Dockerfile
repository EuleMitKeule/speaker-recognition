# Home Assistant add-on: Speaker Recognition
#
# The recognition server only builds on Python 3.9 (resemblyzer / torch pin to
# `python_version < '3.10'`). Rebuilding that ML stack on the Home Assistant
# Alpine base image fails, and copying the repo sources into the add-on build
# context only happens in CI (not when the Supervisor builds the add-on from
# git), which is why local installs failed.
#
# Instead we base the add-on on the already-published server image (built
# `FROM python:3.9-slim`). This makes the Supervisor build trivial and reliable:
# no source copying, no native ML compilation inside the add-on build.
#
# NOTE: `FROM` is hardcoded on purpose. The Supervisor / HA builder injects a
# `BUILD_FROM` build-arg, but we intentionally ignore it so the correct Python
# 3.9 base is always used regardless of the arch base image.
FROM ghcr.io/eulemitkeule/speaker-recognition:1.0.12

# Map the add-on options (/data/options.json) to the env vars the server reads,
# then launch the (already installed) recognition service.
COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]

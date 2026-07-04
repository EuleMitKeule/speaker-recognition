# Contributor Notes

## Repository Safety

- Verify the target remote before every push. Do not assume `origin` is the intended fork.
- Keep unrelated local files untouched; `.idea/` may exist as an untracked local directory.
- Prefer small, reviewable branches for each Home Assistant add-on milestone.
- Do not rewrite `master`/default branch unless the user explicitly asks for it.

## Current Branch Model

- `master` was restored to the clean fork baseline at `a5996a0`.
- Add-on startup stabilization work lives in `fix/stabilize-addon-startup`.
- The stabilization branch version is `0.1.1`.
- Home Assistant Supervisor supports custom repository branches with `#branch`.
- Test this branch in Home Assistant with:

```text
https://github.com/<owner>/speaker-recognition#fix/stabilize-addon-startup
```

## Add-on Publishing

- Add-on image is published to GHCR as:

```text
ghcr.io/<owner>/speaker-recognition-addon:<version>
```

- `speaker_recognition_addon/config.yaml` must use the same version as the image tag.
- `image:` in add-on config should not include a tag; Supervisor manages tags from `version`.
- The dedicated workflow is `.github/workflows/addon-image.yml`.
- Branch builds work for `push` events when the workflow branch filter includes the branch.
- `workflow_dispatch` behavior depends on workflows available from the default branch; prefer push-triggered branch builds for branch testing.

## Runtime Lessons

- Home Assistant add-on rootfs files under `speaker_recognition_addon/rootfs/**` must use LF line endings.
- s6 service `type` must be exactly `oneshot`, `longrun`, or `bundle`; CRLF breaks this.
- Do not add `CMD ["/init"]` to the add-on Dockerfile when using the Home Assistant base image.
- s6 service registration needs the user bundle marker:
  - `etc/s6-overlay/s6-rc.d/user/contents.d/speaker-recognition`
- Avoid setting Docker env `LOG_LEVEL=INFO`; it conflicts with bashio internals. Pass app log level as CLI args instead.
- The Python app has no `serve` subcommand. Run it as:

```text
python3 -m speaker_recognition --host ... --port ...
```

## Dependency Lessons

- The working standalone runtime used Python 3.9.
- The add-on must align with that runtime, because ML dependencies are guarded by Python version markers.
- Alpine/Python 3.12 skipped required ML packages such as `numpy`, causing runtime import failures.
- The add-on Dockerfile was stabilized by using a Debian Bullseye Home Assistant base with Python 3.9 packages.
- Keep a build-time smoke import for core runtime modules, including `numpy`, `torch`, `resemblyzer`, and `speaker_recognition.api`.

## Home Assistant Test Target

- Home Assistant host used during stabilization: `192.168.1.50:8123`.
- Add-on API/health target: `http://192.168.1.50:8099/health`.
- Add-on should bind to `0.0.0.0` and expose port `8099/tcp`.

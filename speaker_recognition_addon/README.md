# Speaker Recognition Add-on

Speaker recognition service for Home Assistant using voice embeddings.

## About

This add-on provides a speaker recognition service that can identify speakers based on their voice characteristics. It uses voice embeddings to create unique speaker profiles and can identify speakers in real-time.

## Configuration

### Option: `host`

The network address to bind the service to.

- Default: `0.0.0.0`
- Type: `string`

### Option: `port`

The port the service listens on.

- Default: `8099`
- Type: `port`

### Option: `log_level`

The logging level for the service.

- Default: `info`
- Type: `list(debug|info|warning|error|critical)`

### Option: `access_log`

Enable or disable HTTP access logging.

- Default: `true`
- Type: `bool`

### Option: `embeddings_dir`

Directory where voice embeddings are stored.

- Default: `/share/speaker_recognition/embeddings`
- Type: `string`

## API

The service exposes a REST API on the configured port. See the main project documentation for API endpoints and usage.

## Support

For issues and feature requests, please use the GitHub repository.

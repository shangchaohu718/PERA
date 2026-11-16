# GitHub Actions templates

The current publishing token has `repo` access without the `workflow` scope. These templates are provided as source and are not installed as active workflows.

An administrator with workflow-authorized credentials can copy `cpu.yml.example` to `.github/workflows/cpu.yml`. Its checks can already be run locally using the documented commands.

`pages.yml.example` additionally requires Pages hosting support for the private repository. GitHub currently rejects that configuration with HTTP 422. It publishes only `website/dist` and is manually triggered after installation.

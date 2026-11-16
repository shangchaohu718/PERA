# HoloMotion release status

The canonical release is the private [PERA repository](https://github.com/shangchaohu718/PERA). Its README, website source and reproducible inference resources belong together here.

| Component | Validation scope |
|---|---|
| Official public base | Fixed v1.4 `model_14000`; publisher download only |
| Single expert bank | 25 matching FFNs/routers; each checked against its independently loaded original |
| Scene suite | 15 tasks in three sequential simulation groups |
| Policy isolation | One Actor per group, independent robot caches, zero or one active domain per call |
| Source closure | Project imports confined to the extracted checkout |
| Installation | Isolated environment from reused local binary wheels plus small missing dependencies; no repeated CUDA download |
| Delivery | Private source repository and paired release archive; actual retrieval state is in the reproduction record |
| Website | Build and 40 ON/OFF media checks pass; Pages creation rejected by GitHub with HTTP 422 because the account plan does not support this private repository |
| Training reproduction | Not established by this inference release |

[SCENE_REPRODUCTION_CHECK.json](../provenance/SCENE_REPRODUCTION_CHECK.json) records actual completed checks. [WEBSITE_MIGRATION.json](../provenance/WEBSITE_MIGRATION.json) records the website migration. Shared-world contact sheets remain local diagnostics and are excluded from the homepage.

Scene replay verifies deterministic execution, tracking and interface preservation. It does not establish paper-wide success rates, DR generalization, old-motion retention, 15 uninterrupted task transitions or success of every historical expert.

The former public `pera-project` website is retained for rollback. PERA remains private; no visibility or account-plan change is performed. Historical audits are retained in [history/](history/HOLOMOTION_RELEASE_STATUS_20261113.md).

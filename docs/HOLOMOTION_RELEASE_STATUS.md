# HoloMotion release status

The canonical release is the [PERA repository](https://github.com/shangchaohu718/PERA). Its README, website source and reproducible inference resources belong together here.

| Component | Validation scope |
|---|---|
| Official public base | Fixed v1.4 `model_14000`; publisher download only |
| Single expert bank | 25 matching FFNs/routers; each checked against its independently loaded original |
| Scene suite | 15 tasks in three sequential simulation groups |
| Policy isolation | One Actor per group, independent robot caches, zero or one active domain per call |
| Source closure | Project imports confined to the extracted checkout |
| Installation | Isolated environment from reused local binary wheels plus small missing dependencies; no repeated CUDA download |
| Delivery | Source repository and paired release archive; current delivery is recorded in the live repository receipt |
| Website | Build and 40 ON/OFF media checks pass; Pages creation rejected by GitHub with HTTP 422 because the account plan does not support this private repository |
| Training reproduction | Not established by this inference release |

[RELEASE_HYGIENE.json](../provenance/RELEASE_HYGIENE.json) records the latest cleanup and fifteen-scene replay. [SCENE_REPRODUCTION_CHECK.json](../provenance/SCENE_REPRODUCTION_CHECK.json) retains the earlier isolated-install check. [The live delivery receipt](https://github.com/shangchaohu718/PERA/blob/main/provenance/RELEASE_DELIVERY.json) records the published asset and download verification. [WEBSITE_MIGRATION.json](../provenance/WEBSITE_MIGRATION.json) records the website migration. Shared-world contact sheets remain local diagnostics and are excluded from the homepage.

Scene replay verifies execution, tracking and interface preservation; small non-bitwise contact-trace differences are recorded in the latest check. It does not establish paper-wide success rates, DR generalization, old-motion retention, 15 uninterrupted task transitions or success of every historical expert.

The former public `pera-project` website is retained for rollback. Repository visibility is controlled separately by the owner. Historical audits are retained in [history/](https://github.com/shangchaohu718/PERA/blob/b7bdc1ab98a9af433a554140ea10124a19b5a27a/docs/history/HOLOMOTION_RELEASE_STATUS_20261113.md).

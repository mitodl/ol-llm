# Contracts

The rules every client in this repository implements, whatever the language. They are written
in [mitodl/ol-infrastructure `docs/plans/shared-llm-library-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/shared-llm-library-spec.md)
and move here as each is implemented:

- [the error contract](error-contract.md) (spec §3, with the policy and errors of §6)
- [the attribution schema](attribution.md) (spec §4)
- the model catalog, as data: [`python/src/ol_llm/catalog.toml`](../python/src/ol_llm/catalog.toml)
  ([`llm-usage-policies-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/llm-usage-policies-spec.md) §2.2)
- the entry points of the call contract (spec §6.6)
- prompt fetch semantics ([`llm-opik-lifecycle-spec.md`](https://github.com/mitodl/ol-infrastructure/blob/main/docs/plans/llm-opik-lifecycle-spec.md) §3)
- [evaluation gates](evaluation-gate.md) (lifecycle spec §5.1)

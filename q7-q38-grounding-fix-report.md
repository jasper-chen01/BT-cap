# Q7/Q38 grounding regression fix

## Scope

This fix is limited to the portal reasoning service and its grounding tests. It does not modify the Week 6 regression runner, generated regression artifacts, or model/data files.

## Root causes

- **Q7:** target-intent parsing treated the verb after `pathway` as a named pathway. In `Which receptors does the Glutamate pathway hit in this table?`, the service retained the valid `Glutamate` target but also invented an unsupported pathway named `hit`, so the API replaced the valid extractive result with `not in the tables`.
- **Q38:** the table tools intentionally support gene-family prefixes, so `GRIA` correctly produced empty T-cell/Ephys_2 lookups while the contrast and immune-pathway tools returned relevant evidence. Exact standalone-gene validation nevertheless required a literal `GRIA` row and rejected this valid negative-control result.

## Changes

- Pathway-target parsing now recognizes common post-`pathway` predicates such as `hit`, `bind`, `use`, `occur`, `target`, `signal`, and `involve` as verbs rather than pathway names.
- A family-level empty lookup can count as grounded negative evidence only when all of the following are true:
  - the question uses explicit negative-control language such as `refute`, `falsify`, `disprove`, `counterevidence`, or `contradict`; a bare `claim` is not sufficient;
  - the family prefix maps to real table genes elsewhere (for example, `GRIA2`, `GRIA3`, or `GRIA4` for `GRIA`);
  - the requested cell/Ephys-scoped family lookup is empty; and
  - a nonempty pathway contrast is routed for the same compartment and Ephys comparison, or a pathway trace matches the requested compartment/Ephys scope and the named family/program.
- Answers admitted through that negative-control rule stay deterministic and extractive even if `gemini` or `medgemma` is requested. The provider is not called.
- Ordinary positive family assertions and invented targets continue to fail closed as `not in the tables`.

## Regression coverage

- Q7 intent contains only `Glutamate` and no unsupported `hit` target.
- Q7 and Q38 reasoning API function results exactly equal the independent CLI engine's `--no-llm` extractive answers.
- Q7/Q38 extractive paths use a provider spy that fails the test if generation is attempted.
- Q38 remains extractive when `gemini` is explicitly requested.
- A positive, unsupported T-cell/Ephys_2 `GRIA` assertion still returns `not in the tables` without calling a provider.
- Positive `claim + contrast` requests for both `GRIA` and the broader `NRX` prefix remain unsupported and provider-free.
- Real, nonempty MK/tumor/Ephys_1 rows cannot corroborate an empty GRIA/T-cell/Ephys_2 lookup.

## Verification

- Focused Q7/Q38 and safeguard tests: **8 passed**.
- Backend reasoning, chat-grounding, and API tests: **212 passed**.
- Full available project suite (`backend/tests`, `fall_semester/rag_bioreasoning/tests`, and `tests`): **274 passed**.
- The only output was five existing dependency deprecation warnings (Pydantic and Starlette TestClient); there were no failures.


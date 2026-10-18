# Q7/Q38 grounding regression fix

## Scope

This fix is limited to the portal reasoning service and its grounding tests. It does not modify the Week 6 regression runner, generated regression artifacts, or model/data files.

## Root causes

- **Q7:** target-intent parsing treated the verb after `pathway` as a named pathway. In `Which receptors does the Glutamate pathway hit in this table?`, the service retained the valid `Glutamate` target but also invented an unsupported pathway named `hit`, so the API replaced the valid extractive result with `not in the tables`.
- **Q38:** the table tools intentionally support gene-family prefixes, so `GRIA` correctly produced empty T-cell/Ephys_2 lookups while the contrast and immune-pathway tools returned relevant evidence. Exact standalone-gene validation nevertheless required a literal `GRIA` row and rejected this valid negative-control result.

## Changes

- Pathway-target parsing now recognizes common post-`pathway` predicates such as `hit`, `bind`, `use`, `occur`, `target`, `signal`, and `involve` as verbs rather than pathway names.
- A family-level empty lookup can count as grounded negative evidence only when all of the following are true:
  - the question uses affirmative negative-control language such as `refute`, `falsify`, `disprove`, `counterevidence`, or `contradict`; a bare `claim` is not sufficient, and clause-aware negation handles contractions such as `don't`, `do not`, and arbitrarily long `without ... disprove/falsify` spans without activating the exception;
  - the family prefix maps to real table genes elsewhere (for example, `GRIA2`, `GRIA3`, or `GRIA4` for `GRIA`);
  - the requested cell/Ephys-scoped family lookup is empty; and
  - corroborating pathway rows contain the requested normalized cell identity and Ephys label in the same source, target, or celltype participant, match the requested compartment, and match the named family/program. A broad trace-level Ephys filter cannot substitute for that participant-level match.
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
- Positive support prompts containing `without refuting it` or `Do not refute it` remain unsupported and provider-free.
- Exact contraction and long-span prompts (`Don't refute ...` and `Without ... carefully disprove or falsify ...`) remain unsupported and provider-free.
- Real PTN rows from cycling/OPC/AC tumor identities cannot corroborate an empty MES-like/Ephys_2 GRIA lookup.
- A real `PPIA_BSG` row with T-cell/Ephys_1 as source and tumor/Ephys_2 as target cannot be misread as T-cell/Ephys_2 evidence.

## Verification

- Focused Q7/Q38 and safeguard tests: **14 passed**.
- Backend reasoning, chat-grounding, and API tests: **218 passed**.
- Remaining RAG and integration tests: **62 passed**.
- Full available project suite (`backend/tests`, `fall_semester/rag_bioreasoning/tests`, and `tests`), run in two resource-bounded groups: **280 passed**.
- The only warnings were five existing dependency deprecation warnings (Pydantic and Starlette TestClient); there were no failures.


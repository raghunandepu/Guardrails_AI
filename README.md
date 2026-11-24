# Guardrails AI - Quick Reference

This project notebook demonstrates how to validate assistant inputs and outputs, handle validation failures, extract structured data, and add guardrails to LangChain and streaming workflows.

Notebook: [Guardrails_ai_notebook.ipynb](./Guardrails_ai_notebook.ipynb)

Section numbers below match the notebook so you can follow the same sequence in both references.

## 1. Setup and Guardrails model

### What Guardrails does

A `Guard` runs one or more validators against text or structured output. A validator reports whether content passes and, when it fails, `OnFailAction` determines what happens next.

For direct text validation:

```python
result = guard.validate(text)
print(result.validation_passed)
print(result.validated_output)
```

For an LLM call, pass the model and messages to the guard. Guardrails validates the model response and can retry when configured to re-ask. The result includes the raw response, validation summaries, the final `validated_output`, and a `validation_passed` flag.

### Environment and imports

Use the existing `guardrailsAI` virtual environment and install dependencies from the project root:

```powershell
.\guardrailsAI\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The notebook prompts for `GROQ_API_KEY` without echoing it. Do not hard-code or commit API keys. The notebook uses LiteLLM's `groq/` provider prefix for `BOT_MODEL`; the LangChain `ChatGroq` example uses the model identifier expected by that adapter.

For readable tutorial output, the setup cell disables OpenTelemetry and Guardrails tracing and suppresses only identified library warnings/logs. It filters a PyTorch `FutureWarning` specific to Python 3.14; this is only an output cleanup, not a compatibility fix, so use a supported Python/PyTorch combination for production. Validation, model, and API errors remain visible.

`guardrails-ai` is the core framework. Each validator is a separate PyPI dependency, imported directly from its package:

```python
from guardrails import Guard, OnFailAction
from guardrails_ai.detect_pii import DetectPII
from guardrails_ai.competitor_check import CompetitorCheck
from guardrails_ai.toxic_language import ToxicLanguage
from guardrails_ai.restricttotopic import RestrictToTopic
```

Avoid the deprecated `guardrails.hub` import path for these validators.

## 2-5. Validators in the notebook

| Notebook section | What it checks | Configuration and example |
|---|---|---|
| **2. `DetectPII`** | Detects selected personal data such as credit-card numbers, email addresses, and phone numbers. | Uses local detection and `FIX` to replace detected values with placeholders such as `<CREDIT_CARD>` and `<EMAIL_ADDRESS>`. |
| **3. `CompetitorCheck`** | Finds configured competitor names in a response. | Checks Stripe, PayPal, and Square. A passing response has no mentions; `FIX` replaces detected names with `[COMPETITOR]`. |
| **4. `ToxicLanguage`** | Scores language for toxicity. | Uses sentence-level checks and a threshold of `0.5`. In the notebook's rude-text example, the configured `FIX` action returns an empty string. |
| **5. `RestrictToTopic`** | Checks whether text stays within an allowed topic scope and avoids disallowed topics. | Allows payments, refunds, accounts, cards, and wallets; rejects topics such as politics, medical advice, and investing tips. |

`use_local=True` selects local validator behavior for the PII, competitor, and toxic-language examples. Topic classification uses a custom LLM callback: the notebook calls LiteLLM with the Groq model and expects JSON containing `topics_present`.

The topic example sets `disable_classifier=True` to skip the built-in classifier, passes `groq_topic_classifier` as `llm_callable`, and uses `EXCEPTION` for failures. The callback validates the expected JSON shape and lets model, parsing, or malformed-response errors propagate instead of treating them as an empty topic list.

## 6. Choosing an `OnFailAction`

`OnFailAction` controls the response to a validator failure; it does not change what the validator checks.

| Action | Behavior | Notebook use |
|---|---|---|
| `FIX` | Apply the validator's correction to the invalid content, when available. | Replaces PII or competitor spans; the toxic-language example removes the offending output. |
| `FILTER` | Filter invalid content. For a structured response, the affected field can be omitted. | Demonstrated with competitor mentions. |
| `REFRAIN` | Withhold the output rather than return content that failed. | In the structured example, the guarded output is empty. |
| `EXCEPTION` | Raise `ValidationError` to let the caller handle the failure explicitly. | Used for off-topic input and competitor mentions. |
| `REASK` | Ask the LLM to produce a corrected response, subject to the retry limit. | The example allows up to two re-asks and inspects each iteration in `guard.history`. |

Choose the action based on the application: redact or repair when safe, filter or refrain when content should not be exposed, raise when the caller must decide, and re-ask when the model can reasonably correct its response.

## 7. Structured extraction with Pydantic

The refund example turns an unstructured customer email into a typed record:

```python
from typing import Literal
from pydantic import BaseModel, Field

class RefundRequest(BaseModel):
    customer_name: str
    amount: float
    reason: str
    category: Literal["billing", "technical", "account", "other"]
    urgency: Literal["low", "medium", "high"]
```

`Field(description=...)` provides field guidance, while `Literal` constrains values to an allowed set. Create a guard with `Guard.for_pydantic(RefundRequest)` and call it with the model and messages. The guard checks the returned data against the schema and can re-ask when parsing or validation fails. Inspect `result.validated_output` and `result.validation_passed` before using the extracted data.

The notebook also attaches a validator to an individual Pydantic field using `json_schema_extra={"validators": [validator]}`. This demonstrates that failure handling can be scoped to a field: `FILTER` can omit the invalid field, while `REFRAIN` can suppress the structured output.

## 8. LangChain integration

The notebook uses LangChain's composable pipeline:

```python
chain = prompt | llm | StrOutputParser()
```

The prompt template supplies system and user messages, `ChatGroq` calls the model, and `StrOutputParser` converts the model result to text. To validate that text, convert a Guardrails guard to a runnable and add it after the parser:

```python
guarded_chain = prompt | llm | StrOutputParser() | guard.to_runnable()
```

The last stage ensures downstream callers receive guard-processed text. `ConsoleCallbackHandler` can be passed in the LangChain config to inspect execution traces.

## 9. Validating streamed output

Set `stream=True` in a Guardrails model call to receive output incrementally. Each chunk can expose `validated_output`; the notebook prints non-empty validated chunks as they arrive. This keeps validation in the streaming path rather than validating only after the full response has been assembled.

## Quick troubleshooting

- **Validator import fails:** install the corresponding `guardrails-ai-...` package from `requirements.txt`, then use its `guardrails_ai.<validator>` import path.
- **`langchain_groq` is missing:** install the requirements in the existing `guardrailsAI` environment and ensure the notebook kernel uses that environment.
- **`ValidationError` appears:** this is expected when a guard uses `OnFailAction.EXCEPTION` and the input fails validation.
- **Remote model call fails:** verify `GROQ_API_KEY`, the selected model, and the provider prefix used by the calling library.
- **Telemetry export warning:** an OpenTelemetry export timeout is separate from validator pass/fail; inspect the Guardrails result or exception to determine whether validation itself failed.

"""Tool schema shared by the live agent loop, in Anthropic tool-use format."""

TOOLS = [
    {
        "name": "click",
        "description": "Click an element. Provide a css_selector and/or visible text; the surface tries css_selector first, falling back to text, falling back to ocr_text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "css_selector": {"type": "string"},
                "text": {"type": "string"},
                "ocr_text": {
                    "type": "string",
                    "description": (
                        "last resort: text you can see in the screenshot's pixels (the 'OCR text detected' section "
                        "of the prompt, if present) but that has NO DOM/accessible-text equivalent -- canvas-rendered "
                        "content, an image, a font icon with no name. Clicks the matched word's on-screen position, "
                        "found by a fresh OCR pass, not a DOM query. Only use this when css_selector/text genuinely "
                        "can't work; for ordinary markup, DOM selectors and visible text are far more robust."
                    ),
                },
                "reasoning": {"type": "string", "description": "why this is a robust way to find this element"},
                "idempotency_note": {
                    "type": "string",
                    "description": "optional: what re-clicking this same element after a partial success would do (e.g. 'safe, no-op' vs 'would double-submit') -- only meaningful when recording a reusable capability, ignored otherwise",
                },
            },
        },
    },
    {
        "name": "type",
        "description": "Type text into an input field.",
        "input_schema": {
            "type": "object",
            "properties": {
                "css_selector": {"type": "string", "description": "locates the input field -- NOT the text to type"},
                "find_field_by_text": {
                    "type": "string",
                    "description": "alternative way to locate the input field by its own visible label text -- NOT the text to type",
                },
                "find_field_by_ocr_text": {
                    "type": "string",
                    "description": (
                        "last resort: locate the input field by a label visible only in the screenshot's pixels, "
                        "with no DOM/accessible-text equivalent -- see click's ocr_text field for when this applies. "
                        "NOT the text to type."
                    ),
                },
                "text": {"type": "string", "description": "the actual text to type into the field once located"},
                "reasoning": {"type": "string"},
                "idempotency_note": {"type": "string", "description": "see click's field of the same name"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "select",
        "description": "Choose an option from a <select> element.",
        "input_schema": {
            "type": "object",
            "properties": {
                "css_selector": {"type": "string"},
                "value": {"type": "string"},
                "reasoning": {"type": "string"},
                "idempotency_note": {"type": "string", "description": "see click's field of the same name"},
            },
            "required": ["css_selector", "value"],
        },
    },
    {
        "name": "wait",
        "description": "Wait for a fixed duration before observing again.",
        "input_schema": {"type": "object", "properties": {"ms": {"type": "integer"}}},
    },
    {
        "name": "assert_text",
        "description": "Check whether text is present on the current page.",
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    {
        "name": "go_to",
        "description": "Navigate to a URL. Only allowlisted domains/routes will succeed.",
        "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    },
    {
        "name": "done",
        "description": "Declare the goal achieved. Call this once the checkpoint condition is visibly true.",
        "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}},
    },
    {
        "name": "ask_user",
        "description": "Stop and ask the human a question instead of guessing.",
        "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
    },
]

# Only added to a decider's tool list when recording a reusable capability
# (agent/loop.py's run_discovery_loop, via scripts/record_capability.py) --
# never exposed to an ordinary live run (agent/loop.py's plain run_agent_loop,
# with no script wiring it up to a CLI at the moment), where they'd have
# nothing to attach to and would just bloat the prompt. These are
# the tools that let a recording session declare its own contract -- the
# /declare_checkpoint, /declare_output, /declare_business_outcome, and
# /declare_recoverable HTTP endpoints -- same declared_outcomes.json shape,
# same consumer (artifact/from_run.py), just called by the model itself
# instead of relayed by a human driving curl.
DISCOVERY_TOOLS = [
    {
        "name": "declare_checkpoint",
        "description": (
            "Declare the condition that proves the goal was genuinely achieved -- this becomes the reusable "
            "capability's success check on every future replay. Call this exactly once, once you can see the "
            "condition is true, before calling done."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["text_present", "text_absent", "url_contains", "element_visible", "selector_visible"], "description": "text_present/text_absent: a substring of the page text. url_contains: a substring of the URL. element_visible: exact VISIBLE TEXT that must be on screen (not a selector). selector_visible: a CSS or Playwright selector for an element that must be visible -- use this for a structural condition like \"the row for this member exists\"."},
                "value": {"type": "string", "description": "the text, URL substring, or selector -- whichever `type` says"},
            },
            "required": ["type", "value"],
        },
    },
    {
        "name": "declare_output",
        "description": (
            "Declare one piece of data this capability should hand back to whoever calls it, and exactly where to "
            "read it from once the checkpoint is met. Call once per output value, before calling done. Skip this "
            "entirely if the goal has no data to return (e.g. it's a pure action like 'submit this form')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "output field name, e.g. 'view_count'"},
                "value_type": {"type": "string", "enum": ["string", "integer", "number"], "description": "defaults to string if omitted"},
                "target_kind": {"type": "string", "enum": ["dom_selector", "text"], "description": "how to locate the element to read the value from"},
                "target_value": {"type": "string", "description": "the css selector or exact visible text to read this value from"},
            },
            "required": ["name", "target_kind", "target_value"],
        },
    },
    {
        "name": "declare_business_outcome",
        "description": (
            "Declare a legitimate non-success result this goal can hit (e.g. 'no such item found') -- so a future "
            "replay reports it as expected data, not a crash. Only call this if you actually observe such a state "
            "during this run; never guess one that didn't happen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "short identifier, e.g. 'no_matching_result'"},
                "match_type": {"type": "string", "enum": ["text_present", "text_absent", "url_contains", "element_visible", "selector_visible"], "description": "text_present/text_absent: a substring of the page text. url_contains: a substring of the URL. element_visible: exact VISIBLE TEXT that must be on screen (not a selector). selector_visible: a CSS or Playwright selector for an element that must be visible -- use this for a structural condition like \"the row for this member exists\"."},
                "match_value": {"type": "string", "description": "the text, URL substring, or selector -- whichever `match_type` says"},
            },
            "required": ["name", "match_type", "match_value"],
        },
    },
    {
        "name": "declare_recoverable",
        "description": (
            "Declare a transient, dismissable state you actually had to work around during this run (e.g. a "
            "newsletter popup) -- so a future replay can recognize and clear it automatically instead of failing. "
            "Only call this if you actually hit and recovered from such a state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "match_type": {"type": "string", "enum": ["text_present", "text_absent", "url_contains", "element_visible", "selector_visible"], "description": "text_present/text_absent: a substring of the page text. url_contains: a substring of the URL. element_visible: exact VISIBLE TEXT that must be on screen (not a selector). selector_visible: a CSS or Playwright selector for an element that must be visible -- use this for a structural condition like \"the row for this member exists\"."},
                "match_value": {"type": "string", "description": "the text, URL substring, or selector -- whichever `match_type` says"},
                "recovery": {"type": "string", "description": "human-readable description of how you cleared it"},
            },
            "required": ["name", "match_type", "match_value", "recovery"],
        },
    },
]

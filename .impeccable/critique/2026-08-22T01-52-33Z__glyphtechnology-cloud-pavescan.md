---
target: "https://glyphtechnology.cloud/Pavescan/"
total_score: 25
max_score: 36
na_heuristics: 7
p0_count: 0
p1_count: 3
timestamp: 2026-08-22T01-52-33Z
slug: glyphtechnology-cloud-pavescan
---
**Method: dual-agent (A: general-purpose subagent · B: general-purpose subagent)**

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | Submitting the classify form triggers real CNN inference with zero feedback until full page reload; nothing prevents a double-submit |
| 2 | Match Between System and Real World | 2 | Result page shows the raw model string twice: a "LOW SEVERITY" chip, then the unformatted class string "pothole low severity" right below it |
| 3 | User Control and Freedom | 3 | Low-stakes flow, easy return path, nothing destructive |
| 4 | Consistency and Standards | 3 | Tokens/components applied consistently; `.navbar-pill` class name no longer matches what it renders (code hygiene only) |
| 5 | Error Prevention | 2 | No client-side file-type check or preview before submit; native `required` popup is unstyled |
| 6 | Recognition Rather Than Recall | 3 | Icons carry text labels; filename shown after selection but no image thumbnail preview |
| 7 | Flexibility and Efficiency | n/a | Single-action demo tool, no expert/repeat workflow to accelerate |
| 8 | Aesthetic and Minimalist Design | 3 | Large dead black voids on /classify and /model at desktop width read as unfinished, not minimal |
| 9 | Error Recovery | 4 | The invalid-file error state is genuinely excellent, plain language, names the fix, preserves the form |
| 10 | Help and Documentation | 3 | /model functions as real, task-focused documentation of the system |
| **Total** | | **25/36** | **Acceptable, bordering on Good (69%)** |

## Design Specificity Verdict

Category-interchangeable, with a thin layer of product-specific dressing. Strip the road favicon and the word "pavement" and this is a generic dark dev-tool shell. Homepage sells "upload a photo, get a classification" using zero photographs of pavement; five distress types described in identical text-only cards with no visual differentiation; severity chips are the generic CI/uptime-monitor status-light pattern. The one place the system speaks in its own voice, .hero-status's real model numbers, is the exception, not the rule.

Deterministic scan (detect.mjs --json templates/) ran (exit 0) but self-reported DEGRADED, required parser modules unreachable from the skill install, fell back to regex matching with contrast/selector evaluation disabled, returned []. Undercount, not a clean bill of health. No false positives (nothing fired).

In place of the disabled contrast engine, manual computation from main.scss's CSS custom properties found a real failure: .dropzone-label's dashed border (#232833) against its own background (#1a1e26) computes to approximately 1.13:1, far under the WCAG 1.4.11 minimum of 3:1 for UI-component boundaries. Also flagged: .model-table th and .dropzone-hint text at approximately 4.85:1 (passes AA, no margin), decorative FontAwesome icons missing aria-hidden across every template, target="_blank" footer links with no new-window warning, identical <title> tags on all four pages.

Visual overlays unavailable, no interactive browser automation tool exposed in this session, only static headless-screenshot capability.

## Overall Impression

Coherent, accessible in its primary text/color contrast, one genuinely excellent moment (the error state). But presently a well-executed template rather than a product-specific interface: nothing shows pavement, a photo, or a prediction until the visitor does the work themselves, and the moment that should feel most convincing, the actual result, is undercut by a raw, redundant debug-looking string.

## What's Working

1. The error state (.form-alert--error) is the best-designed moment in the product: specific, actionable, preserves the form, never crashes.
2. The severity chip pattern (neutral surface + colored dot + tinted border) is a documented, deliberate accessibility decision, verified at 13.85:1 text contrast regardless of which status is active.
3. The .hero-status line is the one place the design commits to a real, verifiable fact instead of decoration.

## Priority Issues

- [P1] Dropzone drop-target border is functionally invisible. .dropzone-label's dashed border computes to approximately 1.13:1 contrast against its own background, versus the 3:1 WCAG minimum. The design's own signal for its primary interaction silently fails. Fix: lighten the dashed border color so it clears 3:1 against $surface-raised. Suggested command: /impeccable harden.

- [P1] Raw, redundant class label on the result page. result.html prints {{ predicted_class }} verbatim ("pothole low severity"), restating the severity chip above it in different casing, with raw backend strings leaking straight to the UI. This is the moment meant to prove the model is real and credible. Fix: split type and severity; show the type once, title-cased and de-hyphenated, matching the homepage's vocabulary. Suggested command: /impeccable clarify.

- [P1] No way to try the model without your own photo. Sample images exist in the repo but /classify gives no one-click way to try the model. The stated primary user (technical evaluator) faces friction finding a qualifying photo. Fix: surface 2-3 sample images as one-click "try a sample" thumbnails on /classify. Suggested command: /impeccable onboard.

- [P2] No loading/pending state during inference. Submitting triggers real CNN inference with zero visual feedback until page reload; button stays fully clickable, inviting double-submits. Fix: disable the button and swap its label to "Analyzing..." on submit. Suggested command: /impeccable harden.

- [P2] Dead space on /classify and /model. At desktop width, large empty black canvas around the card / around the content column. Reads as unfinished rather than deliberately minimal. Fix: don't vertically center in a full-viewport flex container, or give the space a job. Suggested command: /impeccable layout.

## Persona Red Flags

Alex (technical evaluator, the actual stated primary persona): Has to leave the site to find a pavement photo before seeing the demo work at all, despite sample images sitting unused on disk. Once they get a result, "pothole low severity" as raw text reads as an unfinished detail.

Jordan (first-timer): Clicks "Analyze image," sees nothing happen for the duration of inference, may click again assuming it's broken. Sees the same fact stated twice in two formats. If they land on a "Rutting" result (the one class with no severity split), they get no severity chip at all with no explanation.

Sam (accessibility-dependent): No FontAwesome icon across any template carries aria-hidden="true", independently flagged by both assessments. Counterpoint, genuinely solid: the dropzone's visually-hidden-not-display:none input is correctly keyboard-reachable with a visible focus ring, and severity is never conveyed by color alone.

## Minor Observations

- .navbar-pill no longer describes what it renders (a flat underline bar, not a pill).
- "See road distress the moment you spot it" implies live/real-time capture; actual flow is upload-after-the-fact.
- Result image's alt="Uploaded road surface" doesn't include the classification outcome.
- "Rutting" class renders with no severity chip and no explanation why, unlike every other result.
- All four pages share an identical <title>Distress Detection</title>.
- Footer's target="_blank" links carry no "opens in new tab" indication.
- .model-table th and .dropzone-hint text sit at approximately 4.85:1 contrast, passes AA but with no margin.

## Questions to Consider

- If the evaluator's whole judgment happens in the ~30 seconds after clicking "Analyze," why does that exact moment show the least polished string in the app?
- What would the homepage look like if it had to sell the product with one real photo and one real prediction instead of three abstract numbers?
- Given sample images already sit unused on disk, what's the actual cost of surfacing them versus the friction they'd remove from the one persona this product exists to convince?

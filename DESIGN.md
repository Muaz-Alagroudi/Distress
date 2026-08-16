# Design

Restyled 2026-08-13 to match Glyph's standard design system: flat/minimal, dark,
CSS custom property tokens, no inline styles, mobile-first. Core functionality
(upload an image, POST to `/predict`, show the predicted class) is unchanged,
only the templates and stylesheet were rewritten from scratch.

## Tokens

All colors live as CSS custom properties in `:root`, defined once in
`static/styles/main.scss`. Everything else (this file's Sass, the templates)
reads them through the short aliases in `static/styles/_variables.scss`
(`$amber`, `$asphalt`, `$surface`, `$offwhite`, plus `-rgb` companions for
`rgba()` calls). Change a hex value in one place and it propagates everywhere.

Palette: an asphalt-dark background (`--color-bg`) with a road-marking amber
accent (`--color-primary`), plus `--color-success` / `--color-warning` /
`--color-danger` used to color-code the predicted severity (low / medium /
high) on the result page.

## Structure

- `static/styles/_variables.scss`: Sass aliases + breakpoints, no rules
- `static/styles/main.scss`: source of truth, edit this
- `static/styles/main.css`: compiled output, what the templates actually load, regenerate with `npm run build:css` (see [README.md](README.md))
- `static/js/main.js`: one small progressive enhancement, shows the chosen filename in the dropzone. The form works fine without it.

## Pages

- `/` (`templates/home.html`): explains what the model detects and how it works, links to `/classify` and `/model`. No upload form here.
- `/classify` (`templates/classify.html`): the upload form, this is what used to live at `/`.
- `/model` (`templates/model.html`): a 4-step pipeline (input image → preprocessing → model → output) followed by the layer-by-layer architecture table. The numbers came from actually loading the model and reading `model.summary()` / `layer.count_params()`, not from the commented-out training code in `main.py` (that snippet is a similar but not necessarily identical draft), and were cross-checked against the sum of the table's own `Params` column (1,626,253, matches `model.count_params()` exactly). Re-derive it the same way if the model is ever retrained with a different architecture. Note the trainable-weight size (~6.2MB) is not the same number as `distressCNN.keras`'s file size on disk (~18.6MB): the saved file also bundles the Adam optimizer's moment-estimate state, roughly 2x the weight count, don't confuse the two when quoting a "model size."
- `/predict` (`templates/result.html`): POST target, shows the image, severity badge, and predicted class. "Analyze another image" goes back to `/classify`.
- `templates/navbar.html` and `templates/footer.html` are shared includes across all pages. The navbar highlights the active link off `request.endpoint` (not `request.path`, which under `APP_PREFIX` doesn't include the `/Pavescan` prefix that `url_for()` adds). Route function for `/model` is named `model_page`, not `model`, that name is taken by the loaded Keras model in `backend.py`.
- `static/favicon.svg`: the FontAwesome "road" glyph (amber) on a rounded asphalt-dark square, linked from every template's `<head>`.

## Conventions

- **Text aligns left, not center.** `.card`, `.hero`, and `.section` all use `text-align: left`. Earlier drafts centered everything; it read as generic marketing-page style rather than a tool. Self-contained widgets (the dropzone, buttons) can still center their own icon+label internally, that's a widget choice, not body text.
- **Never let the plain `a` reset touch `.btn`.** The global link color rule is `a:not(.btn)`, not bare `a`. Reasoning: a plain `a` and a one-class button selector like `.btn--primary:hover` tie on CSS specificity (one class/pseudo-class each); the tie-break then falls to the element selector in `a:hover`, which silently wins and overrides the button's intended hover color. Concretely this made CTA link-buttons go invisible on hover (text and background both resolved to the same tint). Any new global element-level reset in this file (links, or anything else applied via a bare tag selector) should get the same `:not(.btn)` treatment, or explicitly redeclare `color` inside every `.btn*:hover` block, don't rely on specificity ties resolving the way you'd expect.

## Icons

FontAwesome, loaded from a CDN `<link>` in each template (no build step needed
since these are plain Jinja templates, not a bundled React app). Covers both
UI icons (`fa-solid`) and the brand icons in the footer (`fa-brands`).

## What changed vs. what didn't

- Rewrote `templates/index.html` and `templates/result.html` markup and all CSS from scratch.
- Removed `static/styles/results.css` (unused, dead file, not referenced by any template).
- Added a severity badge (low/medium/high, color-coded) on the result page, derived from the existing `predicted_class` string, no backend change.
- `backend.py` routes, model loading, and the `/predict` contract are untouched.

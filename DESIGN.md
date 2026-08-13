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

## Icons

FontAwesome, loaded from a CDN `<link>` in each template (no build step needed
since these are plain Jinja templates, not a bundled React app). Covers both
UI icons (`fa-solid`) and the brand icons in the footer (`fa-brands`).

## What changed vs. what didn't

- Rewrote `templates/index.html` and `templates/result.html` markup and all CSS from scratch.
- Removed `static/styles/results.css` (unused, dead file, not referenced by any template).
- Added a severity badge (low/medium/high, color-coded) on the result page, derived from the existing `predicted_class` string, no backend change.
- `backend.py` routes, model loading, and the `/predict` contract are untouched.

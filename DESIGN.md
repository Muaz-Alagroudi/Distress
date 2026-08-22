# Design

Rebuilt from scratch 2026-08-22 as **The Pavement Condition Survey**, a
full visual-world replacement chosen through `/impeccable shape`: the site
now presents itself as the literal DOT/PCI field-inspection form the
product automates, not a dashboard performing intelligence. This replaces
the same-day "engineering console" dark system (road palette, JetBrains
Mono + IBM Plex Sans), which itself replaced a light "Ventriloc" editorial
theme, which replaced the original dark asphalt-amber theme from
2026-08-13. The user's framing going in: "i dont like how the website
looks like, i want to recreate it from scratch," explicitly ruling out
carrying anything forward from the prior looks. Core functionality
(upload an image, POST to `/predict`, show the predicted class and
severity) is unchanged; every template and the stylesheet were rebuilt.

The direction was one of seven grounded candidates from a concept-seed
roll (seed key `e0a861d5`), presented as "IMPECCABLE'S PICK" rather than
the dice-assigned card ("The Structural Test Report," a materials-testing
lab-report genre) and chosen by the user over a competitive "Creator
Hardware Bench" alternate (a physical desk-instrument metaphor). No image
generation was available in this environment, so this was a code-led
build with no approved comp; ambition lived in the direction contract's
FIRST VIEWPORT block instead (still readable as an HTML comment at the
top of `templates/home.html`'s `<body>`). The build went through one
finish-review round and one fix batch (banned kicker badges removed, a
hard-offset button shadow corrected to a real soft shadow, severity
checkbox colors collapsed to keep verdict-red singular, an unused
hazard-stripe motif put to real structural use, a 13-classes-vs-5-types
ambiguity clarified) before shipping clean on the verdict pass.

## Direction

A real DOT/PCI pavement-condition survey form, rendered at full fidelity:
white form paper, inspection-navy ink, a hazard-yellow header band, ruled
ledger tables, a numbered distress-code legend, printer's registration
corners on anything that behaves like a document field, and a rubber-stamp
certification mark on the result page. The honest risk named at brief time
was that this is the most literal possible reading of "pavement distress,"
so it earns its keep only through genre-accurate form conventions and real
content, never generic yellow-and-navy civil-engineering vibes.

**Public Sans carries every heading, label, and UI string.** Chosen for
its literal institutional tie (it is the US federal design system's own
typeface), not as a display costume. **IBM Plex Mono is reserved for
genuine tabular data only**: the distress-code legend's code column, the
architecture table, the form-footnote line, badges/tags that quote a real
value. Nothing decorative borrows it.

**Verdict-red (`--color-verdict`) is reserved for exactly one meaning: the
classification moment.** It appears on the result page's high-severity
checkbox/stamp and nowhere else — not as a second accent, not as a
"warning" color elsewhere, not on the error banner beyond its top rule.
An earlier draft gave low/medium severity their own green/amber tokens;
finish review flagged this as breaking the "verdict-red only" rule (two
more hues carrying meaning outside the one reserved site), so those tokens
were removed. Low and medium severity now confirm in plain `$navy`, the
system's ordinary ink, so red keeps its singular weight.

**The hazard-stripe motif is structural, never decorative filler.** It
renders once, as `.navbar-boundary`, the seam between the header band and
every page's content — the one place a caution-stripe genuinely marks a
boundary. It does not appear as a section divider or ornament anywhere
else. (An earlier draft defined the mixin but never called it anywhere;
finish review caught the unused motif and it was wired into that single
structural seam.)

## Tokens

All colors live as CSS custom properties in `:root`, defined once in
`static/styles/main.scss`. Everything else (this file's Sass, the
templates) reads them through the short aliases in
`static/styles/_variables.scss` (`$paper`, `$ink`, `$navy`, `$hazard`,
`$verdict`, etc., plus `-rgb` companions for `rgba()` calls).

Palette: `--color-paper` (#f5f4ef, page background, a cool-neutral form
stock, deliberately not a warm cream/parchment tone) and `--color-paper-alt`
(#ffffff, cards and table cells) are the two form-paper surfaces.
`--color-ink` (#20242a) is primary text; `--color-ink-dim` (#565d64,
6.06:1 on paper) covers every secondary/tertiary text need, there is no
third "faint" tier. `--color-navy` (#0f2a4a) is the system's one ink color
for structure: nav, buttons, borders, headings that need weight,
severity's default checked state. `--color-hazard` (#f4c11e) is the header
band and the hazard-stripe's bright half; `--color-hazard-ink` aliases to
navy for anything printed on the yellow band (8.61:1). `--color-verdict`
(#c73a26) is reserved as described above. `--color-line` (#c9c4b4, under
3:1, decorative hairlines only) and `--color-line-strong` (#726c5d,
4.75:1) split hairline rules from anything an interaction depends on
seeing (the dropzone border, registration corners).

## Structure

- `static/styles/_variables.scss`: Sass aliases + breakpoints, no rules
- `static/styles/main.scss`: source of truth, edit this
- `static/styles/main.css`: compiled output, what the templates actually load, regenerate with `npm run build:css` (see [README.md](README.md))
- `static/js/main.js`: one small progressive enhancement, shows the chosen filename in the dropzone. The form works fine without it.

## Pages

- `/` (`templates/home.html`): a report cover sheet. Left column: title,
  intro, "Log sample unit" / "View methodology" CTAs. Right column: a
  ruled `.summary-panel` with dotted-leader key:value rows (distress
  classes, severity levels, photos per analysis) — a real ledger table,
  never a stat-card cluster. A mono `.form-footnote` strip below states
  real model facts ("FORM PC-1 · MODEL distressCNN.keras · 1,626,253
  PARAMS"). Below that, the `.legend-table` lists the 5 real-world
  distress types as numbered codes (01–05), with a clarifying line
  explaining the 5-types-times-3-severities-plus-rutting = 13 trained
  classes arithmetic, so the "13" in the summary panel and the 5-row
  legend don't read as contradictory.
- `/classify` (`templates/classify.html`): the upload form, `.dropzone`
  with printer's registration corners, samples rendered as evidence tiles
  ("SAMPLE · Pothole" etc). `.form-alert--error` renders a stamped
  rejection banner (neutral surface, a `border-top: 3px solid $verdict`,
  never a colored left border) when `error` is passed.
- `/model` (`templates/model.html`): "Test methodology," a 4-step
  procedure followed by the layer-by-layer architecture table in
  `.model-table` (IBM Plex Mono, `tabular-nums`) — this page was always
  the system's best genre fit and needed the least conceptual reframing.
- `/predict` (`templates/result.html`): a completed, stamped survey
  record. A hand-inked certification stamp (SVG, `feTurbulence` +
  `feDisplacementMap` roughing the ring only, never the text — text and
  the checkmark/dash glyph live in a separate unfiltered `<g>`, or the
  displacement destroys the `textPath` lettering) animates in once on
  load (`stamp-impress` keyframes, respects `prefers-reduced-motion`).
  Ring text reads "CERTIFIED SURVEY RECORD" when severity applies,
  "SAMPLE LOGGED · NOT SEVERITY-RATED" when it doesn't (Rutting). A
  `.severity-checklist` (☐ Low ☐ Medium ☐ High, drawn as real boxes with
  an `fa-check` icon, never a raw Unicode glyph) shows the matched
  severity checked in navy, or verdict-red only for High. The photo sits
  in an `.exhibit-frame` with the same registration-corner treatment as
  the dropzone, tagged "Exhibit A."
- `templates/navbar.html` / `templates/footer.html`: shared includes. The
  navbar is a hazard-yellow band (brand + "FORM PC-1" tag left, tabs
  center, filled navy "Classify" CTA right), with `.navbar-boundary` (the
  hazard-stripe) as the one structural seam under it. Nav highlights the
  active link off `request.endpoint`, not `request.path`.
- `static/favicon.svg`: the FontAwesome "road" glyph, navy on a
  hazard-yellow rounded background.

## Conventions

- **Text aligns left.** No centered hero copy; this reads as a document, not a marketing page.
- **No kickers or eyebrows above headings.** A small label directly above an `<h1>` (tried in an early draft as "FIELD ENTRY" above "Log a sample unit," "SURVEY RECORD · COMPLETE" above the result heading) was flagged in finish review and removed outright — the heading carries its own weight, full stop, no exception.
- **Verdict-red is singular.** See Direction above; do not add a second severity color without removing this line.
- **The hazard-stripe motif renders in exactly one place** (`.navbar-boundary`). Do not add it as a section divider elsewhere without a structural reason as strong as "boundary between header and content."
- **IBM Plex Mono is for real data only**: codes, param counts, table figures, the form-footnote line. Everything else, including nav/buttons/labels, is Public Sans.
- **Shadows carry an offset and a blur, never a flat `Npx Npx 0`.** This system isn't neobrutalist; a hard offset shadow on the primary button was caught and corrected to `0 2px 5px rgba(navy, .28)`.
- **The dropzone and exhibit-frame inputs use printer's registration corners** (`registration-corners` mixin) as the form-field framing device; this is the one place corner marks appear, not a repeated decorative motif.
- **The dropzone input is visually hidden, not `display:none`.** Keeps it keyboard-reachable; the label gets a focus ring via `.dropzone:focus-visible + .dropzone-label`.
- **`/predict` never crashes to a raw 500 page.** A missing or invalid file re-renders `classify.html` with `.form-alert--error`.
- **The result page never prints a raw `class_names` string.** Always through `describe_prediction()` in `backend.py`.
- **Non-text UI boundaries need 3:1 contrast, not just 4.5:1 body text.** `$line` (1.58:1) is decorative-only; anything an interaction depends on seeing (`dropzone`/`exhibit-frame` borders) uses `$line-strong` (4.75:1) instead.
- **The certification stamp's ink-roughening filter (`feTurbulence`/`feDisplacementMap`) applies only to the ring strokes, never to the `<textPath>` text or the checkmark path.** Applying it to text at this scale erases the glyphs entirely; keep the filtered group and the text/icon group separate.

## Icons

FontAwesome, loaded from a CDN `<link>` in each template. Covers UI icons (`fa-solid`), the footer's brand icons (`fa-brands`), and the severity checklist's check glyph (`fa-check`, not a raw Unicode character).

## What changed vs. what didn't

- Full token and component rewrite in `static/styles/main.scss` /
  `_variables.scss`: white form-paper surfaces replace the dark
  road-palette system; Public Sans + IBM Plex Mono replace IBM Plex Sans +
  JetBrains Mono; hazard-yellow/navy/verdict-red replace the clay/rust
  accent; ruled ledger tables and a numbered code legend replace the
  stat-card hero and card-grid "What it detects" section.
- Every template rebuilt: `navbar.html` (hazard band + hazard-stripe
  boundary, "FORM PC-1" tag), `home.html` (report cover sheet, summary
  panel, legend table with clarifying note), `classify.html` (registration
  corners on the dropzone, evidence-tile samples), `model.html` (mostly a
  visual reskin, its 4-step + architecture-table structure already fit),
  `result.html` (certification stamp with signature impress animation,
  exhibit frame, severity checklist).
- `favicon.svg` recolored to navy-on-hazard-yellow.
- `backend.py`'s model loading, prediction, and error-guard logic are
  untouched; only the `cracking` sample's `file` path was updated from
  `samples/cracking.png` to `samples/cracking.jpg` to point at the correct
  bundled asset.
- (Finish review + fix batch, same pass) Removed two banned kicker badges;
  corrected a hard-offset button shadow to a real soft shadow; removed the
  green/amber severity-color tokens so verdict-red stays the only
  semantic accent beyond navy; wired the previously-unused hazard-stripe
  mixin into the navbar boundary; added the 13-classes/5-types clarifying
  line on the home page legend.

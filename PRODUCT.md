# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: technical evaluators, mostly on desktop, viewing the live demo to assess the from-scratch Keras CNN and how Glyph presents ML work (prospective clients, recruiters, fellow developers). Secondary/aspirational: someone could hypothetically use it as a field pavement inspector, but the product is not built, tuned, or validated for that operational use today.

## Product Purpose

Demonstrates a custom-trained Keras CNN that classifies pavement distress (type + severity) from a single uploaded photo. Exists as an ML/portfolio demo for Glyph (Muaz Alagroudi), not as a tool anyone operationally depends on. Success means a visitor understands what the model does, sees it produce a real prediction on a real photo, and can inspect the architecture behind it.

## Positioning

Turns pavement condition assessment into a single-photo classification (distress type + severity, in seconds) instead of measurement-based manual inspection or dedicated survey software, using nothing more than a photo and a lightweight CNN.

## Operating Context

Live demo at glyphtechnology.cloud/Pavescan/ (Glyph's shared demo hosting, not this product's own domain). Runs as a Flask app (`backend.py`) behind gunicorn under pm2 (`distress-demo-api`), no auth, no persistence beyond the single most-recently-uploaded image on disk. Visitors browse on a normal desktop or mobile web browser; no outdoor/field usage is assumed at this stage.

## Capabilities and Constraints

- Upload a JPG/PNG of a road surface → `/predict` → the Keras CNN (`distressCNN.keras`, 1,626,253 params, 7-layer sequential architecture) returns one of 13 classes: pothole, patch, "Longitudinal-traverse" cracking, and "Ravelling and weathering" each split into low/medium/high severity, plus a single undifferentiated "Rutting" class with no severity split.
- Preprocessing: grayscale, resize to 64×64, normalize to 0–1.
- Only the top predicted class label is shown; no confidence score or probability breakdown is surfaced.
- No user accounts, no history of past uploads; `static/uploaded_image.png` is overwritten by each new prediction.
- Whether this needs to support real field/inspection use one day (accuracy guarantees, offline mode, batch upload, confidence scores) is an open, undecided question; current scope is demo-only.

## Brand Commitments

"Distress Detection" (demo path name "PaveScan" / `/Pavescan`), built by Muaz Alagroudi (Glyph). The footer credit "Built with a Keras sequential CNN by Muaz Alagroudi" with email/LinkedIn/GitHub links is an existing authorship commitment to preserve.

Visual direction (confirmed 2026-08-22): dark, technical "engineering console" aesthetic, sleek and professional rather than literal terminal/hacker chrome. JetBrains Mono for data/labels/numbers, IBM Plex Sans for headings/body, a precise cyan accent. See DESIGN.md for the full system.

## Evidence on Hand

- The trained model (`distressCNN.keras`) is real; predictions shown in the demo are genuine, not staged or mocked. Three of the images from `image samples/` are now surfaced directly on `/classify` as "try a sample" thumbnails (`static/samples/`, wired through `/classify/sample/<name>`), so a visitor can see a real prediction without sourcing their own photo.
- No case studies, testimonials, or real-world field-deployment data exist. Future work must not imply operational validation that hasn't happened.

## Product Principles

1. The demo's credibility rests on being genuinely functional, never fake or exaggerate a prediction, an accuracy figure, or a capability the model doesn't have.
2. Design for a technical evaluator on desktop first; don't over-invest in field/outdoor mobile ergonomics unless the product's purpose actually shifts toward real inspection use.
3. Keep the model's real constraints visible rather than hidden, the `/model` page's honest layer-by-layer breakdown is a feature of the demo, not a limitation to paper over.
4. This is a portfolio artifact for Glyph: changes should keep reflecting well on Glyph's craft, not only on the model's accuracy.

## Accessibility & Inclusion

No product-specific requirement established beyond ordinary web accessibility (keyboard operability, color contrast), consistent with a desktop-first technical-evaluator audience.

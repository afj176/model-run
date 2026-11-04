# Matrix workspace design QA

Source: `C:/Users/afj17/.codex/generated_images/01a1a49e-42e2-71b2-8ed3-e99e56e1e7e1/exec-50a8aa1a-db75-4086-a72d-0d856380a5bc.png` (1488×1058); implementation `reports/screenshots/matrix-attention.png` (default browser viewport, real Qwen L12/H7 selection); mobile `reports/screenshots/matrix-mobile.png`. The source was generated with illustrative 64×64 data; implementation screenshot uses a real 11×11 matrix, so density and numeric content intentionally differ. An earlier same-sized desktop browser check was captured during iteration, but the final file uses the browser's actual viewport.

Findings: selected design's warm chrome, dark stage, 3D value bars, pink selection, linked inspector and heatmap, ordinal replay are implemented. Remaining P2 fidelity gap: lighting, camera, matrix navigation and layout still lack the reference's richer immersive presentation; no fabricated masking or aggregation is shown because those data are not recorded. The pictured detail inset and full-matrix overview are not implemented. Source and implementation were opened together, though the final screenshot has a different viewport, so pixel-fidelity evaluation is incomplete.

Tests: real Qwen attention 11×11, selection query 5/key 2 = 0.0481824, matching inspector and heatmap; keyboard arrow advances selection; activation prefix marks missing cell and column paging preserves offsets; mobile document had no horizontal overflow after fixing timeline minimum width; browser console errors empty. Frontend tests cover bounded windows, data semantics, missing and non-finite data, attention range, and transport. The 32×32 window, leading-index-zero limitation, and no inferred mask are disclosed.

final result: blocked

Reason: incomplete same-size final visual comparison and remaining P2 fidelity differences. This is a functional first implementation, not a claim of an exact match to the concept.

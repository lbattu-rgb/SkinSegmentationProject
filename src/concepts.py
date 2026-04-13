import cv2
import numpy as np


def _normalize_score(value, low, high):
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _resize_image(image_rgb, size):
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)


def _build_regions(mask_binary):
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask_binary, kernel, iterations=1)
    eroded = cv2.erode(mask_binary, kernel, iterations=1)
    perimeter = np.clip(dilated - eroded, 0, 1).astype(np.uint8)

    outer_ring = np.clip(cv2.dilate(mask_binary, kernel, iterations=3) - dilated, 0, 1).astype(np.uint8)
    inner_core = cv2.erode(mask_binary, kernel, iterations=3).astype(np.uint8)
    return perimeter, outer_ring, inner_core


def _safe_mean(values):
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _join_phrases(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def analyze_prediction_concepts(image, mean_pred, uncertainty, threshold=0.5):
    image_rgb = np.array(image.convert("RGB"))
    size = mean_pred.shape[0]
    image_rgb = _resize_image(image_rgb, size)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask_binary = (mean_pred > threshold).astype(np.uint8)

    mask_area = float(mask_binary.mean())
    perimeter, outer_ring, inner_core = _build_regions(mask_binary)

    boundary_pixels = perimeter.astype(bool)
    outer_pixels = outer_ring.astype(bool)
    core_pixels = inner_core.astype(bool)
    lesion_pixels = mask_binary.astype(bool)

    lesion_intensity = _safe_mean(gray[lesion_pixels])
    outer_intensity = _safe_mean(gray[outer_pixels])
    boundary_contrast = abs(lesion_intensity - outer_intensity)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    boundary_edge_strength = _safe_mean(gradient_mag[boundary_pixels])

    local_mean = cv2.GaussianBlur(gray, (9, 9), 0)
    local_sq_mean = cv2.GaussianBlur(gray ** 2, (9, 9), 0)
    local_variance = np.clip(local_sq_mean - local_mean ** 2, 0.0, None)
    outer_texture = _safe_mean(local_variance[outer_pixels])

    total_uncertainty = float(uncertainty.mean())
    perimeter_uncertainty = _safe_mean(uncertainty[boundary_pixels])
    outer_uncertainty = _safe_mean(uncertainty[outer_pixels])
    core_uncertainty = _safe_mean(uncertainty[core_pixels])

    num_components, _ = cv2.connectedComponents(mask_binary)
    fragment_count = max(0, num_components - 1)

    concept_scores = {
        "low contrast boundary": _normalize_score(0.16 - boundary_contrast, 0.0, 0.16),
        "fuzzy lesion edge": _normalize_score(0.12 - boundary_edge_strength, 0.0, 0.12),
        "background skin texture confusion": min(
            1.0,
            0.65 * _normalize_score(outer_texture, 0.002, 0.02)
            + 0.35 * _normalize_score(outer_uncertainty, 0.01, 0.08),
        ),
        "fragmented prediction": min(
            1.0,
            0.7 * _normalize_score(fragment_count, 1, 5)
            + 0.3 * _normalize_score(total_uncertainty, 0.01, 0.08),
        ),
        "high perimeter uncertainty": min(
            1.0,
            0.75 * _normalize_score(perimeter_uncertainty, 0.01, 0.08)
            + 0.25 * _normalize_score(perimeter_uncertainty - core_uncertainty, 0.0, 0.05),
        ),
    }

    ranked_concepts = sorted(concept_scores.items(), key=lambda item: item[1], reverse=True)
    active_concepts = [name for name, score in ranked_concepts if score >= 0.35][:3]

    if mask_area < 0.01:
        trust_summary = "Prediction is very limited, so the mask should be treated cautiously."
    elif total_uncertainty < 0.01 and perimeter_uncertainty < 0.015:
        trust_summary = "Prediction looks stable, with uncertainty concentrated at a low level across the lesion."
    elif perimeter_uncertainty > core_uncertainty:
        trust_summary = "Region-aware trust is strongest in the lesion core and weaker near the outer boundary."
    else:
        trust_summary = "Prediction is moderately stable, but some regions deserve manual review."

    if active_concepts:
        explanation = f"Model uncertainty is likely driven by {_join_phrases(active_concepts)}."
    else:
        explanation = "No strong visual failure mode was triggered by the post-hoc concept rules."

    if mask_area < 0.01:
        implication = "This suggests the model may not have found a clear enough lesion region to outline confidently."
    elif perimeter_uncertainty > core_uncertainty:
        implication = "This suggests the model may struggle to accurately define lesion boundaries in this case."
    elif total_uncertainty >= 0.03:
        implication = "This suggests the full mask should be reviewed carefully before trusting the lesion extent."
    else:
        implication = "This suggests the lesion outline is relatively stable, with only limited areas needing extra review."

    return {
        "concept_scores": ranked_concepts,
        "active_concepts": active_concepts,
        "summary": trust_summary,
        "explanation": explanation,
        "implication": implication,
        "metrics": {
            "boundary_contrast": boundary_contrast,
            "boundary_edge_strength": boundary_edge_strength,
            "outer_texture": outer_texture,
            "mean_uncertainty": total_uncertainty,
            "perimeter_uncertainty": perimeter_uncertainty,
            "core_uncertainty": core_uncertainty,
            "fragment_count": fragment_count,
            "mask_area": mask_area,
        },
    }

#!/usr/bin/env python3

import json
import re
from pathlib import Path

import yaml


CATEGORY_MAPPING = {
    "Basic Info": ["basic_info", "Basic Info"],
    "Technical Features": ["technical_features", "technical_characteristics", "Technical Features"],
    "Performance Metrics": ["performance_metrics", "performance", "Performance Metrics"],
    "Milestone Significance": ["milestone_significance", "milestones", "Milestone Significance"],
    "Business Info": ["business_info", "commercial_info", "Business Info"],
    "Competition & Ecosystem": ["competition_ecosystem", "competition", "Competition & Ecosystem"],
    "History": ["history", "History"],
    "Market Positioning": ["market_positioning", "market", "Market Positioning"],
}

SUMMARY_FIELDS = ["summary", "recommended_next_step"]
INTERNAL_FIELDS = {"_source_file"}
INTERNAL_OR_CATEGORY_KEYS = {
    "basic_info",
    "technical_features",
    "technical_characteristics",
    "performance_metrics",
    "performance",
    "milestone_significance",
    "milestones",
    "business_info",
    "commercial_info",
    "competition_ecosystem",
    "competition",
    "history",
    "market_positioning",
    "market",
}


def repo_root_from(script_dir):
    marker = ".scratch"
    parts = script_dir.parts
    if marker in parts:
        return Path(*parts[: parts.index(marker)])
    return script_dir


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = repo_root_from(SCRIPT_DIR)
OUTLINE_PATH = SCRIPT_DIR / "outline.yaml"
FIELDS_PATH = SCRIPT_DIR / "fields.yaml"
REPORT_PATH = SCRIPT_DIR / "report.md"


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def slugify(text):
    slug = re.sub(r"[^a-z0-9 -]", "", str(text).lower())
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug or "item"


def item_slug(name):
    slug = re.sub(r"[^A-Za-z0-9_ ]", "", str(name))
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug


def human_label(name):
    return str(name).replace("_", " ").strip().title()


def display_category_name(category):
    aliases = CATEGORY_MAPPING.get(category, [])
    if category in {"basic_info", "Basic Info"}:
        return "Basic Info"
    if category in {"technical_features", "technical_characteristics", "Technical Features"}:
        return "Technical Features"
    if category in {"performance_metrics", "performance", "Performance Metrics"}:
        return "Performance Metrics"
    if category in {"milestone_significance", "milestones", "Milestone Significance"}:
        return "Milestone Significance"
    if category in {"business_info", "commercial_info", "Business Info"}:
        return "Business Info"
    if category in {"competition_ecosystem", "competition", "Competition & Ecosystem"}:
        return "Competition & Ecosystem"
    if category in {"history", "History"}:
        return "History"
    if category in {"market_positioning", "market", "Market Positioning"}:
        return "Market Positioning"
    for canonical, mapped in CATEGORY_MAPPING.items():
        if category == canonical or category in mapped or any(alias in aliases for alias in mapped):
            return canonical
    return human_label(category)


def category_aliases(category):
    aliases = set(CATEGORY_MAPPING.get(category, []))
    aliases.add(category)
    display = display_category_name(category)
    aliases.add(display)
    aliases.update(CATEGORY_MAPPING.get(display, []))
    return aliases


def contains_uncertain(value):
    if isinstance(value, str):
        return "[uncertain]" in value
    if isinstance(value, dict):
        return any(contains_uncertain(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_uncertain(v) for v in value)
    return False


def empty_value(value):
    return value is None or value == "" or value == [] or value == {}


def traverse_find(obj, field_name):
    if isinstance(obj, dict):
        if field_name in obj:
            return obj[field_name]
        for value in obj.values():
            found = traverse_find(value, field_name)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = traverse_find(item, field_name)
            if found is not None:
                return found
    return None


def lookup_field(data, field_name, category):
    if isinstance(data, dict) and field_name in data:
        return data[field_name]
    if isinstance(data, dict):
        for alias in category_aliases(category):
            category_value = data.get(alias)
            if isinstance(category_value, dict) and field_name in category_value:
                return category_value[field_name]
    return traverse_find(data, field_name)


def should_skip(field_name, value, uncertain_fields):
    return (
        field_name in uncertain_fields
        or empty_value(value)
        or contains_uncertain(value)
    )


def flatten_defined_fields(fields_doc):
    categories = []
    defined = set()
    for category_doc in fields_doc.get("field_categories", []):
        category = category_doc.get("category", "Other")
        fields = []
        for field_doc in category_doc.get("fields", []):
            name = field_doc.get("name")
            if not name:
                continue
            defined.add(name)
            fields.append(name)
        categories.append((category, fields))
    return categories, defined


def collect_extra_fields(obj, defined_fields, prefix=None):
    extras = {}
    prefix = prefix or []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in INTERNAL_FIELDS or key == "uncertain":
                continue
            if not prefix and key in INTERNAL_OR_CATEGORY_KEYS:
                collect = collect_extra_fields(value, defined_fields, prefix)
                extras.update(collect)
                continue
            if key in defined_fields:
                continue
            if isinstance(value, dict):
                child_extras = collect_extra_fields(value, defined_fields, prefix + [key])
                if child_extras:
                    extras.update(child_extras)
                elif not empty_value(value):
                    extras[".".join(prefix + [key])] = value
            elif isinstance(value, list):
                extras[".".join(prefix + [key])] = value
                for item in value:
                    if isinstance(item, dict):
                        extras.update(collect_extra_fields(item, defined_fields, prefix + [key]))
            else:
                extras[".".join(prefix + [key])] = value
    elif isinstance(obj, list):
        for item in obj:
            extras.update(collect_extra_fields(item, defined_fields, prefix))
    return extras


def format_scalar(value):
    text = str(value)
    if len(text) > 100:
        return text.replace("\n", "<br>")
    return text


def format_value(value):
    if isinstance(value, dict):
        parts = []
        for key, child in value.items():
            if empty_value(child) or contains_uncertain(child):
                continue
            parts.append(f"{human_label(key)}: {format_value(child)}")
        return "; ".join(parts)
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            lines = []
            for item in value:
                parts = []
                for key, child in item.items():
                    if empty_value(child) or contains_uncertain(child):
                        continue
                    parts.append(f"{human_label(key)}: {format_value(child)}")
                if parts:
                    lines.append("- " + " | ".join(parts))
            return "\n".join(lines)
        simple = [format_value(item) for item in value if not empty_value(item) and not contains_uncertain(item)]
        if len(simple) <= 3 and sum(len(item) for item in simple) <= 160:
            return ", ".join(simple)
        return "\n".join(f"- {item}" for item in simple)
    return format_scalar(value)


def first_sentence(value):
    if isinstance(value, (dict, list)):
        text = format_value(value)
    else:
        text = str(value)
    text = " ".join(text.replace("\n", " ").split())
    if len(text) <= 180:
        return text
    return text[:177].rstrip() + "..."


def append_field(lines, name, value):
    formatted = format_value(value)
    if "\n" in formatted:
        lines.append(f"**{human_label(name)}:**")
        lines.append("")
        lines.append(formatted)
    else:
        lines.append(f"**{human_label(name)}:** {formatted}")
    lines.append("")


def item_name(data, fallback):
    value = traverse_find(data, "item_name")
    if value:
        return value
    value = traverse_find(data, "name")
    if value:
        return value
    return fallback


def ordered_items(outline, results_dir):
    by_slug = {path.stem: path for path in results_dir.glob("*.json")}
    ordered = []
    seen = set()
    for item in outline.get("items", []):
        name = item.get("name")
        if not name:
            continue
        slug = item_slug(name)
        path = by_slug.get(slug)
        if path:
            ordered.append(path)
            seen.add(path)
    for path in sorted(results_dir.glob("*.json")):
        if path not in seen:
            ordered.append(path)
    return ordered


def main():
    outline = load_yaml(OUTLINE_PATH)
    fields_doc = load_yaml(FIELDS_PATH)
    results_dir = resolve_project_path(outline.get("execution", {}).get("output_dir", SCRIPT_DIR / "results"))
    categories, defined_fields = flatten_defined_fields(fields_doc)
    paths = ordered_items(outline, results_dir)
    entries = []
    for path in paths:
        data = load_json(path)
        name = item_name(data, path.stem)
        entries.append((name, path, data, set(data.get("uncertain") or [])))

    lines = [
        "# Hibiki Realtime Output Strategy Research Report",
        "",
        "## Table of Contents",
        "",
    ]

    for index, (name, _, data, uncertain_fields) in enumerate(entries, start=1):
        summary_bits = []
        for field_name in SUMMARY_FIELDS:
            value = lookup_field(data, field_name, "")
            if should_skip(field_name, value, uncertain_fields):
                continue
            summary_bits.append(f"{human_label(field_name)}: {first_sentence(value)}")
        suffix = f" - {' | '.join(summary_bits)}" if summary_bits else ""
        lines.append(f"{index}. [{name}](#{slugify(name)}){suffix}")
    lines.append("")
    lines.append("## Detailed Content")
    lines.append("")

    for name, path, data, uncertain_fields in entries:
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"_Source: `{path.name}`_")
        lines.append("")

        rendered_fields = set()
        for category, field_names in categories:
            category_lines = []
            for field_name in field_names:
                value = lookup_field(data, field_name, category)
                if should_skip(field_name, value, uncertain_fields):
                    continue
                append_field(category_lines, field_name, value)
                rendered_fields.add(field_name)
            if category_lines:
                lines.append(f"#### {display_category_name(category)}")
                lines.append("")
                lines.extend(category_lines)

        extras = collect_extra_fields(data, defined_fields)
        extra_lines = []
        for field_name in sorted(extras):
            value = extras[field_name]
            leaf_name = field_name.split(".")[-1]
            if should_skip(leaf_name, value, uncertain_fields):
                continue
            append_field(extra_lines, field_name, value)
        if extra_lines:
            lines.append("#### Other Info")
            lines.append("")
            lines.extend(extra_lines)

        if uncertain_fields:
            lines.append("#### Uncertain Fields")
            lines.append("")
            for field_name in sorted(uncertain_fields):
                lines.append(f"- {field_name}")
            lines.append("")

    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Canonical label vocab + type->field-group rules for the Stage-1 target.

The annotations are already normalised when the frame folders are built, but these
maps are kept here as the single source of truth and as a defensive fallback in the
target builder (unknown values -> Other/Unknown rather than crashing).
"""

# ---- type -> attribute field group ----
DYNAMIC_TYPES = {
    "Car", "Truck", "Bus", "Motorcyclist", "Construction vehicle", "Pedestrian",
    "Cyclist", "Scooter rider", "Emergency vehicle", "School bus", "Other",
}
CONTROL_TYPES = {"Traffic light", "Sign"}          # has `content`
TYPE_ONLY = {"Stop line", "Cross walk", "Bump", "Temporary control"}  # type+rank only
LOCATION_ONLY = {"Object", "Animal"}               # location only

ALL_TYPES = DYNAMIC_TYPES | CONTROL_TYPES | TYPE_ONLY | LOCATION_ONLY


def fields_for_type(t):
    """Return the ordered list of attribute fields valid for a given type."""
    if t in DYNAMIC_TYPES:
        return ["location", "intention", "state"]
    if t in CONTROL_TYPES:
        return ["content"]
    if t in LOCATION_ONLY:
        return ["location"]
    return []  # TYPE_ONLY and unknown -> nothing extra


def norm_type(t):
    return t if t in ALL_TYPES else "Other"


# ---- context enums (reference vocabularies). ``norm_context`` maps missing values to "Unknown". ----
WEATHER = {"Sunny", "Cloudy", "Rainy", "Fog", "Fair"}
DAYTIME = {"Day", "Night"}
VISIBILITY = {"Clear", "Reduced", "Limited", "Poor"}
ROAD = {"Mid-block", "Intersection", "Diverge", "Merge", "Roundabout"}
# scenario is an open-ish set after cleaning; we pass it through verbatim.

CONTEXT_KEYS = ["weather", "daytime", "visibility", "scenario", "road"]


def norm_context(ctx):
    out = {}
    for k in CONTEXT_KEYS:
        v = (ctx or {}).get(k)
        out[k] = v if v not in (None, "") else "Unknown"
    return out

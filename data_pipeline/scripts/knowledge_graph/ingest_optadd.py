"""
将 evaluation/Vis/data/optadd_supplement_raw_rows.csv 中的 OptAdd
结构化数据并入光电知识图谱 (Neo4j)。

设计要点:
- 不走 LLM 抽取: 数据已结构化, 直接映射列到 Entity/Relation。
- 实体 ID 由 Entity.generate_id(name, type) 确定, 因此跨行的同名
  Donor/Acceptor/Metric 节点会自然去重 (MERGE)。
- 关系 ID 在 hash 中加入 record_id, 让每行的 HAS_PROPERTY/USES
  在 Neo4j 中独立存在 (否则同一 Device 对同一 Metric 只剩一条边)。
- 幂等: 重复运行不会重复插入。
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))  # data_pipeline/

from core.knowledge_graph import Entity, EntityType, Relation, RelationType
from core.neo4j_manager import Neo4jManager

CSV_PATH = _HERE.parents[3] / "evaluation/Vis/data/optadd_supplement_raw_rows.csv"

METRIC_DEFS = {
    "pce_percent":  ("Power Conversion Efficiency",      "%"),
    "jsc_ma_cm2":   ("Short-Circuit Current Density",    "mA/cm^2"),
    "voc_v":        ("Open-Circuit Voltage",             "V"),
    "ff":           ("Fill Factor",                      ""),
}

DEVICE_PROP_COLS = [
    "da_ratio", "donor_conc_mg_ml", "acceptor_conc_mg_ml",
    "additive_conc_mg_ml", "additive_vol_pct", "additive_wt_pct",
    "solvent_main", "solvent_second", "solvent_ratio",
    "total_conc_mg_ml", "spin_speed_rpm", "film_thickness_nm",
    "anneal_temp_c", "anneal_time_min", "anneal_atmosphere",
    "device_structure",
]


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    return v if v else None


def _float_or_none(value: Optional[str]) -> Optional[float]:
    v = _clean(value)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _rel_id(source_id: str, target_id: str, rtype: RelationType, suffix: str = "") -> str:
    raw = f"{source_id}|{target_id}|{rtype.value}|{suffix}"
    return f"{rtype.value}_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


def _upsert_entity(
    bucket: Dict[str, Entity],
    name: str,
    etype: EntityType,
    properties: Dict,
    description: str,
    doi: Optional[str],
) -> str:
    eid = Entity.generate_id(name, etype)
    if eid in bucket:
        existing = bucket[eid]
        for k, v in properties.items():
            if v is None or v == "":
                continue
            existing.properties.setdefault(k, v)
        if doi and doi not in existing.source_papers:
            existing.source_papers.append(doi)
        if description and description not in existing.descriptions:
            existing.descriptions.append(description)
        return eid
    clean_props = {k: v for k, v in properties.items() if v not in (None, "")}
    bucket[eid] = Entity(
        id=eid,
        name=name,
        type=etype,
        properties=clean_props,
        description=description,
        source_papers=[doi] if doi else [],
        confidence=1.0,
    )
    return eid


def _build_metric_entities(bucket: Dict[str, Entity]) -> Dict[str, str]:
    """Pre-create the four shared METRIC nodes and return col -> entity_id."""
    out = {}
    for col, (metric_name, unit) in METRIC_DEFS.items():
        eid = _upsert_entity(
            bucket,
            name=metric_name,
            etype=EntityType.METRIC,
            properties={"unit": unit, "dataset_sources": ["OptAdd"]} if unit else {"dataset_sources": ["OptAdd"]},
            description=f"{metric_name} metric (shared node).",
            doi=None,
        )
        out[col] = eid
    return out


def build(rows: List[Dict]) -> tuple[List[Entity], List[Relation]]:
    entities: Dict[str, Entity] = {}
    relations: List[Relation] = []

    metric_ids = _build_metric_entities(entities)

    for row in rows:
        record_id = _clean(row.get("record_id"))
        doi = _clean(row.get("source"))
        if not record_id:
            continue

        donor_name = _clean(row.get("donor_name"))
        acceptor_name = _clean(row.get("acceptor_name"))
        additive_id = _clean(row.get("additive_id"))

        # --- entities ---
        donor_id = _upsert_entity(
            entities,
            name=donor_name,
            etype=EntityType.MATERIAL,
            properties={
                "smiles": _clean(row.get("donor_smiles")),
                "material_type": _clean(row.get("donor_type")),
                "role": "donor",
                "dataset": "OptAdd",
            },
            description=f"Donor {donor_name} (OptAdd dataset).",
            doi=doi,
        ) if donor_name else None

        acceptor_id = _upsert_entity(
            entities,
            name=acceptor_name,
            etype=EntityType.MATERIAL,
            properties={
                "smiles": _clean(row.get("acceptor_smiles")),
                "material_type": _clean(row.get("acceptor_type")),
                "role": "acceptor",
                "dataset": "OptAdd",
            },
            description=f"Acceptor {acceptor_name} (OptAdd dataset).",
            doi=doi,
        ) if acceptor_name else None

        additive_entity_id = None
        if additive_id:
            additive_name = f"OptAdd-{additive_id}"
            additive_entity_id = _upsert_entity(
                entities,
                name=additive_name,
                etype=EntityType.MATERIAL,
                properties={
                    "additive_id": additive_id,
                    "role": "additive",
                    "dataset": "OptAdd",
                },
                description=f"Additive {additive_id} from OptAdd dataset.",
                doi=doi,
            )

        # record_id is unique per paper in OptAdd, not globally — include a
        # short DOI hash so cross-paper records don't collide.
        doi_short = hashlib.md5(doi.encode()).hexdigest()[:8] if doi else "nodoi"
        device_name = f"OptAdd-{doi_short}-{record_id}"
        device_props = {col: _clean(row.get(col)) for col in DEVICE_PROP_COLS}
        device_props["record_id"] = record_id
        device_props["dataset"] = "OptAdd"
        device_id = _upsert_entity(
            entities,
            name=device_name,
            etype=EntityType.DEVICE,
            properties=device_props,
            description=(
                f"OPV device record {record_id}: {donor_name}/{acceptor_name}"
                + (f" with additive {additive_id}" if additive_id else "")
                + (f", DOI {doi}" if doi else "")
                + "."
            ),
            doi=doi,
        )

        paper_id = None
        if doi:
            paper_id = _upsert_entity(
                entities,
                name=doi,
                etype=EntityType.PAPER,
                properties={"doi": doi, "dataset": "OptAdd"},
                description=f"Paper (DOI {doi}) -- source of OptAdd record(s).",
                doi=doi,
            )

        # --- relations ---
        def add_rel(src, tgt, rtype, props, desc, suffix=""):
            if not src or not tgt:
                return
            rid = _rel_id(src, tgt, rtype, suffix=suffix)
            relations.append(Relation(
                id=rid,
                source_id=src,
                target_id=tgt,
                type=rtype,
                properties={k: v for k, v in props.items() if v not in (None, "")},
                description=desc,
                source_papers=[doi] if doi else [],
                confidence=1.0,
            ))

        if donor_id:
            add_rel(
                device_id, donor_id, RelationType.USES,
                {"concentration_mg_ml": _float_or_none(row.get("donor_conc_mg_ml"))},
                f"Device {device_name} uses donor {donor_name}.",
                suffix=f"{record_id}|donor",
            )
        if acceptor_id:
            add_rel(
                device_id, acceptor_id, RelationType.USES,
                {"concentration_mg_ml": _float_or_none(row.get("acceptor_conc_mg_ml"))},
                f"Device {device_name} uses acceptor {acceptor_name}.",
                suffix=f"{record_id}|acceptor",
            )
        if additive_entity_id:
            add_rel(
                device_id, additive_entity_id, RelationType.USES,
                {
                    "concentration_mg_ml": _float_or_none(row.get("additive_conc_mg_ml")),
                    "vol_pct": _float_or_none(row.get("additive_vol_pct")),
                    "wt_pct": _float_or_none(row.get("additive_wt_pct")),
                },
                f"Device {device_name} uses additive {additive_id}.",
                suffix=f"{record_id}|additive",
            )
            add_rel(
                additive_entity_id, device_id, RelationType.ENHANCES,
                {
                    "delta_pce": _float_or_none(row.get("delta_pce")),
                    "pce_baseline": _float_or_none(row.get("pce_baseline")),
                    "morphology_effect": _clean(row.get("morphology_effect")),
                    "mechanism": _clean(row.get("mechanism")),
                },
                f"Additive {additive_id} enhances device {device_name}.",
                suffix=record_id,
            )

        for col, (metric_name, unit) in METRIC_DEFS.items():
            value = _float_or_none(row.get(col))
            if value is None:
                continue
            metric_id = metric_ids[col]
            props = {"value": value}
            if unit:
                props["unit"] = unit
            add_rel(
                device_id, metric_id, RelationType.HAS_PROPERTY,
                props,
                f"Device {device_name} {metric_name} = {value}{(' ' + unit) if unit else ''}.",
                suffix=f"{record_id}|{col}",
            )

        if paper_id:
            add_rel(
                paper_id, device_id, RelationType.CONTAINS,
                {"dataset": "OptAdd"},
                f"Paper {doi} contains device record {record_id}.",
                suffix=record_id,
            )

    return list(entities.values()), relations


def _reconcile_with_existing(mgr: Neo4jManager,
                             entities: List[Entity],
                             relations: List[Relation]) -> tuple[List[Entity], List[Relation], int]:
    """Avoid collisions with the existing `Entity.name` uniqueness constraint.

    For each prepared entity, look up an existing node with the same name. If
    one exists (regardless of type) reuse its id: rewrite relations that
    reference the prepared id, and drop the prepared entity from the insert
    batch. Otherwise keep the entity as-is.
    """
    names = [e.name for e in entities]
    existing = mgr.execute_custom_query(
        "UNWIND $names AS n MATCH (e:Entity {name: n}) RETURN e.name AS name, e.id AS id, e.type AS type",
        {"names": names},
    )
    name_to_existing = {row["name"]: row for row in existing}

    id_remap: Dict[str, str] = {}
    keep: List[Entity] = []
    reused = 0
    for ent in entities:
        hit = name_to_existing.get(ent.name)
        if hit and hit["id"] != ent.id:
            id_remap[ent.id] = hit["id"]
            reused += 1
        else:
            keep.append(ent)

    if id_remap:
        for rel in relations:
            if rel.source_id in id_remap:
                rel.source_id = id_remap[rel.source_id]
            if rel.target_id in id_remap:
                rel.target_id = id_remap[rel.target_id]
            # rel.id was hashed off the old ids, but it only needs to be
            # globally unique; recomputing isn't required.

    return keep, relations, reused


def main() -> int:
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return 1

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} OptAdd rows from {CSV_PATH}")

    entities, relations = build(rows)
    print(f"Prepared {len(entities)} entities, {len(relations)} relations.")

    mgr = Neo4jManager()
    try:
        before = mgr.get_statistics()
        print(f"Before: {before['total_entities']} entities, {before['total_relations']} relations")

        entities, relations, reused = _reconcile_with_existing(mgr, entities, relations)
        print(f"Reused {reused} existing nodes by name; inserting {len(entities)} new entities.")

        n_ent = mgr.batch_insert_entities(entities)
        n_rel = mgr.batch_insert_relations(relations)
        print(f"Inserted/merged: {n_ent} entities, {n_rel} relations")

        after = mgr.get_statistics()
        print(f"After:  {after['total_entities']} entities, {after['total_relations']} relations")
        print(f"Net delta: +{after['total_entities'] - before['total_entities']} entities, "
              f"+{after['total_relations'] - before['total_relations']} relations")
    finally:
        mgr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
